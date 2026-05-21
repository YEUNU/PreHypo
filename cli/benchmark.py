import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from core.config import RAGConfig
from core.vllm_client import get_llm_client
from models.prehypo.graphrag import GraphRAG
from models.naive.naive_rag import NaiveRAG
from utils.io import _safe_float
from utils.metrics import evaluate_financebench_response, evaluate_multihoprag_response
from utils.reporting import _write_model_report_artifacts


logger = logging.getLogger("PreHypo")


_BOXED_RE = re.compile(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}")
_FINAL_LABEL_RE = re.compile(
    r"(?is)(?:final\s+answer|@@ANSWER|answer)\s*:?\s*(.+?)(?:\n\n|\Z)"
)


def _extract_final_answer(answer_text: str) -> str:
    """Extract the final answer from a model response that may contain
    step-by-step reasoning. Order: \\boxed{...} > 'Final Answer:' marker >
    last 300 chars. Avoids substring-matching the reasoning body for
    abstain detection.
    """
    if not answer_text:
        return ""
    boxed = _BOXED_RE.findall(answer_text)
    if boxed:
        return boxed[-1].strip()
    matches = _FINAL_LABEL_RE.findall(answer_text)
    if matches:
        return matches[-1].strip()[:400]
    return answer_text[-300:].strip()


def _build_benchmark_query(query: str, item: dict[str, Any]) -> str:
    """Return the user-facing query as-is.

    The previous implementation appended `[Benchmark Output Format]` blocks
    that forced verbose CoT inside `\\boxed{}`. That suffix leaked into
    retrieval embeddings as noise and collided with the citation-first
    answer format. The judge prompt extracts `\\boxed{}` / `Final Answer:`
    internally, so the scaffolding adds no signal upstream.
    """
    _ = item  # kept for signature stability; type detection no longer alters the query.
    return query


def _apply_judge_label(result_item: dict[str, Any], is_financebench: bool, is_multihoprag: bool) -> None:
    """Derive answer_attempted / financebench_label (+ hallucination fallback)
    from the (possibly just-patched) llm_judge_score. Idempotent, so it can be
    re-run after the OpenAI batch resolves a deferred judge score.
    """
    if not (is_financebench or is_multihoprag):
        return
    from utils.abstain import financebench_label, is_abstain
    answer_text = str(result_item.get("answer", "") or "")
    has_error = bool(result_item.get("error"))
    # Detect abstain on the EXTRACTED final answer (\\boxed{} / 'Final Answer:'),
    # not the full CoT body which often uses 'insufficient evidence' mid-reason.
    final_answer = _extract_final_answer(answer_text).lower()
    abstained = is_abstain(final_answer)
    judge_score = _safe_float(result_item.get("llm_judge_score", 0.0), 0.0)
    # Judge override: a score >= 0.5 means a usable answer regardless of phrasing.
    if has_error:
        answer_attempted = 0.0
    elif judge_score >= 0.5:
        answer_attempted = 1.0
    else:
        answer_attempted = 0.0 if abstained else 1.0
    result_item["answer_attempted"] = answer_attempted
    result_item["final_answer_extracted"] = final_answer[:300]
    if not isinstance(result_item.get("hallucination"), (int, float)):
        result_item["hallucination"] = 1.0 if (answer_attempted > 0.0 and judge_score < 1.0) else 0.0
    result_item["financebench_label"] = financebench_label(judge_score, final_answer)


async def run_benchmark(
    queries_file: str,
    strategy: str,
    model_id: str,
    is_batch: bool = False,
    sample_companies: Optional[list[str]] = None,
    corpus_tag: str = "default",
    output_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    seed: Optional[int] = None,
):
    """Run benchmark once. When `seed` is provided, RAGConfig.LLM_SEED is set
    so all vLLM/OpenAI chat.completions calls in this run use that seed, and
    the output directory gets a `seed_<S>` subdir to avoid clobbering other
    seeds' results. Multi-seed orchestration lives in run_benchmark_multi_seed.
    """
    if seed is not None:
        RAGConfig.LLM_SEED = int(seed)

    try:
        if strategy in ("prehypo", "hyporeflect"):
            # Legacy "hyporeflect" alias keeps the EMNLP-rebuttal ablation
            # pointed at the pre-built HY_<corpus>_* labels without re-indexing.
            engine = GraphRAG(strategy=strategy, corpus_tag=corpus_tag)
        elif strategy == "naive":
            engine = NaiveRAG(strategy=strategy, corpus_tag=corpus_tag)
        elif strategy == "hoprag":
            from models.hoprag.hoprag_adapter import HopRAGAdapter

            engine = HopRAGAdapter(model_id=model_id, corpus_tag=corpus_tag)
        elif strategy == "ms_graphrag":
            from models.ms_graphrag.ms_adapter import MSGraphRAGAdapter

            engine = MSGraphRAGAdapter(model_id=model_id, corpus_tag=corpus_tag)
        else:
            logger.error("Unknown strategy: %s", strategy)
            return None

        vllm = get_llm_client(model_id)
    except Exception as exc:
        logger.error("Failed to initialize engine for %s: %s", strategy, exc)
        return None

    if not os.path.exists(queries_file):
        logger.error("Queries file %s not found.", queries_file)
        return None

    with open(queries_file, "r", encoding="utf-8") as file:
        benchmark_data = json.load(file)

    if sample_companies:
        initial_len = len(benchmark_data)
        benchmark_data = [item for item in benchmark_data if item.get("company") in sample_companies]
        logger.info(
            "Filtering for %d sample companies: %d -> %d queries",
            len(sample_companies),
            initial_len,
            len(benchmark_data),
        )

    if limit is not None:
        benchmark_data = benchmark_data[: max(0, int(limit))]
        logger.info("--limit %d: evaluating %d queries", limit, len(benchmark_data))

    # Dataset dispatch via the per-query `dataset` marker. Both branches are
    # LLM-judge scored, so the 3-way label / abstain post-processing below is
    # shared; only the eval function (prompt + retrieval metrics) differs.
    dataset_marker = (benchmark_data[0].get("dataset", "") if benchmark_data else "").strip().lower()
    if dataset_marker == "multihoprag":
        dataset_kind = "multihoprag"
        dataset_name = "MultiHop-RAG"
    else:
        # Default to FinanceBench (legacy datasets without a marker).
        dataset_kind = "financebench"
        dataset_name = "FinanceBench"
    is_financebench = dataset_kind == "financebench"
    is_multihoprag = dataset_kind == "multihoprag"
    results = []
    category_results = {}

    logger.info(
        "Starting benchmark [%s] on %s | Queries: %d",
        strategy,
        dataset_name,
        len(benchmark_data),
    )

    if output_dir:
        results_dir = output_dir
    else:
        env_ts = os.environ.get("RAG_BENCHMARK_TIMESTAMP")
        start_timestamp = env_ts if env_ts else time.strftime("%Y%m%d_%H%M%S")
        results_dir = Path("data/results") / start_timestamp

    results_dir.mkdir(parents=True, exist_ok=True)
    model_results_dir = results_dir / strategy
    model_results_dir.mkdir(parents=True, exist_ok=True)
    ablation_results_dir = model_results_dir / corpus_tag
    ablation_results_dir.mkdir(parents=True, exist_ok=True)

    sample_suffix = "_sample" if sample_companies else ""
    output_results_dir = ablation_results_dir
    if seed is not None:
        output_results_dir = output_results_dir / f"seed_{int(seed)}"
    output_results_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_results_dir / f"{strategy}_{corpus_tag}{sample_suffix}.json"
    summary: dict[str, Any] = {}

    benchmark_concurrency = max(1, int(os.environ.get("RAG_BENCHMARK_CONCURRENCY", "4")))
    query_sem = asyncio.Semaphore(benchmark_concurrency)
    write_lock = asyncio.Lock()
    total_queries = len(benchmark_data)
    if benchmark_concurrency > 1:
        logger.info("Benchmark concurrency: %d queries in flight", benchmark_concurrency)

    # Optional OpenAI Batch API judge (opt-in via RAG_JUDGE_BATCH; OpenAI
    # EVAL_MODEL only). Collects judge prompts during the pass below, then one
    # batch resolves all scores after `gather`. Falls back to sync on failure.
    batch_collector = None
    if RAGConfig.JUDGE_BATCH and vllm is not None:
        eval_model = str(RAGConfig.EVAL_MODEL or "")
        is_openai_eval = eval_model.startswith(("gpt-", "o1", "o3", "o4", "chatgpt"))
        if is_openai_eval and RAGConfig.OPENAI_API_KEY:
            from utils.batch_judge import OpenAIBatchJudge
            batch_collector = OpenAIBatchJudge(
                eval_model,
                RAGConfig.OPENAI_API_KEY,
                poll_seconds=RAGConfig.JUDGE_BATCH_POLL_SECONDS,
            )
            logger.info("Judge: OpenAI Batch API mode (RAG_JUDGE_BATCH) — deferring %d judge calls to one batch", total_queries)
        else:
            logger.warning(
                "RAG_JUDGE_BATCH set but EVAL_MODEL '%s' is not an OpenAI model (or OPENAI_API_KEY missing); using synchronous judge.",
                eval_model,
            )

    def _recompute_and_persist() -> dict[str, Any]:
        """Rebuild the summary from `results` and write the result file +
        report artifacts. Called incrementally per query and once more after
        the batch judge patches deferred scores."""
        ref = results[-1] if results else {}
        s: dict[str, Any] = {
            "strategy": strategy,
            "corpus_tag": corpus_tag,
            "dataset": dataset_name,
            "queries_count": len(results),
            "total_queries": total_queries,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "in_progress" if len(results) < total_queries else "completed",
            "models": {
                "default": RAGConfig.DEFAULT_MODEL,
                "embedding": RAGConfig.EMBEDDING_MODEL,
                "eval": RAGConfig.EVAL_MODEL,
            },
            "ablation": {
                "table_to_text": RAGConfig.ABLATION_TABLE_TO_TEXT,
                "adaptive_chunking": RAGConfig.ABLATION_ADAPTIVE_CHUNKING,
                "rolling_summary": RAGConfig.ABLATION_ROLLING_SUMMARY,
            },
        }
        denom = len(results) or 1
        for key in ref.keys():
            if isinstance(ref.get(key), (int, float)) and not isinstance(ref.get(key), bool):
                s[f"avg_{key}"] = sum(r[key] for r in results) / denom
        cat_summaries = {}
        for cat, cat_list in category_results.items():
            cat_sum: dict[str, Any] = {"count": len(cat_list)}
            for key in ref.keys():
                if isinstance(ref.get(key), (int, float)) and not isinstance(ref.get(key), bool):
                    cat_sum[f"avg_{key}"] = sum(r[key] for r in cat_list) / (len(cat_list) or 1)
            cat_summaries[cat] = cat_sum
        s["category_summaries"] = cat_summaries

        if is_financebench or is_multihoprag:
            fb_counts = {"Correct Answer": 0, "Incorrect Answer": 0, "Refusal": 0}
            for r in results:
                label = r.get("financebench_label")
                if label in fb_counts:
                    fb_counts[label] += 1
            total = len(results) or 1
            s["financebench_correct_count"] = fb_counts["Correct Answer"]
            s["financebench_incorrect_count"] = fb_counts["Incorrect Answer"]
            s["financebench_refusal_count"] = fb_counts["Refusal"]
            s["financebench_correct_rate"] = fb_counts["Correct Answer"] / total
            s["financebench_incorrect_rate"] = fb_counts["Incorrect Answer"] / total
            s["financebench_refusal_rate"] = fb_counts["Refusal"] / total

        s["details"] = results
        # Report artifacts first (writes full traces), then a slim main JSON
        # (no interaction_trace, no internal _-prefixed keys e.g. _deferred_judge).
        try:
            _write_model_report_artifacts(s, result_file)
        except Exception as exc:
            logger.warning("Failed to write report artifacts for %s: %s", result_file, exc)
        slim_details = [
            {k: v for k, v in d.items() if k != "interaction_trace" and not k.startswith("_")}
            if isinstance(d, dict) else d
            for d in (s.get("details") or [])
        ]
        with open(result_file, "w", encoding="utf-8") as file:
            json.dump({**s, "details": slim_details}, file, indent=2, ensure_ascii=False)
        return s

    async def _process_query(idx: int, item: dict[str, Any]):
      nonlocal summary
      async with query_sem:
        original_query = item["query"]
        query = _build_benchmark_query(original_query, item)
        ground_truth = item.get("ground_truth", "")
        category = item.get("category", "Uncategorized")

        started = time.time()
        try:
            response, retrieved_sources, trace = await engine.run_workflow(query, [])
            latency = time.time() - started

            if is_multihoprag:
                metrics = await evaluate_multihoprag_response(
                    query=original_query,
                    response=response,
                    ground_truth=ground_truth,
                    retrieved_sources=retrieved_sources,
                    evidence_facts=item.get("evidence_facts", []),
                    evidence_docs=item.get("evidence_docs", []),
                    question_type=item.get("question_type", ""),
                    vllm_client=vllm,
                    batch_collector=batch_collector,
                    custom_id=str(idx),
                )
                expected_sources = {
                    "docs": item.get("evidence_docs", []),
                    "facts": item.get("evidence_facts", []),
                }
            else:
                metrics = await evaluate_financebench_response(
                    query=original_query,
                    response=response,
                    ground_truth=ground_truth,
                    retrieved_sources=retrieved_sources,
                    expected_doc=item.get("evidence_doc", ""),
                    expected_page=item.get("evidence_page"),
                    vllm_client=vllm,
                    batch_collector=batch_collector,
                    custom_id=str(idx),
                )
                expected_sources = {
                    "doc": item.get("evidence_doc", ""),
                    "page": item.get("evidence_page"),
                    "text": item.get("evidence_text", ""),
                }
            result_item = {
                "query": original_query,
                "category": category,
                "answer": response,
                "ground_truth": ground_truth,
                "expected_sources": expected_sources,
                "retrieved_sources": retrieved_sources,
                "interaction_trace": trace,
                "latency": latency,
                **metrics,
            }
        except Exception as exc:
            logger.error("Error processing query '%s': %s", original_query, exc)
            import traceback

            logger.error(traceback.format_exc())
            latency = time.time() - started
            error_text = f"{type(exc).__name__}: {exc}"

            metrics = {
                "llm_judge_score": 0.0,
                "llm_judge_reason": "runtime_error",
                "hallucination": 0.0,
                "hallucination_reason": "runtime_error",
                "hallucination_source": "runtime_error",
                "hallucination_model": str(RAGConfig.EVAL_MODEL or ""),
                "doc_match": 0.0,
                "page_match": 0.0,
            }
            if is_multihoprag:
                # Keep the same numeric keys the success path emits so the
                # summary auto-averaging stays consistent across queries.
                metrics.update({
                    "mrr@10": 0.0, "map@10": 0.0, "hits@4": 0.0, "hits@10": 0.0,
                    "evidence_doc_recall": 0.0,
                })
                expected_sources = {
                    "docs": item.get("evidence_docs", []),
                    "facts": item.get("evidence_facts", []),
                }
            else:
                expected_sources = {
                    "doc": item.get("evidence_doc", ""),
                    "page": item.get("evidence_page"),
                    "text": item.get("evidence_text", ""),
                }
            result_item = {
                "query": original_query,
                "category": category,
                "answer": f"@@ANSWER: ERROR - {error_text}",
                "ground_truth": ground_truth,
                "expected_sources": expected_sources,
                "retrieved_sources": [],
                "interaction_trace": [{"step": "error", "output": error_text}],
                "latency": latency,
                "error": error_text,
                **metrics,
            }

        if query != original_query:
            result_item["benchmark_query"] = query

        # Both FinanceBench and MultiHop-RAG are LLM-judge scored, so the
        # 3-way label / answer_attempted / abstain post-processing is shared.
        # In batch-judge mode the score here is the tentative heuristic; phase 2
        # re-runs this after the real batch score lands.
        _apply_judge_label(result_item, is_financebench, is_multihoprag)

        async with write_lock:
            results.append(result_item)
            if category not in category_results:
                category_results[category] = []
            category_results[category].append(result_item)

            error_suffix = " [ERROR]" if result_item.get("error") else ""
            print(
                f"[{strategy}] ({len(results)}/{total_queries}) [{category}]{error_suffix} "
                f"Judge: {metrics['llm_judge_score']:.1f} | Hallu: {result_item.get('hallucination', 0.0):.0f} "
                f"| DocMatch: {metrics['doc_match']:.0f} | Latency: {latency:.1f}s"
            )

            summary = _recompute_and_persist()

    await asyncio.gather(
        *[_process_query(i, it) for i, it in enumerate(benchmark_data)],
        return_exceptions=False,
    )

    if not results:
        return None

    # Phase 2 (batch judge): resolve every deferred score from one OpenAI batch,
    # patch the result rows, recompute the 3-way labels, and rewrite the summary.
    if batch_collector is not None and batch_collector.count > 0:
        from utils.metrics import _resolve_judge_fields, _call_judge_llm
        payloads: Optional[dict[str, Any]] = None
        try:
            payloads = await batch_collector.run()
        except Exception as exc:
            logger.error("Batch judge failed (%s); falling back to synchronous judge for deferred rows.", exc)
        for r in results:
            deferred = r.pop("_deferred_judge", None)
            if not deferred:
                continue
            cid = deferred["custom_id"]
            payload = payloads.get(cid) if payloads is not None else None
            if payload is None and payloads is None:
                # Whole batch failed → sync fallback per row (keeps scores valid).
                _, payload = await _call_judge_llm(deferred["prompt"], vllm)
            judge_fields = _resolve_judge_fields(
                payload,
                deferred["response"],
                deferred["judge_model"],
                (deferred["heuristic_score"], deferred["heuristic_reason"]),
                has_client=True,
            )
            r.update(judge_fields)
            _apply_judge_label(r, is_financebench, is_multihoprag)
        summary = _recompute_and_persist()

    def _make_gate_check(actual: float, target: float, mode: str) -> dict[str, Any]:
        passed = actual >= target if mode == "min" else actual <= target
        return {"mode": mode, "target": target, "actual": actual, "passed": passed}

    gate_payload: dict[str, Any] = {"enabled": RAGConfig.BENCHMARK_GATE_ENABLED, "passed": None, "checks": {}}
    if RAGConfig.BENCHMARK_GATE_ENABLED:
        checks: dict[str, dict[str, Any]] = {}
        avg_latency = float(summary.get("avg_latency", 0.0))
        checks["avg_latency"] = _make_gate_check(avg_latency, RAGConfig.BENCHMARK_MAX_AVG_LATENCY, "max")
        checks["avg_llm_judge_score"] = _make_gate_check(
            float(summary.get("avg_llm_judge_score", 0.0)),
            RAGConfig.BENCHMARK_MIN_LLM_JUDGE,
            "min",
        )
        checks["avg_doc_match"] = _make_gate_check(
            float(summary.get("avg_doc_match", 0.0)),
            RAGConfig.BENCHMARK_MIN_DOC_MATCH,
            "min",
        )

        gate_payload["checks"] = checks
        gate_payload["passed"] = all(item.get("passed", False) for item in checks.values())
    summary["benchmark_gate"] = gate_payload

    with open(result_file, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    try:
        _write_model_report_artifacts(summary, result_file)
    except Exception as exc:
        logger.warning("Failed to write final report artifacts for %s: %s", result_file, exc)

    print(f"\n{'=' * 50}")
    print(f"[{strategy.upper()}] Benchmark Complete - {dataset_name}")
    print(f"{'=' * 50}")
    for key, value in summary.items():
        if key.startswith("avg_"):
            print(f"  Overall {key}: {value:.4f}")

    print("\nCategory Breakdown:")
    for cat, cat_sum in summary["category_summaries"].items():
        print(f"  - {cat} (n={cat_sum['count']}):")
        for key, value in cat_sum.items():
            if key.startswith("avg_"):
                print(f"    {key}: {value:.4f}")

    if summary["benchmark_gate"]["enabled"]:
        gate_result = "PASS" if summary["benchmark_gate"]["passed"] else "FAIL"
        print(f"\nBenchmark Gate: {gate_result}")
        for name, check in summary["benchmark_gate"]["checks"].items():
            target_str = f">= {check['target']:.4f}" if check["mode"] == "min" else f"<= {check['target']:.4f}"
            print(f"  {name}: {check['actual']:.4f} (target {target_str}) -> {'PASS' if check['passed'] else 'FAIL'}")

    print(f"\n  Final results saved to: {result_file}")
    print(f"{'=' * 50}\n")
    return summary


def _parse_seeds_env(raw: str) -> list[int]:
    """Parse comma/space-separated seed list. Empty -> []."""
    out: list[int] = []
    for token in re.split(r"[,\s]+", (raw or "").strip()):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            logger.warning("Ignoring non-integer seed token: %r", token)
    return out


def _aggregate_seed_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean / std / 95% CI per metric across N seeds.

    CI = mean ± 1.96 * std / sqrt(N)  (normal-approx; fine for N>=3 + smooth metrics).
    Per-category aggregates are computed only over keys that appear in every seed.
    """
    import math

    def _agg_keys(values: list[float]) -> dict[str, float]:
        n = len(values)
        if n == 0:
            return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n": 0}
        mean = sum(values) / n
        if n == 1:
            return {"mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean, "n": 1}
        var = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(var)
        margin = 1.96 * std / math.sqrt(n)
        return {"mean": mean, "std": std, "ci95_low": mean - margin, "ci95_high": mean + margin, "n": n}

    if not summaries:
        return {}

    avg_keys = sorted({k for s in summaries for k in s.keys() if k.startswith("avg_")})
    overall: dict[str, Any] = {}
    for key in avg_keys:
        vals = [_safe_float(s.get(key, 0.0), 0.0) for s in summaries if key in s]
        overall[key] = _agg_keys(vals)

    # Category-level aggregation: only categories that all seeds reported
    common_cats: Optional[set[str]] = None
    for s in summaries:
        cats = set((s.get("category_summaries") or {}).keys())
        common_cats = cats if common_cats is None else (common_cats & cats)
    common_cats = common_cats or set()

    categories: dict[str, dict[str, Any]] = {}
    for cat in sorted(common_cats):
        cat_keys = sorted({
            k
            for s in summaries
            for k in (s.get("category_summaries", {}).get(cat, {}) or {}).keys()
            if k.startswith("avg_")
        })
        per_cat = {}
        for key in cat_keys:
            vals = [
                _safe_float(s["category_summaries"][cat].get(key, 0.0), 0.0)
                for s in summaries
                if cat in (s.get("category_summaries") or {})
            ]
            per_cat[key] = _agg_keys(vals)
        per_cat["count"] = int(summaries[0].get("category_summaries", {}).get(cat, {}).get("count", 0))
        categories[cat] = per_cat

    return {"overall": overall, "categories": categories}


async def run_benchmark_multi_seed(
    queries_file: str,
    strategy: str,
    model_id: str,
    seeds: Optional[list[int]] = None,
    is_batch: bool = False,
    sample_companies: Optional[list[str]] = None,
    corpus_tag: str = "default",
    output_dir: Optional[Path] = None,
    limit: Optional[int] = None,
):
    """Run the benchmark once per seed, then write a `seeds_aggregate.json`
    with mean/std/95%-CI per metric. When seeds is empty/None, behaves
    identically to a single run_benchmark() call.
    """
    if seeds is None:
        seeds = _parse_seeds_env(os.environ.get("RAG_BENCHMARK_SEEDS", ""))

    if not seeds:
        return await run_benchmark(
            queries_file=queries_file,
            strategy=strategy,
            model_id=model_id,
            is_batch=is_batch,
            sample_companies=sample_companies,
            corpus_tag=corpus_tag,
            output_dir=output_dir,
            limit=limit,
        )

    # Pin a single timestamp across all seeds so they share one result root.
    if not os.environ.get("RAG_BENCHMARK_TIMESTAMP"):
        os.environ["RAG_BENCHMARK_TIMESTAMP"] = time.strftime("%Y%m%d_%H%M%S")

    summaries: list[dict[str, Any]] = []
    for s in seeds:
        logger.info("=== Seed %d (%d/%d) ===", s, len(summaries) + 1, len(seeds))
        summary = await run_benchmark(
            queries_file=queries_file,
            strategy=strategy,
            model_id=model_id,
            is_batch=is_batch,
            sample_companies=sample_companies,
            corpus_tag=corpus_tag,
            output_dir=output_dir,
            limit=limit,
            seed=s,
        )
        if summary is not None:
            summary["_seed"] = s
            summaries.append(summary)

    if not summaries:
        return None

    timestamp = os.environ.get("RAG_BENCHMARK_TIMESTAMP") or time.strftime("%Y%m%d_%H%M%S")
    parent_root = (output_dir or (Path("data/results") / timestamp))
    seeds_root = parent_root / strategy / corpus_tag

    aggregate = _aggregate_seed_summaries(summaries)
    payload = {
        "strategy": strategy,
        "corpus_tag": corpus_tag,
        "seeds": seeds,
        "n_seeds": len(summaries),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "aggregate": aggregate,
    }
    seeds_root.mkdir(parents=True, exist_ok=True)
    out_path = seeds_root / "seeds_aggregate.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 50}")
    print(f"[{strategy.upper()}] Multi-seed Aggregate (N={len(summaries)} seeds={seeds})")
    print(f"{'=' * 50}")
    for key, stats in aggregate.get("overall", {}).items():
        print(
            f"  {key}: {stats['mean']:.4f} ± {stats['std']:.4f}  "
            f"(95%CI [{stats['ci95_low']:.4f}, {stats['ci95_high']:.4f}], n={stats['n']})"
        )
    print(f"\n  Aggregate saved to: {out_path}")
    print(f"{'=' * 50}\n")
    return payload
