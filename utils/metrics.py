import logging
import re
import string
from difflib import SequenceMatcher
from typing import List, Any, Optional
from core.config import RAGConfig
from utils.prompts import FINANCEBENCH_JUDGE_PROMPT, MULTIHOPRAG_JUDGE_PROMPT

logger = logging.getLogger(__name__)


def normalize_answer(s):
    """Normalize answer text for comparison."""
    if not s:
        return ""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# --- FinanceBench Specific Metrics ---

def extract_numeric_value(s: str) -> float | None:
    """
    금융 값에서 숫자 추출.
    "$1,577.00" → 1577.0
    "8.70 billion" → 8.7 (단위 변환은 별도 처리 필요)
    """
    if not s:
        return None
    
    # 통화 기호 및 쉼표 제거
    cleaned = re.sub(r'[$€£¥,]', '', s.strip())
    
    # 숫자 패턴 매칭 (음수 포함)
    match = re.search(r'-?\d+\.?\d*', cleaned)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def calculate_financebench_accuracy(prediction: str, ground_truth: str) -> dict:
    """
    FinanceBench 금융 값 정확도 계산.
    
    Returns:
        dict with 'exact_match', 'numeric_match', 'contains_match'
    """
    if not prediction or not ground_truth:
        return {"exact_match": 0.0, "numeric_match": 0.0, "contains_match": 0.0}
    
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)
    
    # 1. Exact Match (정규화 후)
    exact_match = 1.0 if pred_norm == gt_norm else 0.0
    
    # 2. Numeric Match (숫자 추출 후 비교)
    pred_num = extract_numeric_value(prediction)
    gt_num = extract_numeric_value(ground_truth)
    
    numeric_match = 0.0
    if pred_num is not None and gt_num is not None:
        # 상대 오차 5% 이내면 매칭
        if gt_num != 0:
            rel_error = abs(pred_num - gt_num) / abs(gt_num)
            numeric_match = 1.0 if rel_error < 0.05 else 0.0
        else:
            numeric_match = 1.0 if pred_num == 0 else 0.0
    
    # 3. Contains Match (ground truth가 prediction에 포함)
    contains_match = 1.0 if gt_norm in pred_norm else 0.0
    
    return {
        "exact_match": exact_match,
        "numeric_match": numeric_match,
        "contains_match": contains_match
    }


def calculate_evidence_match(
    retrieved_sources: List[Any], 
    expected_doc: str, 
    expected_page: int | None = None
) -> dict:
    """
    FinanceBench 증거 매칭 - 문서/페이지 레벨.
    Supports both string filenames and structured [title, page, ...] lists.
    
    Args:
        retrieved_sources: List of strings or lists [title, page, sent_id]
        expected_doc: 예상 문서명 (e.g., "3M_2018_10K")
        expected_page: 예상 페이지 번호 (optional)
    
    Returns:
        dict with 'doc_match', 'page_match'
    """
    if not retrieved_sources or not expected_doc:
        return {"doc_match": 0.0, "page_match": 0.0}
    
    doc_match = 0.0
    page_match = 0.0

    def normalize_doc_id(value: str) -> str:
        if not value:
            return ""
        lowered = str(value).lower().strip()
        lowered = re.sub(r"\.(pdf|txt|md|json)$", "", lowered)
        lowered = lowered.replace("10-k", "10k").replace("10-q", "10q")
        lowered = re.sub(r"[^a-z0-9]+", "", lowered)
        return lowered

    def tokenize_doc_id(value: str) -> set[str]:
        if not value:
            return set()
        lowered = str(value).lower()
        lowered = lowered.replace("10-k", "10k").replace("10-q", "10q")
        lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
        return {tok for tok in lowered.split() if tok}

    expected_doc_norm = normalize_doc_id(expected_doc)
    expected_doc_tokens = tokenize_doc_id(expected_doc)

    for source in retrieved_sources:
        src_title = ""
        src_page = None

        # Dict Source: {"doc": ..., "page": ..., "text": ...}
        if isinstance(source, dict):
            src_title = str(source.get("doc", "")).lower()
            src_page = source.get("page")
        
        # Structured Source: [title, page, sent_id]
        elif isinstance(source, (list, tuple)) and len(source) >= 2:
            src_title = str(source[0]).lower()
            src_page = source[1]
            
        # String Source: "Title" or "Title_page_5"
        elif isinstance(source, str):
            src_title = source

        src_doc_norm = normalize_doc_id(src_title)
        src_doc_tokens = tokenize_doc_id(src_title)

        is_doc_match = False
        if expected_doc_norm and src_doc_norm:
            if expected_doc_norm in src_doc_norm or src_doc_norm in expected_doc_norm:
                is_doc_match = True
            else:
                sim = SequenceMatcher(None, expected_doc_norm, src_doc_norm).ratio()
                if sim >= 0.92:
                    is_doc_match = True

        if not is_doc_match and expected_doc_tokens and src_doc_tokens:
            overlap = len(expected_doc_tokens.intersection(src_doc_tokens))
            min_required = max(1, int(len(expected_doc_tokens) * 0.6))
            if overlap >= min_required:
                is_doc_match = True

        if is_doc_match:
            doc_match = 1.0
            if expected_page is not None:
                if isinstance(src_page, (int, float)) and int(src_page) == expected_page:
                    page_match = 1.0
                    break
                if isinstance(source, str):
                    source_lower = source.lower()
                    page_pattern = f"page_{expected_page:03d}" if isinstance(expected_page, int) else f"page_{expected_page}"
                    if page_pattern in source_lower or f"_page_{expected_page}" in source_lower:
                        page_match = 1.0
                        break

    return {"doc_match": doc_match, "page_match": page_match}

def _parse_unit_score(raw: Any) -> Optional[float]:
    try:
        if raw is None:
            return None
        value = float(raw)
        return max(0.0, min(1.0, value))
    except Exception:
        return None


def _is_insufficient_text(text: Any) -> bool:
    # Matches the 3-way taxonomy: any recognized abstain phrase (Hypo's
    # "insufficient evidence", HopRAG's "I do not know", or natural-language
    # refusals) → not a hallucination.
    from utils.abstain import is_abstain
    return is_abstain(text)


async def _run_combined_judge(
    judge_prompt: str,
    response: str,
    vllm_client,
    heuristic_fn,
) -> dict:
    """Run the shared single-call LLM judge and resolve (score, hallucination).

    Both FinanceBench and MultiHop-RAG use the SAME judging machinery — one
    LLM call returns `{score, hallucination, reason}` so the two judgements
    stay internally consistent (score=1.0 ⇒ hallucination=0.0; honest abstain
    ⇒ both 0). Only the prompt and the no-client heuristic differ per dataset,
    so those are passed in. Returns the dataset-agnostic judge fields; callers
    append their own evidence/retrieval metrics.
    """
    judge_model, judge_payload = await _call_judge_llm(judge_prompt, vllm_client)
    return _resolve_judge_fields(
        judge_payload, response, judge_model, heuristic_fn(), has_client=bool(vllm_client)
    )


async def _call_judge_llm(judge_prompt: str, vllm_client) -> tuple[str, Optional[dict]]:
    """The synchronous LLM judge call (with one fallback-model retry).

    Returns (judge_model, judge_payload). Factored out so the batch path can
    reuse `_resolve_judge_fields` on payloads it fetched elsewhere.
    """
    judge_model = RAGConfig.EVAL_MODEL
    judge_payload: Optional[dict] = None
    if vllm_client:
        try:
            judge_payload = await vllm_client.generate_json(
                [{"role": "user", "content": judge_prompt}],
                model=RAGConfig.EVAL_MODEL,
            )
            if _parse_unit_score((judge_payload or {}).get("score")) is None:
                fallback_model = RAGConfig.DEFAULT_MODEL
                if fallback_model and fallback_model != RAGConfig.EVAL_MODEL:
                    logger.warning(
                        "Judge response missing score with model '%s'. Retrying with fallback model '%s'.",
                        RAGConfig.EVAL_MODEL,
                        fallback_model,
                    )
                    judge_payload = await vllm_client.generate_json(
                        [{"role": "user", "content": judge_prompt}],
                        model=fallback_model,
                    )
                    judge_model = fallback_model
        except Exception as e:
            logger.error(f"LLM Judge failed: {e}")
            judge_payload = None
    return judge_model, judge_payload


def _resolve_judge_fields(
    judge_payload: Optional[dict],
    response: str,
    judge_model: str,
    heuristic: tuple[float, str],
    has_client: bool = True,
) -> dict:
    """Turn a judge payload (sync or batch) into the judge/hallucination fields.

    Pure / no I/O so both the synchronous and OpenAI-Batch paths resolve scores
    identically. `heuristic` is the precomputed (score, reason) fallback used
    when the payload has no usable score.
    """
    parsed_score = _parse_unit_score((judge_payload or {}).get("score"))
    parsed_hallu = _parse_unit_score((judge_payload or {}).get("hallucination"))

    if parsed_score is not None:
        judge_score = parsed_score
        judge_reason = str((judge_payload or {}).get("reason", "")) or "combined_judge"
    else:
        judge_score, judge_reason = heuristic
        if not has_client:
            judge_reason = "fallback_heuristic_without_judge_client"

    # Resolve hallucination from the same combined call when possible. Apply
    # the honest-abstain rule deterministically (so the LLM cannot label a
    # genuine abstention as a hallucination).
    if _is_insufficient_text(response):
        hallucination = 0.0
        hallucination_reason = "non_answer_insufficient"
        hallucination_source = "rule_non_answer"
    elif not str(response or "").strip():
        hallucination = 0.0
        hallucination_reason = "non_answer_empty"
        hallucination_source = "rule_non_answer"
    elif parsed_hallu is not None:
        hallucination = 1.0 if parsed_hallu >= 0.5 else 0.0
        hallucination_reason = str((judge_payload or {}).get("reason", "")) or "combined_judge"
        hallucination_source = "combined_judge"
    else:
        hallucination = 1.0 if judge_score < 1.0 else 0.0
        hallucination_reason = "fallback_llm_judge_due_invalid_payload"
        hallucination_source = "llm_judge_fallback"

    return {
        "llm_judge_score": judge_score,
        "llm_judge_reason": judge_reason,
        "hallucination": hallucination,
        "hallucination_reason": hallucination_reason,
        "hallucination_source": hallucination_source,
        "hallucination_model": judge_model,
    }


async def _judge_or_defer(
    judge_prompt: str,
    response: str,
    vllm_client,
    heuristic_fn,
    batch_collector=None,
    custom_id=None,
) -> dict:
    """Run the judge synchronously, or — when a batch collector is supplied —
    register the prompt and return a tentative (heuristic) result tagged with
    `_deferred_judge` so the benchmark can patch in the real score after the
    OpenAI batch completes.
    """
    if batch_collector is not None and custom_id is not None:
        batch_collector.register(str(custom_id), judge_prompt)
        heuristic = heuristic_fn()
        judge = _resolve_judge_fields(None, response, RAGConfig.EVAL_MODEL, heuristic, has_client=True)
        judge["_deferred_judge"] = {
            "custom_id": str(custom_id),
            "prompt": judge_prompt,
            "response": response,
            "heuristic_score": heuristic[0],
            "heuristic_reason": heuristic[1],
            "judge_model": RAGConfig.EVAL_MODEL,
        }
        return judge
    return await _run_combined_judge(judge_prompt, response, vllm_client, heuristic_fn)


async def evaluate_financebench_response(
    query: str,
    response: str,
    ground_truth: str,
    retrieved_sources: List[Any],
    expected_doc: str,
    expected_page: Optional[int] = None,
    vllm_client = None,
    batch_collector = None,
    custom_id = None,
) -> dict:
    """
    FinanceBench 통합 평가 인터페이스 (LLM-as-a-judge + Evidence Match).
    Score and hallucination come from a SINGLE LLM call (see
    `_run_combined_judge`); evidence match is doc/page level.
    """
    def _heuristic_judge() -> tuple[float, str]:
        fallback_acc = calculate_financebench_accuracy(response, ground_truth)
        score = max(
            fallback_acc["exact_match"],
            fallback_acc["numeric_match"],
            fallback_acc["contains_match"],
        )
        return score, "fallback_heuristic: exact/numeric/contains max"

    judge_prompt = FINANCEBENCH_JUDGE_PROMPT.format(
        query=query,
        ground_truth=ground_truth,
        response=response,
    )
    judge = await _judge_or_defer(
        judge_prompt, response, vllm_client, _heuristic_judge, batch_collector, custom_id
    )

    # Evidence Match (Doc & Page). Supports dict/list/str source types.
    evidence_metrics = calculate_evidence_match(retrieved_sources, expected_doc, expected_page)

    return {
        **judge,
        "doc_match": evidence_metrics["doc_match"],
        "page_match": evidence_metrics["page_match"],
    }


# --- MultiHop-RAG Specific Metrics ---

def _fact_matches_chunk(fact_norm: str, chunk_norm: str) -> bool:
    """True if a gold evidence fact is contained in / overlaps a retrieved
    chunk. `fact_norm`/`chunk_norm` are already `normalize_answer`-ed.

    A MultiHop-RAG `fact` is a sentence pulled from a source article, so the
    retrieved chunk that supports it should contain that sentence (or share
    most of its tokens after table/whitespace reflow). Substring first, then
    a 0.6 token-coverage fallback for reflowed chunks.
    """
    if not fact_norm or not chunk_norm:
        return False
    if fact_norm in chunk_norm or chunk_norm in fact_norm:
        return True
    fact_tokens = set(fact_norm.split())
    if not fact_tokens:
        return False
    overlap = len(fact_tokens & set(chunk_norm.split()))
    return (overlap / len(fact_tokens)) >= 0.6


def _source_chunk_text(source: Any) -> str:
    """Pull the chunk body text out of a retrieved source of any shape
    (dict {"text"}, list [title, page, text], or raw string)."""
    if isinstance(source, dict):
        return str(source.get("text", "") or "")
    if isinstance(source, (list, tuple)) and len(source) >= 3:
        return str(source[2] or "")
    if isinstance(source, str):
        return source
    return ""


def calculate_retrieval_ranking_metrics(
    retrieved_sources: List[Any],
    gold_facts: List[str],
    ks: tuple[int, ...] = (4, 10),
) -> dict:
    """MultiHop-RAG retrieval metrics (Tang & Yang, 2024): fact-level
    MRR@10, MAP@10, Hits@K. Gold unit is each evidence `fact`; relevance is
    `_fact_matches_chunk` against the ranked retrieved chunk texts.

    - hits@k : recall of distinct gold facts within the top-k chunks.
    - mrr@10 : reciprocal rank of the first chunk covering ANY gold fact.
    - map@10 : average precision over the ranked chunks (a chunk is
               "relevant" when it covers a not-yet-covered gold fact),
               normalized by the gold-fact count.
    """
    result: dict[str, float] = {f"hits@{k}": 0.0 for k in ks}
    result["mrr@10"] = 0.0
    result["map@10"] = 0.0

    gold_norm = [g for g in (normalize_answer(f) for f in (gold_facts or []) if f) if g]
    if not gold_norm or not retrieved_sources:
        return result

    chunk_norms = [normalize_answer(_source_chunk_text(s)) for s in retrieved_sources]
    total_gold = len(gold_norm)

    # MRR / MAP over the ranked list (count a gold fact once, at first cover).
    covered: set[int] = set()
    first_hit_rank: Optional[int] = None
    relevant_count = 0
    ap_sum = 0.0
    for rank, cn in enumerate(chunk_norms, start=1):
        newly = {
            gi for gi, g in enumerate(gold_norm)
            if gi not in covered and _fact_matches_chunk(g, cn)
        }
        if not newly:
            continue
        covered |= newly
        relevant_count += 1
        if first_hit_rank is None:
            first_hit_rank = rank
        if rank <= 10:
            ap_sum += relevant_count / rank

    result["mrr@10"] = (1.0 / first_hit_rank) if (first_hit_rank and first_hit_rank <= 10) else 0.0
    result["map@10"] = ap_sum / total_gold

    # Hits@k: distinct gold facts recalled within the top-k chunks.
    for k in ks:
        top_k = chunk_norms[:k]
        hit = sum(
            1 for g in gold_norm
            if any(_fact_matches_chunk(g, cn) for cn in top_k)
        )
        result[f"hits@{k}"] = hit / total_gold

    return result


def calculate_multihop_doc_recall(retrieved_sources: List[Any], gold_docs: List[str]) -> float:
    """Coarse doc-level recall: fraction of gold articles (by title) that
    appear among the retrieved sources. Complements the fact-level ranking
    metrics with a title-match view robust to chunk reflow."""
    gold = [d for d in (gold_docs or []) if d and str(d).strip()]
    if not gold:
        return 0.0
    hit = 0
    for doc in gold:
        m = calculate_evidence_match(retrieved_sources, doc, expected_page=None)
        if m["doc_match"] >= 1.0:
            hit += 1
    return hit / len(gold)


async def evaluate_multihoprag_response(
    query: str,
    response: str,
    ground_truth: str,
    retrieved_sources: List[Any],
    evidence_facts: Optional[List[str]] = None,
    evidence_docs: Optional[List[str]] = None,
    question_type: str = "",
    vllm_client = None,
    batch_collector = None,
    custom_id = None,
) -> dict:
    """MultiHop-RAG evaluation: type-aware LLM judge + fact-level retrieval
    ranking metrics (MRR/MAP/Hits@K), per Tang & Yang (2024). Judging shares
    `_run_combined_judge` with FinanceBench but uses MULTIHOPRAG_JUDGE_PROMPT
    (news multi-hop framing, no numeric-unit reconciliation) and a
    contains/exact text heuristic instead of FinanceBench's numeric one.
    """
    def _heuristic_judge() -> tuple[float, str]:
        pred = normalize_answer(response)
        gt = normalize_answer(ground_truth)
        if not gt:
            return 0.0, "fallback_heuristic: empty_ground_truth"
        score = 1.0 if (pred == gt or gt in pred) else 0.0
        return score, "fallback_heuristic: exact/contains"

    judge_prompt = MULTIHOPRAG_JUDGE_PROMPT.format(
        question_type=question_type or "unknown",
        query=query,
        ground_truth=ground_truth,
        response=response,
    )
    judge = await _judge_or_defer(
        judge_prompt, response, vllm_client, _heuristic_judge, batch_collector, custom_id
    )

    ranking = calculate_retrieval_ranking_metrics(retrieved_sources, evidence_facts or [])
    doc_recall = calculate_multihop_doc_recall(retrieved_sources, evidence_docs or [])

    return {
        **judge,
        **ranking,
        "evidence_doc_recall": doc_recall,
        # doc_match kept for cross-dataset reporting parity (1.0 if any gold
        # article surfaced); page_match is N/A for the news corpus.
        "doc_match": 1.0 if doc_recall > 0.0 else 0.0,
        "page_match": 0.0,
    }
