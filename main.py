import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def _preset_rag_domain() -> None:
    """Set RAG_DOMAIN (financial|news) BEFORE the prompt modules import, so the
    model-side prompts (hypothetical-query gen, rewrite, rerank, synthesis)
    pick the right framing. Explicit RAG_DOMAIN / --domain win; otherwise infer
    from the queries file's `dataset` marker or the dataset path. Must run
    before `from cli.* import ...` because those transitively import the
    prompt constants, which are selected at import time."""
    if os.environ.get("RAG_DOMAIN", "").strip():
        return

    argv = sys.argv

    def _argval(flag: str):
        if flag in argv:
            idx = argv.index(flag)
            if idx + 1 < len(argv):
                return argv[idx + 1]
        for tok in argv:
            if tok.startswith(flag + "="):
                return tok.split("=", 1)[1]
        return None

    domain = _argval("--domain")
    if not domain:
        marker = ""
        queries_file = _argval("--queries_file")
        if queries_file and os.path.exists(queries_file):
            try:
                import json
                with open(queries_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if data:
                    marker = str(data[0].get("dataset", "")).strip().lower()
            except Exception:
                marker = ""
        dataset = _argval("--dataset") or ""
        domain = "news" if (marker == "multihoprag" or "multihoprag" in dataset) else "financial"

    os.environ["RAG_DOMAIN"] = domain


_preset_rag_domain()

from cli.benchmark import run_benchmark_multi_seed, reconcile_pending_judges
from cli.index import run_indexing
from models.prehypo.indexing.ocr import run_ocr
from core.neo4j_service import Neo4jService
from core.vllm_client import VLLMClient
from utils.io import get_sample_companies


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("PreHypo")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["index", "benchmark", "benchmark_all", "ocr"], required=True)
    parser.add_argument("--strategy", choices=["naive", "prehypo", "hyporeflect", "hoprag", "ms_graphrag"], default="prehypo")
    parser.add_argument("--domain", choices=["financial", "news"], default=None,
                        help="Prompt domain for model-side prompts. Default: auto-detected from "
                             "--queries_file dataset marker / --dataset path (financebench->financial, "
                             "multihoprag->news). Set RAG_DOMAIN to override.")
    parser.add_argument("--model", default="local")
    parser.add_argument("--dataset", default="data/finance_corpus")
    parser.add_argument("--queries_file", default="data/financebench_queries.json")
    parser.add_argument("--pdf_dir", default="data/finance_pdfs", help="Directory containing PDF files for OCR")
    parser.add_argument("--ocr_output", default="data/finance_corpus_ocr", help="Output directory for OCR results")
    parser.add_argument("--convert_tables", action="store_true", default=True, help="Convert tables to text during OCR")
    parser.add_argument("--no_convert_tables", action="store_false", dest="convert_tables", help="Do NOT convert tables to text")
    parser.add_argument("--raw-ocr", action="store_true", help="Use original OCR data (data/finance_corpus) without table-to-text conversion")
    parser.add_argument("--clear-graph", action="store_true", help="Clear all Neo4j data before indexing to prevent duplicates")
    parser.add_argument("--corpus-tag", default=None, help="Tag to identify corpus in Neo4j (e.g., 'raw', 'ocr'). Different tags prevent data conflicts.")
    parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate results to data/debug/ for inspection")
    parser.add_argument("--sample", action="store_true", help="Run on sample companies (one per sector)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of benchmark queries to evaluate")
    parser.add_argument("--n", type=int, default=None, help="Number of sample companies to include (company count, not file/query count). Auto-enables sample mode")
    parser.add_argument("--ocr", action="store_true", help="Use OCR processed data (combined with --sample for strict logic)")
    parser.add_argument("--save-to", type=str, help="Directory to save the sampled files to (only works with --mode index and --sample)")
    return parser


def _resolve_sample_companies(args: argparse.Namespace) -> tuple[bool, list[str] | None]:
    sample_mode = args.sample or (args.n is not None)
    if args.n is not None and args.n <= 0:
        logger.error("--n must be a positive integer.")
        return sample_mode, None

    if not sample_mode:
        return sample_mode, None

    if args.n is not None and not args.sample:
        logger.info("--n provided without --sample. Enabling sample mode automatically.")
    if args.n is not None:
        logger.info("--n controls sample company count (not file/query count).")

    sample_companies = get_sample_companies()
    if not sample_companies:
        logger.error("Could not determine sample companies. Aborting sample mode.")
        return sample_mode, None

    if args.n is not None:
        sample_companies = sample_companies[:args.n]
        logger.info("Sample subset enabled (--n %d). Using %d companies.", args.n, len(sample_companies))
    logger.info("Sample mode enabled. Selected %d companies: %s", len(sample_companies), sample_companies)
    return sample_mode, sample_companies


def _resolve_index_args(args: argparse.Namespace, sample_mode: bool) -> None:
    corpus_tag = args.corpus_tag
    if sample_mode:
        if args.ocr:
            preferred_sample_ocr_dataset = "data/finance_corpus_sample_ocr/text"
            fallback_sample_ocr_dataset = "data/finance_corpus_ocr/text"
            if os.path.exists(preferred_sample_ocr_dataset):
                args.dataset = preferred_sample_ocr_dataset
            elif os.path.exists(fallback_sample_ocr_dataset):
                args.dataset = fallback_sample_ocr_dataset
                logger.warning(
                    "Sample OCR dataset '%s' not found. Falling back to '%s'.",
                    preferred_sample_ocr_dataset,
                    fallback_sample_ocr_dataset,
                )
            else:
                args.dataset = preferred_sample_ocr_dataset
            if not corpus_tag:
                corpus_tag = "sample_ocr"
            if not args.save_to:
                args.save_to = "data/financebench_sample_ocr"
        else:
            args.dataset = "data/finance_corpus"
            if not corpus_tag:
                corpus_tag = "sample_raw"
            if not args.save_to:
                args.save_to = "data/financebench_sample_raw"
    elif args.raw_ocr:
        args.dataset = "data/finance_corpus"
        if not corpus_tag:
            corpus_tag = "raw"
        logger.info("Using raw OCR data: 'data/finance_corpus' (corpus_tag: %s)", corpus_tag)
    elif args.dataset == "data/finance_corpus":
        if os.path.exists("data/finance_corpus_ocr/text"):
            args.dataset = "data/finance_corpus_ocr/text"
            if not corpus_tag:
                corpus_tag = "ocr"
        elif os.path.exists("data/finance_corpus_ocr"):
            args.dataset = "data/finance_corpus_ocr"
            if not corpus_tag:
                corpus_tag = "ocr"

    args.corpus_tag = corpus_tag


def _resolve_benchmark_corpus_tag(args: argparse.Namespace, sample_mode: bool) -> str:
    corpus_tag = args.corpus_tag
    if sample_mode:
        if args.ocr and not corpus_tag:
            corpus_tag = "sample_ocr"
        elif not args.ocr and not corpus_tag:
            corpus_tag = "sample_raw"
    elif not corpus_tag:
        corpus_tag = "default"
    return corpus_tag


async def main():
    parser = _build_parser()
    args = parser.parse_args()

    sample_mode, sample_companies = _resolve_sample_companies(args)
    if sample_mode and sample_companies is None:
        return

    try:
        if args.mode == "index":
            _resolve_index_args(args, sample_mode)
            if args.clear_graph:
                neo4j = Neo4jService()
                logger.warning("Clearing all Neo4j data before indexing...")
                await neo4j.execute_query("MATCH (n) DETACH DELETE n")
                logger.info("Neo4j graph cleared successfully.")
            await run_indexing(args.dataset, args.strategy, args.model, args.corpus_tag, args.save_intermediate, sample_companies, args.save_to)
        elif args.mode == "benchmark":
            corpus_tag = _resolve_benchmark_corpus_tag(args, sample_mode)
            # Pin the run dir so async judge batches can be reconciled afterwards.
            env_ts = os.environ.get("RAG_BENCHMARK_TIMESTAMP")
            timestamp = env_ts if env_ts else time.strftime("%Y%m%d_%H%M%S")
            os.environ["RAG_BENCHMARK_TIMESTAMP"] = timestamp
            results_dir = Path("data/results") / timestamp
            await run_benchmark_multi_seed(args.queries_file, args.strategy, args.model, sample_companies=sample_companies, corpus_tag=corpus_tag, limit=args.limit)
            await reconcile_pending_judges(results_dir)
        elif args.mode == "benchmark_all":
            corpus_tag = _resolve_benchmark_corpus_tag(args, sample_mode)
            env_ts = os.environ.get("RAG_BENCHMARK_TIMESTAMP")
            timestamp = env_ts if env_ts else time.strftime("%Y%m%d_%H%M%S")
            results_dir = Path("data/results") / timestamp
            results_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Batch benchmark results will be saved to: %s", results_dir)
            for strategy in ["naive", "prehypo", "hoprag", "ms_graphrag"]:
                print(f"\n>>> Running Benchmark for: {strategy.upper()}")
                await run_benchmark_multi_seed(args.queries_file, strategy, args.model, is_batch=True, sample_companies=sample_companies, corpus_tag=corpus_tag, output_dir=results_dir, limit=args.limit)
            # All strategies done. In async batch-judge mode each left a pending
            # manifest; resolve every batch in parallel now (one wait, not four).
            await reconcile_pending_judges(results_dir)
        elif args.mode == "ocr":
            await run_ocr(args.pdf_dir, args.ocr_output, convert_tables=args.convert_tables, sample_companies=sample_companies)
    finally:
        try:
            await Neo4jService.global_close()
        except Exception as exc:
            logger.warning("Failed to close Neo4j driver cleanly: %s", exc)
        try:
            await VLLMClient.global_close()
        except Exception as exc:
            logger.warning("Failed to close VLLM clients cleanly: %s", exc)


if __name__ == "__main__":
    asyncio.run(main())
