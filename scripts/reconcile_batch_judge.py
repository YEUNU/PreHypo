#!/usr/bin/env python
"""Resolve pending async judge batches for a benchmark run.

When the benchmark runs with RAG_JUDGE_BATCH=true and RAG_JUDGE_BATCH_ASYNC=true,
each strategy submits its OpenAI judge batch and continues without blocking,
leaving a ``*.pending_judge.json`` manifest next to its result file. main.py
reconciles automatically after benchmark_all, but if that pass was interrupted
(or you want to re-poll later) run this directly:

    .venv/bin/python scripts/reconcile_batch_judge.py --run-dir data/results/<ts>

It polls every pending batch in parallel, patches the result JSONs by
``judge_custom_id``, recomputes aggregates, refreshes the .summary.json sidecar,
and removes each manifest. Idempotent.
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.benchmark import reconcile_pending_judges  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="Benchmark run dir, e.g. data/results/<timestamp>")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: run dir not found: {run_dir}", file=sys.stderr)
        return 2
    patched = asyncio.run(reconcile_pending_judges(run_dir))
    print(f"reconcile: {patched} result file(s) patched under {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
