#!/usr/bin/env python
"""Build a stratified MultiHop-RAG query sample (balanced by question_type).

The full set (data/multihoprag_queries.json, 2556 queries) is too slow for the
graph baselines (hoprag ~160s/query → ~28h for 2556 even at concurrency 4), so
the k-fold figures run on a balanced sample instead. This draws an equal number
of queries per question_type (comparison/inference/temporal/null) with a fixed
seed, so the sample is reproducible and each fold/type is balanced.

    python data/make_multihoprag_sample.py --per-type 50            # n=200 (default)
    python data/make_multihoprag_sample.py --per-type 25 --seed 42  # n=100

Output: data/multihoprag_sample<N>_queries.json (N = per_type * num_types).
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
FULL_PATH = DATA_DIR / "multihoprag_queries.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-type", type=int, default=50, help="queries sampled per question_type")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (reproducible)")
    ap.add_argument("--out", default=None, help="output path (default derives from total n)")
    args = ap.parse_args()

    if not FULL_PATH.exists():
        print(f"ERROR: {FULL_PATH} not found — run data/prepare_multihoprag.py first.")
        return 2

    with open(FULL_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    by_type: dict[str, list] = defaultdict(list)
    for q in queries:
        by_type[q.get("question_type", "unknown")].append(q)

    rng = random.Random(args.seed)
    sample = []
    for qtype in sorted(by_type):
        pool = sorted(by_type[qtype], key=lambda q: str(q.get("_id", "")))  # stable order before sampling
        if len(pool) < args.per_type:
            print(f"WARN: {qtype} has only {len(pool)} (< {args.per_type}); taking all.")
            picked = pool
        else:
            picked = rng.sample(pool, args.per_type)
        sample.extend(picked)
        print(f"  {qtype}: {len(picked)}")

    total = len(sample)
    out_path = Path(args.out) if args.out else (DATA_DIR / f"multihoprag_sample{total}_queries.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    print(f"Wrote {total} queries (seed={args.seed}) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
