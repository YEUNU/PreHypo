#!/usr/bin/env python3
"""Post-hoc K-fold mean/std/CI from a single full benchmark run.

Running the full set once is enough for figure error bars: this partitions each
strategy's per-query `details` into K folds (fixed seed), computes every numeric
metric per fold, and reports mean / std / 95%-CI across folds. This is the cheap
alternative to `run_benchmark_multi_seed` (which re-runs the whole set N times).

Usage:
    python scripts/kfold_analysis.py --run-dir data/results/<timestamp> [--k 5] [--seed 42]
    python scripts/kfold_analysis.py --results a.json b.json --k 5

Outputs (next to the run dir, or CWD when --results is used):
    kfold_aggregate.json   full per-strategy / per-category stats
    kfold_figure.csv       tidy rows (strategy, scope, metric, mean, std, ci95_low, ci95_high, k)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Optional

# financebench_label is a string; we fold its three rates instead.
_LABELS = ("Correct Answer", "Incorrect Answer", "Refusal")


def _agg(values: list[float]) -> dict[str, float]:
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "k": 0}
    mean = sum(values) / n
    if n == 1:
        return {"mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean, "k": 1}
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))
    margin = 1.96 * std / math.sqrt(n)
    return {"mean": mean, "std": std, "ci95_low": mean - margin, "ci95_high": mean + margin, "k": n}


def _numeric_metric_keys(rows: list[dict]) -> list[str]:
    """Numeric metric keys shared across rows (skip bools and bookkeeping)."""
    skip = {"latency"}  # keep latency? include it — useful for a latency panel.
    skip = set()
    keys: set[str] = set()
    for r in rows:
        for k, v in r.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)) and not k.startswith("_") and k not in skip:
                keys.add(k)
    return sorted(keys)


def _fold_indices(n: int, k: int, seed: int) -> list[list[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    # near-equal contiguous folds over the shuffled order
    folds, base, rem = [], n // k, n % k
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        folds.append(idx[start:start + size])
        start += size
    return [f for f in folds if f]


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _fold_stats(rows: list[dict], k: int, seed: int) -> dict[str, Any]:
    metrics = _numeric_metric_keys(rows)
    folds = _fold_indices(len(rows), k, seed)
    out: dict[str, Any] = {}
    for m in metrics:
        # Per-fold mean over JUDGED rows only: exclude the UNJUDGED sentinel
        # (-1) for llm_judge_score / hallucination / answer_attempted. Every
        # real metric is >= 0, so `>= 0` drops only sentinels.
        fold_means = [
            _mean([float(rows[i][m]) for i in f
                   if isinstance(rows[i].get(m), (int, float)) and not isinstance(rows[i].get(m), bool) and rows[i][m] >= 0])
            for f in folds
        ]
        out[m] = {**_agg(fold_means), "fold_means": fold_means}
    # 3-way label rates per fold (FinanceBench taxonomy; applies to MultiHop too).
    # Denominator = judged rows in the fold (label in _LABELS); "Unjudged" excluded.
    if any("financebench_label" in r for r in rows):
        for label in _LABELS:
            fold_rates = []
            for f in folds:
                judged = [i for i in f if rows[i].get("financebench_label") in _LABELS]
                hits = sum(1 for i in judged if rows[i].get("financebench_label") == label)
                fold_rates.append(hits / len(judged) if judged else 0.0)
            key = "rate_" + label.lower().replace(" ", "_")
            out[key] = {**_agg(fold_rates), "fold_means": fold_rates}
    return out


def _analyze_strategy(summary: dict, k: int, seed: int) -> dict[str, Any]:
    rows = [r for r in (summary.get("details") or []) if isinstance(r, dict)]
    result: dict[str, Any] = {
        "n_queries": len(rows),
        "overall": _fold_stats(rows, k, seed),
        "by_category": {},
    }
    cats: dict[str, list[dict]] = {}
    for r in rows:
        cats.setdefault(str(r.get("category", "Uncategorized")), []).append(r)
    for cat, cat_rows in sorted(cats.items()):
        if len(cat_rows) >= k:  # need at least one row per fold to be meaningful
            result["by_category"][cat] = {"n": len(cat_rows), **{"metrics": _fold_stats(cat_rows, k, seed)}}
    return result


def _load_result_files(run_dir: Optional[Path], explicit: list[str]) -> dict[str, dict]:
    """Return {strategy: summary_dict}. Auto-discovers main result JSONs under
    a run dir (the ones carrying a `details` list), or loads explicit files."""
    paths: list[Path] = [Path(p) for p in explicit]
    if run_dir:
        for p in sorted(run_dir.rglob("*.json")):
            if p.name.endswith((".summary.json", ".stage_diagnostics.json")):
                continue
            paths.append(p)
    out: dict[str, dict] = {}
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("details"), list) and data.get("strategy"):
            out[str(data["strategy"])] = data
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-hoc K-fold mean/std from a single benchmark run.")
    ap.add_argument("--run-dir", type=str, default=None, help="data/results/<timestamp> dir to scan")
    ap.add_argument("--results", nargs="*", default=[], help="explicit result JSON files")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default=None, help="where to write outputs (default: run-dir or CWD)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    summaries = _load_result_files(run_dir, args.results)
    if not summaries:
        raise SystemExit("No strategy result JSONs found (need a `details` list + `strategy`).")

    agg = {strat: _analyze_strategy(s, args.k, args.seed) for strat, s in sorted(summaries.items())}
    meta = {"k": args.k, "seed": args.seed, "strategies": sorted(summaries.keys())}

    out_dir = Path(args.out_dir) if args.out_dir else (run_dir or Path("."))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kfold_aggregate.json").write_text(
        json.dumps({"meta": meta, "strategies": agg}, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # tidy CSV for plotting (overall + per-category)
    with open(out_dir / "kfold_figure.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "scope", "metric", "mean", "std", "ci95_low", "ci95_high", "k"])
        for strat, sa in agg.items():
            for metric, st in sa["overall"].items():
                w.writerow([strat, "overall", metric, f"{st['mean']:.6f}", f"{st['std']:.6f}",
                            f"{st['ci95_low']:.6f}", f"{st['ci95_high']:.6f}", st["k"]])
            for cat, cd in sa["by_category"].items():
                for metric, st in cd["metrics"].items():
                    w.writerow([strat, f"cat:{cat}", metric, f"{st['mean']:.6f}", f"{st['std']:.6f}",
                                f"{st['ci95_low']:.6f}", f"{st['ci95_high']:.6f}", st["k"]])

    # console summary of the headline metrics
    print(f"\n{args.k}-fold (seed={args.seed}) over {meta['strategies']}\n")
    headline = ["llm_judge_score", "hallucination", "mrr@10", "hits@4", "hits@10", "answer_attempted"]
    for strat, sa in agg.items():
        print(f"[{strat}] n={sa['n_queries']}")
        for m in headline:
            if m in sa["overall"]:
                st = sa["overall"][m]
                print(f"  {m:18s} {st['mean']:.4f} ± {st['std']:.4f}  (95% CI {st['ci95_low']:.4f}..{st['ci95_high']:.4f})")
    print(f"\nWrote {out_dir/'kfold_aggregate.json'} and {out_dir/'kfold_figure.csv'}")


if __name__ == "__main__":
    main()
