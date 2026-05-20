"""Q-/Q+ direction-split ablation analysis (EMNLP rebuttal).

Loads the per-query result JSONs for the four cells:
  Full / qminus_only / qplus_only / single_combined
and emits:
  1. 5-fold mean +/- std (seed=42, n=150 shuffle, 5 folds of 30) on
     J / H / DocM / PgM / Att / Lat.
  2. Paired bootstrap CIs (B=10000, n=150, seed=42) on Judge gap
     of Full minus each variant (query indices resampled with
     replacement, fixed pairing).
  3. Conditional precision J | Att per variant + attempted-set
     overlap with Full.
  4. Three Table 2 LaTeX rows.

Inputs are taken from data/results/<RUN_PREFIX>_<variant>/.../<results>.json
where <RUN_PREFIX> defaults to 20260515_qq and variant is
{full, qminus_only, qplus_only, single_combined}. Override per-variant
paths with --result PATH or via env RAG_QQ_RESULTS_<VARIANT>.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Optional

METRIC_KEYS = [
    ("J", "llm_judge_score"),
    ("H", "hallucination"),
    ("DocM", "doc_match"),
    ("PgM", "page_match"),
    ("Att", "answer_attempted"),
    ("Lat", "latency"),
]

VARIANTS = ["full", "qminus_only", "qplus_only", "single_combined"]

DISPLAY_NAME = {
    "full": "PreHypo (Full)",
    "qminus_only": r"$-$Q$^+$ direction (Q$^-$-only)",
    "qplus_only": r"$-$Q$^-$ direction (Q$^+$-only)",
    "single_combined": r"$-$Direction split (Combined)",
}


def _default_result_path(variant: str, run_prefix: str = "20260515_qq") -> Path:
    base = Path("data/results") / f"{run_prefix}_{variant}"
    # Standard layout: <base>/<strategy>/<corpus_tag>/<strategy>_<corpus_tag>.json
    candidates = list(base.glob("hyporeflect/full_v19_hyporeflect/hyporeflect_full_v19_hyporeflect.json"))
    if candidates:
        return candidates[0]
    # Fallback: any *.json that isn't a summary / diagnostics
    candidates = [
        p for p in base.rglob("*.json")
        if not p.name.endswith(".summary.json") and not p.name.endswith(".stage_diagnostics.json")
    ]
    if not candidates:
        raise FileNotFoundError(f"No result JSON found under {base}")
    return candidates[0]


def _load_details(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    details = data.get("details") or []
    if not details:
        raise RuntimeError(f"{path} has no `details` array (probably aborted run)")
    return details


def _coerce_metric(detail: dict, key: str) -> float:
    val = detail.get(key)
    if val is None:
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _metric_vector(details: list[dict], key: str) -> list[float]:
    return [_coerce_metric(d, key) for d in details]


def _fold_assignments(n: int, k: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    fold_size = n // k
    extras = n % k
    folds: list[list[int]] = []
    cursor = 0
    for fold in range(k):
        size = fold_size + (1 if fold < extras else 0)
        folds.append(sorted(idx[cursor:cursor + size]))
        cursor += size
    return folds


def _fold_mean_std(values: list[float], folds: list[list[int]]) -> tuple[float, float, list[float]]:
    fold_means = [statistics.fmean([values[i] for i in fold]) for fold in folds]
    mean = statistics.fmean(fold_means)
    std = statistics.pstdev(fold_means) if len(fold_means) > 1 else 0.0
    return mean, std, fold_means


def _paired_bootstrap_ci(
    full_values: list[float],
    var_values: list[float],
    n_boot: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap CI on (Full - variant) judge gap with fixed pairing."""
    n = len(full_values)
    assert len(var_values) == n, "paired bootstrap requires equal length"
    diffs = [full_values[i] - var_values[i] for i in range(n)]
    observed = statistics.fmean(diffs)
    rng = random.Random(seed)
    boot_means: list[float] = []
    for _ in range(n_boot):
        sample_mean = 0.0
        for _ in range(n):
            sample_mean += diffs[rng.randrange(n)]
        boot_means.append(sample_mean / n)
    boot_means.sort()
    lo = boot_means[int(math.floor(alpha / 2 * n_boot))]
    hi = boot_means[min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)]
    n_pos = sum(1 for m in boot_means if m > 0)
    n_neg = sum(1 for m in boot_means if m < 0)
    # Two-sided bootstrap p-value approximation
    p_two_sided = 2.0 * min(n_pos, n_neg) / n_boot
    return {
        "observed_gap": observed,
        "ci_low": lo,
        "ci_high": hi,
        "p_two_sided": p_two_sided,
        "n_boot": n_boot,
    }


def _conditional_precision_j_given_att(details: list[dict]) -> float:
    j_vals = []
    for d in details:
        if _coerce_metric(d, "answer_attempted") >= 0.5:
            j_vals.append(_coerce_metric(d, "llm_judge_score"))
    if not j_vals:
        return 0.0
    return statistics.fmean(j_vals)


def _attempted_indices(details: list[dict]) -> set[int]:
    return {i for i, d in enumerate(details) if _coerce_metric(d, "answer_attempted") >= 0.5}


def _attempted_overlap(full_idx: set[int], var_idx: set[int]) -> dict[str, float]:
    inter = len(full_idx & var_idx)
    union = len(full_idx | var_idx)
    return {
        "jaccard": (inter / union) if union else 1.0,
        "intersection": inter,
        "union": union,
        "full_only": len(full_idx - var_idx),
        "var_only": len(var_idx - full_idx),
    }


def _format_pm(mean: float, std: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _table2_row(name: str, j: tuple[float, float], pgm: tuple[float, float], att: tuple[float, float]) -> str:
    return f"\\textbf{{{name}}} & {_format_pm(*j)} & {_format_pm(*pgm)} & {_format_pm(*att)} \\\\"


def _ascii_table(rows: list[list[str]], headers: list[str]) -> str:
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(c)) for c in col) for col in cols]
    sep = "  ".join("-" * w for w in widths)
    lines = ["  ".join(str(h).ljust(w) for h, w in zip(headers, widths)), sep]
    for r in rows:
        lines.append("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-prefix", default="20260515_qq")
    for v in VARIANTS:
        ap.add_argument(f"--result-{v.replace('_', '-')}", default=None,
                        help=f"Path to {v} result JSON (overrides default)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--out-md", default=".claude/scratch/qpm_qpp_ablation_fold.md")
    ap.add_argument("--out-json", default=".claude/scratch/qpm_qpp_ablation_fold.json")
    args = ap.parse_args()

    paths: dict[str, Path] = {}
    for v in VARIANTS:
        cli_attr = f"result_{v}"
        override = getattr(args, cli_attr)
        env_override = os.environ.get(f"RAG_QQ_RESULTS_{v.upper()}")
        if override:
            paths[v] = Path(override)
        elif env_override:
            paths[v] = Path(env_override)
        else:
            paths[v] = _default_result_path(v, args.run_prefix)

    print("=== Loading results ===", file=sys.stderr)
    cells: dict[str, list[dict]] = {}
    for v, p in paths.items():
        details = _load_details(p)
        print(f"  {v:18s} n={len(details):3d}  <- {p}", file=sys.stderr)
        cells[v] = details

    # Sanity: all variants must have identical query order and length
    n = len(cells["full"])
    for v in VARIANTS:
        if len(cells[v]) != n:
            print(f"WARN: {v} has {len(cells[v])} queries, full has {n}", file=sys.stderr)

    folds = _fold_assignments(n, args.folds, args.seed)

    report: dict = {
        "config": {
            "n_queries": n,
            "n_folds": args.folds,
            "seed": args.seed,
            "n_boot": args.n_boot,
            "paths": {v: str(p) for v, p in paths.items()},
        },
        "fold_variance": {},
        "bootstrap_judge_gap": {},
        "conditional_precision_j_given_att": {},
        "attempted_overlap_with_full": {},
    }

    # === 1. Fold variance ===
    fold_rows = []
    metric_means: dict[str, dict[str, tuple[float, float]]] = {}
    for v in VARIANTS:
        details = cells[v]
        per_variant = {}
        for label, key in METRIC_KEYS:
            values = _metric_vector(details, key)
            mean, std, fold_means = _fold_mean_std(values, folds)
            per_variant[label] = {"mean": mean, "std": std, "fold_means": fold_means}
        report["fold_variance"][v] = per_variant
        metric_means[v] = {label: (per_variant[label]["mean"], per_variant[label]["std"]) for label, _ in METRIC_KEYS}
        fold_rows.append([
            v,
            _format_pm(*metric_means[v]["J"], digits=3),
            _format_pm(*metric_means[v]["H"], digits=3),
            _format_pm(*metric_means[v]["DocM"], digits=3),
            _format_pm(*metric_means[v]["PgM"], digits=3),
            _format_pm(*metric_means[v]["Att"], digits=3),
            _format_pm(*metric_means[v]["Lat"], digits=2),
        ])

    # === 2. Paired bootstrap CIs on Judge gap vs Full ===
    full_j = _metric_vector(cells["full"], "llm_judge_score")
    for v in VARIANTS:
        if v == "full":
            continue
        var_j = _metric_vector(cells[v], "llm_judge_score")
        ci = _paired_bootstrap_ci(full_j, var_j, n_boot=args.n_boot, seed=args.seed)
        report["bootstrap_judge_gap"][v] = ci

    # === 3. Conditional J | Att + attempted overlap with Full ===
    full_att = _attempted_indices(cells["full"])
    for v in VARIANTS:
        report["conditional_precision_j_given_att"][v] = _conditional_precision_j_given_att(cells[v])
        if v == "full":
            report["attempted_overlap_with_full"][v] = {"jaccard": 1.0, "intersection": len(full_att), "union": len(full_att), "full_only": 0, "var_only": 0}
        else:
            report["attempted_overlap_with_full"][v] = _attempted_overlap(full_att, _attempted_indices(cells[v]))

    # === 4. Table 2 LaTeX rows ===
    table2_rows = {}
    for v in VARIANTS:
        if v == "full":
            continue
        j = metric_means[v]["J"]
        pgm = metric_means[v]["PgM"]
        att = metric_means[v]["Att"]
        table2_rows[v] = _table2_row(DISPLAY_NAME[v], j, pgm, att)
    report["table2_rows"] = table2_rows

    # === Render markdown ===
    md_lines: list[str] = []
    md_lines.append("# Q-/Q+ Direction-Split Ablation — Fold Analysis")
    md_lines.append("")
    md_lines.append(f"- queries: {n}  |  folds: {args.folds}  |  seed: {args.seed}  |  bootstrap B: {args.n_boot}")
    md_lines.append(f"- variants compared: {', '.join(VARIANTS)}")
    md_lines.append("")
    md_lines.append("## 1. Fold-level mean ± std (5-fold, seed=42)")
    md_lines.append("")
    md_lines.append("```")
    md_lines.append(_ascii_table(fold_rows, ["variant", "J", "H", "DocM", "PgM", "Att", "Lat (s)"]))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## 2. Paired bootstrap CIs on Judge gap (Full − variant)")
    md_lines.append("")
    md_lines.append("```")
    boot_rows = []
    for v in VARIANTS:
        if v == "full":
            continue
        ci = report["bootstrap_judge_gap"][v]
        boot_rows.append([
            v,
            f"{ci['observed_gap']:+.3f}",
            f"[{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}]",
            f"{ci['p_two_sided']:.4f}",
        ])
    md_lines.append(_ascii_table(boot_rows, ["variant", "J_full − J_var", "95% CI", "p (2-sided)"]))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## 3. Conditional precision (J | Att) + Attempted-set overlap with Full")
    md_lines.append("")
    md_lines.append("```")
    cond_rows = []
    for v in VARIANTS:
        cp = report["conditional_precision_j_given_att"][v]
        ov = report["attempted_overlap_with_full"][v]
        cond_rows.append([
            v,
            f"{cp:.3f}",
            f"{ov['jaccard']:.3f}",
            f"{ov['intersection']}/{ov['union']}",
            f"{ov['full_only']}",
            f"{ov['var_only']}",
        ])
    md_lines.append(_ascii_table(cond_rows, ["variant", "J|Att", "Jaccard(Att)", "∩/∪", "full-only", "var-only"]))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## 4. Table 2 rows (paste into prehypo.tex around l.393–409)")
    md_lines.append("")
    md_lines.append("```latex")
    for v in VARIANTS:
        if v == "full":
            continue
        md_lines.append(table2_rows[v])
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("Columns above are (J, PgM, Att). If the existing Table 2 uses a different triple, edit the metric tuple in scripts/ablation_analysis.py:_table2_row caller.")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote: {out_md}")
    print(f"Wrote: {args.out_json}\n")

    # Echo a console summary
    print("=== Fold means (mean ± std) ===")
    print(_ascii_table(fold_rows, ["variant", "J", "H", "DocM", "PgM", "Att", "Lat (s)"]))
    print()
    print("=== Bootstrap J-gap vs Full ===")
    print(_ascii_table(boot_rows, ["variant", "J_full − J_var", "95% CI", "p (2-sided)"]))
    print()
    print("=== Table 2 rows ===")
    for v in VARIANTS:
        if v == "full":
            continue
        print(table2_rows[v])


if __name__ == "__main__":
    main()
