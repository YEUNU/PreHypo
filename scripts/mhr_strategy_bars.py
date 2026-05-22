"""Cross-strategy headline bars for MultiHop-RAG sample200, in the house style of
fig/fig_direction_split.png (5-fold mean ± std, value labels, PreHypo highlighted).

One panel per headline metric; bars = the 4 strategies. Error bars are the 5-fold
(seed 42) std, matching the direction-split ablation figure. PreHypo is rendered
dark + black-edged (like "Full" in that figure); baselines are light green.

Usage:
  python scripts/mhr_strategy_bars.py \
    --prehypo data/results/<new>/prehypo/multihoprag/prehypo_multihoprag.json \
    --baselines <base>/{naive,hoprag,ms_graphrag}/multihoprag/*.json \
    --fig fig/mhr_strategy_bars.png
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

# (key, display label, lower_is_better, gold_only) — gold_only metrics drop null queries.
METRICS = [
    ("llm_judge_score", "LLM-Judge Correctness", False, False),
    ("mrr@10", r"MRR@10", False, True),
    ("hits@4", r"Hits@4", False, True),
    ("evidence_doc_recall", "Evidence Doc Recall", False, True),
]
# display order: PreHypo first (highlighted), then baselines
ORDER = ["prehypo", "ms_graphrag", "hoprag", "naive"]
LABELS = {"prehypo": "PreHypo", "ms_graphrag": "MS-GraphRAG", "hoprag": "HopRAG", "naive": "Naive"}
SEED = 42
K = 5


def _load(path: str) -> tuple[str, list[dict]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["strategy"], [r for r in data["details"] if isinstance(r, dict)]


def _fold_mean_std(rows: list[dict], metric: str, gold_only: bool) -> tuple[float, float]:
    """5-fold (seed 42) mean of per-fold means, and std across folds."""
    vals = []
    for r in rows:
        if gold_only and r.get("category") == "null_query":
            continue
        v = r.get(metric)
        if v is None or (metric in ("llm_judge_score", "hallucination") and v < 0):
            continue
        vals.append(float(v))
    if not vals:
        return 0.0, 0.0
    idx = list(range(len(vals)))
    random.Random(SEED).shuffle(idx)
    n, base = len(idx), len(idx) // K
    fold_means = []
    start = 0
    for f in range(K):
        size = base + (1 if f < n % K else 0)
        fold = idx[start:start + size]
        start += size
        if fold:
            fold_means.append(np.mean([vals[i] for i in fold]))
    return float(np.mean(fold_means)), float(np.std(fold_means))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prehypo", required=True)
    ap.add_argument("--baselines", nargs="+", required=True)
    ap.add_argument("--fig", default="fig/mhr_strategy_bars.png")
    args = ap.parse_args()

    rows_by_strat = {}
    s, r = _load(args.prehypo)
    rows_by_strat[s] = r
    for p in args.baselines:
        s, r = _load(p)
        rows_by_strat[s] = r

    strategies = [s for s in ORDER if s in rows_by_strat]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
    })

    DARK = "#2E8B6F"     # PreHypo (highlight)
    LIGHT = "#9ED4BF"    # baselines
    ncol = len(METRICS)
    fig, axes = plt.subplots(1, ncol, figsize=(3.4 * ncol, 4.2))
    if ncol == 1:
        axes = [axes]

    for ax, (key, label, lower, gold_only) in zip(axes, METRICS):
        means, stds = [], []
        for st in strategies:
            m, sd = _fold_mean_std(rows_by_strat[st], key, gold_only)
            means.append(m)
            stds.append(sd)
        x = np.arange(len(strategies))
        for i, st in enumerate(strategies):
            highlight = (st == "prehypo")
            ax.bar(x[i], means[i], width=0.72,
                   color=DARK if highlight else LIGHT,
                   edgecolor="black", linewidth=1.8 if highlight else 0.8,
                   zorder=3)
            ax.errorbar(x[i], means[i], yerr=stds[i], fmt="none",
                        ecolor="black", elinewidth=1.5, capsize=4, capthick=1.5, zorder=4)
            ax.text(x[i], means[i] + stds[i] + 0.012, f"{means[i]:.3f}",
                    ha="center", va="bottom", fontsize=10)
        ax.set_ylabel(label, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[s] for s in strategies], rotation=20, ha="right", fontsize=10)
        ax.set_ylim(0, max(0.001, max(m + s for m, s in zip(means, stds))) * 1.22)
        ax.yaxis.grid(True, ls=":", color="#bbbbbb", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=9)

    fig.suptitle("MultiHop-RAG cross-strategy (sample200, 5-fold mean $\\pm$ std)",
                 fontsize=15, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = Path(args.fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"saved -> {fig_path}")


if __name__ == "__main__":
    main()
