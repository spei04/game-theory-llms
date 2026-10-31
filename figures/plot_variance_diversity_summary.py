#!/usr/bin/env python3
"""
Summary figure: cultural vs direct evolution — variance shares + diversity.

Left panel: four stacked bars (cultural evolution gen 1 / gen 10, direct
evolution beta=50 gen 1 / gen 10). Each bar splits the across-run variance of
per-run cooperation into two components (type-II SS, computed on the fly):
    executor model  |  run stochasticity (residual)
Both settings use the hand-written neutral baseline prompt, so the comparison
is apples-to-apples: cultural = simulations/neutral_baseline/{model},
direct = simulations/direct_evo_50_neutral/{model} (--direct_dir to change;
direct_evo_50_old used an older wording of the baseline prompt).

Right panel: within-run strategy-diversity metrics at gen 1 -> gen 10
(lexical similarity, semantic similarity, distinct strategies, verbatim
pairs), values from the strategy-diversity analysis (ranges across the three
executors).

Output: figures/run_variation/variance_shares_cultural_vs_direct.png

Usage:
    python figures/plot_variance_diversity_summary.py
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variance_decomposition import sse

C_MODEL = "#2a78d6"
C_RUN   = "#c3c2b7"
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

MODELS = ["claude", "gpt", "gemini"]

# Diversity metrics (gen 1 -> gen 10), ranges across the three executors,
# from the lexical/semantic strategy-diversity analysis (word-Levenshtein
# similarity; embedding cosine; unique strategy strings; identical pairs).
DIVERSITY_ROWS = [
    ("Lexical similarity",      "0.20–0.45 → 0.23–0.72", "0.13–0.46 → 0.82–0.89"),
    ("Semantic similarity",     "0.79–0.91 → 0.95–0.97", "0.80–0.91 → 0.97–0.98"),
    ("Distinct strategies /12", "12.0 → 11.7–12.0",      "12.0 → 1.4–1.6"),
    ("Verbatim pairs (gen 10)", "0.0–0.7%",              "77–86%"),
]


def coop_by_gen(path):
    d = json.load(open(path))
    pcts = defaultdict(list)
    for r in d.get("agents_data", []):
        if not r.get("is_donor"):
            continue
        don = r.get("donated", 0) or 0
        rem = r.get("resources", 0) or 0
        if don + rem > 0:
            pcts[r["current_generation"]].append(100.0 * don / (don + rem))
    return {g: float(np.mean(v)) for g, v in pcts.items() if v}


def load_setting(dir_tpl):
    rows = []
    for m in MODELS:
        for p in sorted(glob.glob(dir_tpl.format(model=m) + "/Final_Accum*.json")):
            c = coop_by_gen(p)
            if c:
                rows.append({"model": m, "g1": c.get(1, np.nan), "g10": c.get(10, np.nan)})
    return pd.DataFrame(rows)


def model_run_shares(df, col):
    """(model %, run %) from type-II SS with model as the only factor."""
    sub = df.dropna(subset=[col])
    y = sub[col].to_numpy(float)
    M = pd.get_dummies(sub["model"], drop_first=True).to_numpy(float)
    ss_total = float(((y - y.mean()) ** 2).sum())
    ss_run = sse(M, y)
    ss_model = max(ss_total - ss_run, 0.0)
    tot = ss_model + ss_run
    return ss_model / tot * 100, ss_run / tot * 100, len(sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="figures/run_variation/"
                                        "variance_shares_cultural_vs_direct.png")
    ap.add_argument("--direct_dir", default="simulations/direct_evo_50_neutral")
    args = ap.parse_args()

    cultural = load_setting("simulations/neutral_baseline/{model}")
    direct = load_setting(args.direct_dir + "/{model}")

    bars = []
    for label, df, col in [("Cultural\nevolution\ngen 1", cultural, "g1"),
                           ("Cultural\nevolution\ngen 10", cultural, "g10"),
                           ("Direct\nevolution\ngen 1", direct, "g1"),
                           ("Direct\nevolution\ngen 10", direct, "g10")]:
        m, r, n = model_run_shares(df, col)
        bars.append((label, m, r, n))
        print(f"{label.replace(chr(10), ' '):28s} model={m:5.1f}%  run={r:5.1f}%  (n={n})")

    fig, ax = plt.subplots(figsize=(6.2, 4.9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    xs = np.array([0, 1, 2.35, 3.35])   # gap between the two settings
    for x, (label, m, r, _) in zip(xs, bars):
        ax.bar(x, m, width=0.72, color=C_MODEL, zorder=3)
        ax.bar(x, r, width=0.72, bottom=m, color=C_RUN, zorder=3)
        ax.text(x, m / 2, f"{m:.0f}%", ha="center", va="center",
                fontsize=9, color="white", zorder=4)
        ax.text(x, m + r / 2, f"{r:.0f}%", ha="center", va="center",
                fontsize=9, color=INK, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], fontsize=9, color=INK_2)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of across-run cooperation variance (%)",
                  fontsize=10, color=INK)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_MODEL),
               plt.Rectangle((0, 0), 1, 1, color=C_RUN)]
    ax.legend(handles, ["model", "repeated runs"],
              fontsize=9, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=2, labelcolor=INK_2)

    ax.set_title("Cultural vs direct evolution (β=50), baseline prompt",
                 fontsize=11, color=INK, pad=10)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
