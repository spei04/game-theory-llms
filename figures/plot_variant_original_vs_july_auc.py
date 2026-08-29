#!/usr/bin/env python3
"""
For every prompt variant, compare the gen-0 (initial) coded-strategy
self-cooperation AUC between the earlier full runs ("original") and the
July 2026 strategies-only 200-run batches ("current").

Gen 0 = the initial, pre-evolution strategies (generation==1 in the data).
Original = figures/reactivity_plots/{variant}/          (full runs, May-Jun)
Current  = figures/reactivity_plots/{variant}_200/      (Jul 16 batches)
For the baseline prompt, the earlier full runs are statistically
indistinguishable from the April original setup.

Writes figures/compare_setups/gen0_auc_original_vs_july_by_variant.png

Usage:
    python figures/plot_variant_original_vs_july_auc.py
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_auc import eval_strategy, auc_normalised

OUTPUT = "figures/compare_setups/gen0_auc_original_vs_july_by_variant.png"

MODELS = ["claude", "gpt", "gemini"]
MODEL_LABELS = {"claude": "Claude 3.5 Sonnet", "gpt": "GPT-4o",
                "gemini": "Gemini 2.5 Flash"}

VARIANTS = [
    ("baseline",          "neutral_baseline"),
    ("neutral (Claude)",   "neutral_claude"),
    ("neutral (GPT)",      "neutral_gpt"),
    ("neutral (Gemini)",   "neutral_gemini"),
    ("neutral (DeepSeek)", "neutral_deepseek"),
    ("aggressive",         "aggressive"),
    ("collaborative",      "collaborative"),
]

# Validated categorical pair (dataviz slots 1 and 3, light mode)
C_ORIG = "#2a78d6"
C_JULY = "#e87ba4"

SURFACE  = "#fcfcfb"
INK      = "#0b0b0b"
INK_2    = "#52514e"
MUTED    = "#898781"
GRID     = "#e1e0d9"
BASELINE = "#c3c2b7"
CONNECT  = "#c3c2b7"


def gen0_aucs(analysis_dir, model):
    path = os.path.join("figures/reactivity_plots", analysis_dir, model,
                        f"strategy_analysis_{model}.json")
    if not os.path.exists(path):
        return np.array([])
    with open(path) as f:
        records = json.load(f)["agent_records"]
    vals = []
    for rec in records:
        if rec.get("generation") != 1 or not rec.get("is_specified"):
            continue
        code = rec.get("strategy_code")
        if not code:
            continue
        ys = eval_strategy(code)
        if ys is not None:
            vals.append(auc_normalised(ys))
    return np.array(vals)


def _stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    cells = {}
    for model in MODELS:
        for _, vdir in VARIANTS:
            cells[(model, vdir)] = (gen0_aucs(vdir, model),
                                    gen0_aucs(vdir + "_200", model))

    means = [np.mean(v) for pair in cells.values() for v in pair if len(v)]
    x_lo, x_hi = min(means) - 0.11, max(means) + 0.11

    fig, axes = plt.subplots(1, len(MODELS), figsize=(12.5, 4.6),
                             sharey=True, sharex=True, facecolor=SURFACE)
    y_pos = np.arange(len(VARIANTS))[::-1]

    for ax, model in zip(axes, MODELS):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0)
        ax.grid(axis="x", color=GRID, linewidth=1)
        ax.set_axisbelow(True)

        for (label, vdir), y in zip(VARIANTS, y_pos):
            orig, july = cells[(model, vdir)]
            if len(july) == 0:
                m0 = np.mean(orig)
                ax.errorbar([m0], [y], xerr=[np.std(orig) / np.sqrt(len(orig))],
                            fmt="o", color=C_ORIG, ms=8, mec=SURFACE, mew=2,
                            ecolor=C_ORIG, elinewidth=1.2, capsize=2, zorder=3)
                ax.text(m0 + 0.02, y, "Jul: no coded data", ha="left",
                        va="center", fontsize=7, color=MUTED, style="italic")
                continue

            m0, m1 = np.mean(orig), np.mean(july)
            se0 = np.std(orig) / np.sqrt(len(orig))
            se1 = np.std(july) / np.sqrt(len(july))
            _, p = sps.ttest_ind(orig, july, equal_var=False)

            ax.plot([m0, m1], [y, y], color=CONNECT, linewidth=1.4,
                    zorder=2, solid_capstyle="round")
            # Original dot is larger so it stays visible as a halo when the
            # July dot lands on top of it (delta ~ 0).
            ax.errorbar([m0], [y], xerr=[se0], fmt="o", color=C_ORIG, ms=11,
                        mec=SURFACE, mew=2, ecolor=C_ORIG, elinewidth=1.2,
                        capsize=2, zorder=3)
            ax.errorbar([m1], [y], xerr=[se1], fmt="o", color=C_JULY, ms=8,
                        mec=SURFACE, mew=1.5, ecolor=C_JULY, elinewidth=1.2,
                        capsize=2, zorder=4)

            note = f"{m1 - m0:+.2f}{_stars(p)}"
            if m1 >= m0:
                ax.text(m1 + max(se1, 0.008) + 0.012, y, note, ha="left",
                        va="center", fontsize=7.5, color=INK_2)
            else:
                ax.text(m1 - max(se1, 0.008) - 0.012, y, note, ha="right",
                        va="center", fontsize=7.5, color=INK_2)

        ax.set_title(MODEL_LABELS[model], fontsize=10.5, color=INK, pad=8)
        ax.set_xlabel("Self-cooperation AUC (0–1)", fontsize=8.5, color=INK_2)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([v[0] for v in VARIANTS], fontsize=9)
    axes[0].set_xlim(x_lo, x_hi)
    tick_lo = np.ceil(x_lo * 10) / 10
    axes[0].set_xticks(np.arange(tick_lo, x_hi + 1e-9, 0.1))

    fig.suptitle("Gen-0 (initial) coded-strategy cooperation: "
                 "original vs current (July batch), by prompt variant",
                 x=0.005, ha="left", fontsize=12.5, color=INK)
    fig.text(0.005, 0.90, "Mean AUC ± SEM · original = earlier full runs "
                          "(May–Jun) · current = Jul strategies-only 200-run "
                          "batches · label = Jul − original "
                          "(* p<0.05, ** p<0.01, *** p<0.001, Welch)",
             fontsize=8.5, color=INK_2)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=C_ORIG, ms=8,
                   mec=SURFACE, mew=2, label="Original (full runs)"),
        plt.Line2D([], [], marker="o", ls="", color=C_JULY, ms=8,
                   mec=SURFACE, mew=2, label="Current (Jul 200-run batch)"),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False,
               fontsize=9, ncols=2, bbox_to_anchor=(0.995, 1.0),
               labelcolor=INK_2)

    plt.tight_layout(rect=(0, 0, 1, 0.87))
    plt.savefig(OUTPUT, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
