#!/usr/bin/env python3
"""
Visualize, for each prompt variant, how the coded-strategy self-cooperation
AUC changes between the first generation (gen 0 in figure terminology,
generation==1 in the data) and the final generation (gen 10).

One figure, three panels (one per model). Each row is a prompt variant with
a light dot (gen 0), a dark dot (gen 10), a connector, and the signed delta.

Reads figures/reactivity_plots/{variant}/{model}/strategy_analysis_{model}.json
Writes figures/compare_setups/gen0_vs_gen10_auc_by_variant.png

Usage:
    python figures/plot_gen0_gen10_variant_auc.py
"""

import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_auc import eval_strategy, auc_normalised

OUTPUT = "figures/compare_setups/gen0_vs_gen10_auc_by_variant.png"

MODELS = ["claude", "gpt", "gemini"]
MODEL_LABELS = {"claude": "Claude 3.5 Sonnet", "gpt": "GPT-4o",
                "gemini": "Gemini 2.5 Flash"}

VARIANTS = [
    ("baseline",         "neutral_baseline"),
    ("neutral (Claude)",  "neutral_claude"),
    ("neutral (GPT)",     "neutral_gpt"),
    ("neutral (Gemini)",  "neutral_gemini"),
    ("neutral (DeepSeek)", "neutral_deepseek"),
    ("aggressive",        "aggressive"),
    ("collaborative",     "collaborative"),
]

# Two-step ordinal blue ramp (validated): light = gen 0, dark = gen 10
C_GEN0  = "#86b6ef"
C_GEN10 = "#1c5cab"

SURFACE  = "#fcfcfb"
INK      = "#0b0b0b"
INK_2    = "#52514e"
MUTED    = "#898781"
GRID     = "#e1e0d9"
BASELINE = "#c3c2b7"
CONNECT  = "#c3c2b7"


def gen_aucs(variant_dir, model, generation):
    path = os.path.join("figures/reactivity_plots", variant_dir, model,
                        f"strategy_analysis_{model}.json")
    with open(path) as f:
        records = json.load(f)["agent_records"]
    vals = []
    for rec in records:
        if rec.get("generation") != generation or not rec.get("is_specified"):
            continue
        code = rec.get("strategy_code")
        if not code:
            continue
        ys = eval_strategy(code)
        if ys is not None:
            vals.append(auc_normalised(ys))
    return np.array(vals)


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    # Precompute all cells so the shared x-range can cover every dot + label.
    cells = {}
    for model in MODELS:
        for _, vdir in VARIANTS:
            v0, v10 = gen_aucs(vdir, model, 1), gen_aucs(vdir, model, 10)
            cells[(model, vdir)] = (
                np.mean(v0), np.std(v0) / np.sqrt(len(v0)),
                np.mean(v10), np.std(v10) / np.sqrt(len(v10)))

    lo = min(min(m0 - se0, m10 - se10) for m0, se0, m10, se10 in cells.values())
    hi = max(max(m0 + se0, m10 + se10) for m0, se0, m10, se10 in cells.values())
    x_lo, x_hi = lo - 0.09, hi + 0.09   # room for the delta labels

    fig, axes = plt.subplots(1, len(MODELS), figsize=(12.5, 4.6),
                             sharey=True, sharex=True, facecolor=SURFACE)

    y_pos = np.arange(len(VARIANTS))[::-1]   # first variant on top

    for ax, model in zip(axes, MODELS):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0)
        ax.grid(axis="x", color=GRID, linewidth=1)
        ax.set_axisbelow(True)

        for (label, vdir), y in zip(VARIANTS, y_pos):
            m0, se0, m10, se10 = cells[(model, vdir)]

            ax.plot([m0, m10], [y, y], color=CONNECT, linewidth=1.4,
                    zorder=2, solid_capstyle="round")
            ax.errorbar([m0], [y], xerr=[se0], fmt="o", color=C_GEN0,
                        ms=8, mec=SURFACE, mew=2, ecolor=C_GEN0,
                        elinewidth=1.2, capsize=2, zorder=3)
            ax.errorbar([m10], [y], xerr=[se10], fmt="o", color=C_GEN10,
                        ms=8, mec=SURFACE, mew=2, ecolor=C_GEN10,
                        elinewidth=1.2, capsize=2, zorder=4)

            # signed delta beside the gen-10 end, nudged away from the dumbbell
            delta = m10 - m0
            if m10 >= m0:
                ax.text(m10 + max(se10, 0.008) + 0.012, y, f"{delta:+.2f}",
                        ha="left", va="center", fontsize=7.5, color=INK_2)
            else:
                ax.text(m10 - max(se10, 0.008) - 0.012, y, f"{delta:+.2f}",
                        ha="right", va="center", fontsize=7.5, color=INK_2)

        ax.set_title(MODEL_LABELS[model], fontsize=10.5, color=INK, pad=8)
        ax.set_xlabel("Self-cooperation AUC (0–1)", fontsize=8.5, color=INK_2)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([v[0] for v in VARIANTS], fontsize=9)
    axes[0].set_xlim(x_lo, x_hi)
    tick_lo = np.ceil(x_lo * 10) / 10
    axes[0].set_xticks(np.arange(tick_lo, x_hi + 1e-9, 0.1))

    fig.suptitle("Coded-strategy cooperation: gen 0 vs gen 10, by prompt variant",
                 x=0.005, ha="left", fontsize=12.5, color=INK)
    fig.text(0.005, 0.90, "Mean AUC of coded gen-0 (initial) and gen-10 (final) "
                          "strategies · whiskers = ±SEM · label = gen-10 − gen-0",
             fontsize=9, color=INK_2)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=C_GEN0, ms=8,
                   mec=SURFACE, mew=2, label="Gen 0 (initial)"),
        plt.Line2D([], [], marker="o", ls="", color=C_GEN10, ms=8,
                   mec=SURFACE, mew=2, label="Gen 10 (final)"),
    ]
    fig.legend(handles=handles, loc="upper right", frameon=False,
               fontsize=9, ncols=2, bbox_to_anchor=(0.995, 1.0),
               labelcolor=INK_2)

    plt.tight_layout(rect=(0, 0, 1, 0.88))
    plt.savefig(OUTPUT, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
