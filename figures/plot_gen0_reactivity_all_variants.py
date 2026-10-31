#!/usr/bin/env python3
"""
Overlay generation-0 (initial, t=0) coded-strategy reactivity curves for all
prompt variants on the same plot, one panel per model.

Each agent's strategy_fn is evaluated continuously over pair_prev_pct in
[0, 100] on a 500-point grid (X_VALUES from compute_auc — no binning); the
plot shows, per prompt variant, the pointwise mean curve across agents with
an optional ±1 SD band.

Reads the same per-variant analysis JSONs as compare_gen0_variants.py
(figures/reactivity_plots/{variant}/{model}/strategy_analysis_{model}.json),
generation-1 records only (= the initial strategies).

Usage:
    python figures/plot_gen0_reactivity_all_variants.py \
        [--variants neutral_baseline neutral_claude ...] \
        [--analysis_dir figures/reactivity_plots] \
        [--output figures/gen0_prompt_comparison/gen0_reactivity_all_variants.png] \
        [--no_bands] [--paper] [--font_scale 1.6]

The paper's Figure 1 uses the full-game variants:
    --variants neutral_baseline neutral_claude neutral_gpt neutral_gemini
    neutral_deepseek --paper --font_scale 1.6 [--generation 10]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_auc import X_VALUES
from compare_gen0_variants import (load_variant, style_axes, MODELS,
                                   MODEL_LABELS, VARIANT_LABELS, VARIANT_COLORS,
                                   SURFACE, INK, INK_2)

# Panel titles: short model names (exact API ids are documented in the paper).
SHORT_MODEL_LABELS = {"claude": "Claude", "gpt": "GPT", "gemini": "Gemini"}

# Type sizes. Sized for a figure that is shrunk to column width in the paper,
# where the previous 7-9.5pt labels became unreadable. --font_scale rescales
# the whole set at once.
FS = {
    "suptitle": 20,
    "panel_title": 17,
    "axis_label": 15,
    "tick": 14,
    "legend": 12.5,
}

# Default: the July-era strategies-only grid (~200 runs/model/variant).
# The May-era full-game variants (plain names) stay available via --variants.
ALL_VARIANTS = [
    "neutral_baseline_200", "neutral_claude_200", "neutral_gpt_200",
    "neutral_gemini_200", "neutral_deepseek_200", "collaborative_200",
    "aggressive_200",
]
VARIANT_CHOICES = ALL_VARIANTS + [
    "neutral_baseline", "gen0_strategies", "neutral_claude", "neutral_gpt",
    "neutral_gemini", "neutral_deepseek", "collaborative", "aggressive",
]


def main():
    parser = argparse.ArgumentParser(
        description="Overlay gen-0 reactivity curves for all prompt variants.")
    parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS,
                        choices=VARIANT_CHOICES,
                        help="Prompt variants to overlay (default: all)")
    parser.add_argument("--analysis_dir", default="figures/reactivity_plots")
    parser.add_argument("--output",
                        default="figures/gen0_prompt_comparison/gen0_reactivity_all_variants.png")
    parser.add_argument("--no_bands", action="store_true",
                        help="Draw mean curves only, without the ±1 SD bands")
    parser.add_argument("--paper", action="store_true",
                        help="Print layout: one shared legend below the panels, "
                             "shorter axis labels, no suptitle (the caption says it).")
    parser.add_argument("--font_scale", type=float, default=1.0,
                        help="Multiply every font size by this factor.")
    parser.add_argument("--generation", type=int, default=1,
                        help="Generation to plot (1 = initial/'gen 0', default). "
                             "Use 10 for the final generation of full-game variants; "
                             "the *_200 strategies-only variants have gen 1 only.")
    args = parser.parse_args()
    fs = {k: v * args.font_scale for k, v in FS.items()}
    gen_label = f"Generation-{args.generation}"
    if args.generation != 1 and args.output == parser.get_default("output"):
        args.output = ("figures/gen0_prompt_comparison/"
                       f"gen{args.generation}_reactivity_all_variants.png")

    fig, axes = plt.subplots(1, len(MODELS), figsize=(16.5, 6.0),
                             sharey=True, facecolor=SURFACE)

    for ax, model in zip(axes, MODELS):
        style_axes(ax, grid_axis="both")
        # style_axes fixes labelsize at 9; override after it runs.
        ax.tick_params(labelsize=fs["tick"])
        for variant in args.variants:
            d = load_variant(args.analysis_dir, variant, model,
                             generation=args.generation)
            if d is None or not d["curves"].size:
                print(f"  [{variant}/{model}] no evaluable gen-{args.generation} "
                      "strategies — skipped")
                continue
            mean_y = d["curves"].mean(axis=0)
            color = VARIANT_COLORS[variant]
            ax.plot(X_VALUES, mean_y, color=color, linewidth=2.0,
                    label=VARIANT_LABELS[variant])
            if not args.no_bands:
                std_y = d["curves"].std(axis=0)
                ax.fill_between(X_VALUES,
                                np.clip(mean_y - std_y, 0, 100),
                                np.clip(mean_y + std_y, 0, 100),
                                alpha=0.08, color=color, linewidth=0)
        ax.set_title(SHORT_MODEL_LABELS.get(model, model),
                     fontsize=fs["panel_title"], color=INK)
        ax.set_xlabel("Partner's last donation (%)" if args.paper
                      else "Pair's donation % in previous round",
                      fontsize=fs["axis_label"], color=INK_2)
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 108)
        if not args.paper:
            ax.legend(fontsize=fs["legend"], loc="upper left", frameon=False,
                      labelcolor=INK_2, handlelength=1.6, handletextpad=0.5,
                      borderaxespad=0.2, labelspacing=0.3)

    axes[0].set_ylabel("Own donation (%)" if args.paper else "My donation %",
                       fontsize=fs["axis_label"], color=INK_2)
    if args.paper:
        handles, labels = axes[0].get_legend_handles_labels()
        labels = [l.replace("Neutral (", "").rstrip(")") for l in labels]
        fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                   fontsize=fs["legend"], frameon=False, labelcolor=INK_2,
                   bbox_to_anchor=(0.5, -0.02), handlelength=1.6)
        plt.tight_layout(rect=(0, 0.09, 1, 1))
    else:
        fig.suptitle(f"{gen_label} coded strategies by prompt variant "
                     "(mean curve across agents)", fontsize=fs["suptitle"], color=INK)
        plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
