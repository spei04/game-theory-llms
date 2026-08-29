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
        [--no_bands]
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
                                   SURFACE, INK, INK_2, MUTED)

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
    parser.add_argument("--generation", type=int, default=1,
                        help="Generation to plot (1 = initial/'gen 0', default). "
                             "Use 10 for the final generation of full-game variants; "
                             "the *_200 strategies-only variants have gen 1 only.")
    args = parser.parse_args()
    gen_label = "Generation-0" if args.generation == 1 else f"Generation-{args.generation}"
    if args.generation != 1 and args.output == parser.get_default("output"):
        args.output = ("figures/gen0_prompt_comparison/"
                       f"gen{args.generation}_reactivity_all_variants.png")

    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 5.2),
                             sharey=True, facecolor=SURFACE)

    for ax, model in zip(axes, MODELS):
        style_axes(ax, grid_axis="both")
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
        ax.set_title(MODEL_LABELS[model], fontsize=11.5, color=INK)
        ax.set_xlabel("Pair's donation % in previous round", fontsize=9.5, color=INK_2)
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.legend(fontsize=7, loc="upper left", frameon=False, labelcolor=INK_2)

    axes[0].set_ylabel("My donation %", fontsize=9.5, color=INK_2)
    fig.suptitle(f"{gen_label} coded strategies by prompt variant "
                 "(mean curve across agents)", fontsize=13, color=INK)
    band_note = "" if args.no_bands else " · band: ±1 SD across agents"
    fig.text(0.99, 0.01,
             f"Curves evaluated continuously on a 500-point grid (no binning)"
             f"{band_note}", ha="right", fontsize=7, color=MUTED)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
