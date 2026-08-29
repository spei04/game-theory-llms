#!/usr/bin/env python3
"""
Overlay coded-strategy reactivity curves for all prompt variants at the start
and the end of evolution: one column per model, one row per generation.

Row 1 = generation 1 in the data (the initial strategies, t=0, before any
evolution); row 2 = generation 10 (after ten rounds of selection). Comparing
the two rows shows how much — if at all — evolution moves the strategies away
from where the prompt put them.

Each agent's strategy_fn is evaluated continuously over pair_prev_pct in
[0, 100] on a 500-point grid (X_VALUES from compute_auc — no binning); each
panel shows, per prompt variant, the pointwise mean curve across agents with
an optional +/-1 SD band.

Reads the same per-variant analysis JSONs as compare_gen0_variants.py
(figures/reactivity_plots/{variant}/{model}/strategy_analysis_{model}.json).
Note this needs the full-game variants (plain names, 10 generations); the
"_200" strategies-only reruns contain generation 1 only.

Usage:
    python figures/plot_reactivity_gen0_gen10.py \
        [--variants neutral_baseline neutral_claude ...] \
        [--generations 1 10] \
        [--analysis_dir figures/reactivity_plots] \
        [--output figures/gen0_prompt_comparison/reactivity_gen0_gen10_all_variants.png] \
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

# Full-game variants only: the "_200" reruns stopped after generation 1, so
# they have no gen-10 strategies to plot.
ALL_VARIANTS = [
    "neutral_baseline", "neutral_claude", "neutral_gpt", "neutral_gemini",
    "neutral_deepseek", "collaborative", "aggressive",
]


def gen_label(g):
    """Data generation number -> the label used in the paper (gen 1 = t=0)."""
    return "Generation 0 (initial strategies)" if g == 1 else f"Generation {g}"


def main():
    parser = argparse.ArgumentParser(
        description="Overlay reactivity curves per prompt variant, gen 0 vs gen 10.")
    parser.add_argument("--variants", nargs="+", default=ALL_VARIANTS,
                        choices=sorted(VARIANT_LABELS),
                        help="Prompt variants to overlay (default: all full-game variants)")
    parser.add_argument("--generations", nargs="+", type=int, default=[1, 10],
                        help="Generations to stack as rows (data numbering: 1 = t=0)")
    parser.add_argument("--analysis_dir", default="figures/reactivity_plots")
    parser.add_argument("--output",
                        default="figures/gen0_prompt_comparison/reactivity_gen0_gen10_all_variants.png")
    parser.add_argument("--no_bands", action="store_true",
                        help="Draw mean curves only, without the +/-1 SD bands")
    args = parser.parse_args()

    gens = args.generations
    fig, axes = plt.subplots(len(gens), len(MODELS),
                             figsize=(15, 5.0 * len(gens)),
                             sharex=True, sharey=True, facecolor=SURFACE,
                             squeeze=False)

    for row, gen in enumerate(gens):
        print(f"\n-- {gen_label(gen)}")
        for col, model in enumerate(MODELS):
            ax = axes[row][col]
            style_axes(ax, grid_axis="both")
            for variant in args.variants:
                d = load_variant(args.analysis_dir, variant, model, generation=gen)
                if d is None or not d["curves"].size:
                    print(f"  [{variant}/{model}] no evaluable gen-{gen} "
                          f"strategies — skipped")
                    continue
                mean_y = d["curves"].mean(axis=0)
                color = VARIANT_COLORS[variant]
                ax.plot(X_VALUES, mean_y, color=color, linewidth=2.0,
                        label=f"{VARIANT_LABELS[variant]} (n={len(d['curves'])})")
                if not args.no_bands:
                    std_y = d["curves"].std(axis=0)
                    ax.fill_between(X_VALUES,
                                    np.clip(mean_y - std_y, 0, 100),
                                    np.clip(mean_y + std_y, 0, 100),
                                    alpha=0.08, color=color, linewidth=0)
            if row == 0:
                ax.set_title(MODEL_LABELS[model], fontsize=11.5, color=INK)
            if row == len(gens) - 1:
                ax.set_xlabel("Pair's donation % in previous round",
                              fontsize=9.5, color=INK_2)
            ax.set_xlim(-2, 102)
            ax.set_ylim(-2, 102)
            ax.legend(fontsize=6.5, loc="upper left", frameon=False,
                      labelcolor=INK_2)

        # Row label: generation, on the left-hand panel of the row.
        axes[row][0].set_ylabel(f"{gen_label(gen)}\n\nMy donation %",
                                fontsize=9.5, color=INK_2)

    fig.suptitle("Coded strategies by prompt variant, before and after evolution "
                 "(mean curve across agents)", fontsize=13, color=INK)
    band_note = "" if args.no_bands else " · band: ±1 SD across agents"
    fig.text(0.99, 0.005,
             f"Curves evaluated continuously on a 500-point grid (no binning)"
             f"{band_note} · n = agents with an evaluable coded strategy",
             ha="right", fontsize=7, color=MUTED)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close()
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
