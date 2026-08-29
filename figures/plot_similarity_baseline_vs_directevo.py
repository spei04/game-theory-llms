#!/usr/bin/env python3
"""
Within-run strategy homogeneity: LLM evolution vs direct evolution.

The paper-ready two-condition version of analyze_strategy_similarity.py. Each
panel overlays three distributions of within-run pairwise strategy similarity:

  generation 0            gray fill    — the initial strategies (identical by
                                         construction across conditions, since
                                         direct evolution only alters gens 2+)
  generation 10, baseline model color  — LLM evolution: agents rewrite their
                                         strategy each generation
  generation 10, direct  black outline — direct evolution: agents adopt a
                                         survivor's strategy verbatim, sampled
                                         from a softmax over final scores

Rows are the two similarity metrics (lexical, semantic); columns are models.
See analyze_strategy_similarity.py for the metric definitions and formulas —
they are re-emitted into this script's stats JSON so the figure is
self-documenting.

Beyond the per-condition stats, this runs two cross-condition tests per
model x metric, both on run-level means:
  gen1_equivalence  gen 1 baseline vs gen 1 direct. Should be NON-significant;
                    a significant result means the two arms did not start from
                    the same place and the gen-10 contrast is confounded.
  gen10_contrast    gen 10 baseline vs gen 10 direct. The actual comparison.

Outputs (to --output_dir, default figures/strategy_diversity/):
  {a}_vs_{b}_similarity_hist.png
  {a}_vs_{b}_similarity_stats.json

Usage:
    python figures/plot_similarity_baseline_vs_directevo.py \
        [--baseline_dir figures/reactivity_plots/neutral_baseline] \
        [--direct_dir simulations/direct_evo_50_old] \
        [--baseline_label neutral_baseline] [--direct_label direct_evo_b50] \
        [--metrics lexical semantic] [--generations 1 10] \
        [--output_dir figures/strategy_diversity]
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_gen0_variants import (style_axes, MODEL_LABELS,
                                   SURFACE, INK, INK_2, MUTED)
from analyze_strategy_similarity import (
    MODELS, MODEL_COLORS, GEN0_COLOR, FORMULAS, METRIC_AXIS,
    EmbeddingCache, detect_source, load_raw, load_analysis,
    pairwise, cell_summary, gen_label, fmt_p, sig_stars)

DIRECT_COLOR = "#111111"


def load_condition(data_dir, model, generation, source):
    loader = load_raw if source == "raw" else load_analysis
    return loader(data_dir, model, generation)


def run_test(a, b):
    """Run-level Mann-Whitney + Welch t + Cliff's delta on two run-mean sets."""
    if a.size < 2 or b.size < 2:
        return None
    u, p = sps.mannwhitneyu(a, b, alternative="two-sided")
    t, pt = sps.ttest_ind(a, b, equal_var=False)
    return {
        "n_a": int(a.size), "n_b": int(b.size),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "delta": float(b.mean() - a.mean()),
        "mannwhitney": {"U": float(u), "p": float(p)},
        "welch_t": {"t": float(t), "p": float(pt)},
        "cliffs_delta": float(2 * u / (a.size * b.size) - 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Within-run strategy similarity: LLM vs direct evolution.")
    parser.add_argument("--baseline_dir",
                        default="figures/reactivity_plots/neutral_baseline")
    parser.add_argument("--direct_dir", default="simulations/direct_evo_50_old")
    parser.add_argument("--baseline_label", default="neutral_baseline")
    parser.add_argument("--direct_label", default="direct_evo_b50")
    parser.add_argument("--baseline_legend", default="LLM evolution")
    parser.add_argument("--direct_legend", default="Direct evolution (β=50)")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--metrics", nargs="+", default=["lexical", "semantic"],
                        choices=["lexical", "semantic", "words"])
    parser.add_argument("--generations", nargs="+", type=int, default=[1, 10])
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--output_dir", default="figures/strategy_diversity")
    parser.add_argument("--cache",
                        default="figures/strategy_diversity/_embedding_cache.npz")
    args = parser.parse_args()

    g0, g10 = args.generations[0], args.generations[-1]
    conds = [("baseline", args.baseline_dir, args.baseline_label),
             ("direct", args.direct_dir, args.direct_label)]
    sources = {k: detect_source(d, args.models) for k, d, _ in conds}
    cache = EmbeddingCache(args.cache) if "semantic" in args.metrics else None

    print(f"\n{'=' * 78}")
    print("Within-run strategy similarity — LLM evolution vs direct evolution")
    print(f"{'=' * 78}")
    for k, d, lab in conds:
        print(f"{k:<9} {lab:<20} {d}  (source: {sources[k]})")
    for m in args.metrics:
        print(f"\n{m:9}: {FORMULAS[m]['formula']}")

    # data[metric][model][cond][gen] = (scores, run_means)
    data, summary = {}, {}
    for metric in args.metrics:
        data[metric], summary[metric] = {}, {}
        for model in args.models:
            data[metric][model], summary[metric][model] = {}, {}
            for k, d, _ in conds:
                data[metric][model][k], summary[metric][model][k] = {}, {}
                for gen in args.generations:
                    by_run = load_condition(d, model, gen, sources[k])
                    if not by_run:
                        continue
                    s, rm, npair, ndist = pairwise(by_run, metric, cache)
                    if not s.size:
                        continue
                    data[metric][model][k][gen] = (s, rm)
                    summary[metric][model][k][gen] = cell_summary(
                        s, rm, len(by_run), npair, ndist, metric)

    # ── Per-condition table ──────────────────────────────────────────────────
    for metric in args.metrics:
        print(f"\n{'-' * 78}\n{metric}\n{'-' * 78}")
        print(f"{'model':<8} {'condition':<10} {'runs':>5} {'pairs':>6} "
              f"{'gen0':>7} {'gen10':>7} {'distinct/run g10':>17} "
              f"{'%ident g10':>11}")
        for model in args.models:
            for k, _, _ in conds:
                cell = summary[metric][model][k]
                if g0 not in cell or g10 not in cell:
                    continue
                a, b = cell[g0], cell[g10]
                ident = ("    n/a" if b["frac_identical"] is None
                         else f"{100 * b['frac_identical']:9.1f}%")
                print(f"{model:<8} {k:<10} {b['n_runs']:>5} {b['n_pairs']:>6} "
                      f"{a['run_mean_of_means']:>7.3f} "
                      f"{b['run_mean_of_means']:>7.3f} "
                      f"{b['mean_distinct_strategies_per_run']:>17.1f} "
                      f"{ident:>11}")

    # ── Cross-condition tests ────────────────────────────────────────────────
    tests = {}
    print(f"\n{'=' * 78}")
    print("CROSS-CONDITION TESTS — run-level Mann-Whitney (n = runs)")
    print(f"{'=' * 78}")
    print(f"{'metric':<9} {'model':<8} {'test':<17} {'baseline':>9} "
          f"{'direct':>8} {'delta':>8} {'cliff d':>8} {'p':>12}  sig")
    for metric in args.metrics:
        tests[metric] = {}
        for model in args.models:
            tests[metric][model] = {}
            for name, gen in (("gen1_equivalence", g0), ("gen10_contrast", g10)):
                bl = data[metric][model]["baseline"].get(gen)
                dr = data[metric][model]["direct"].get(gen)
                if bl is None or dr is None:
                    continue
                t = run_test(bl[1], dr[1])
                if t is None:
                    continue
                tests[metric][model][name] = t
                p = t["mannwhitney"]["p"]
                note = sig_stars(p)
                if name == "gen1_equivalence":
                    note += "  <- should be ns" if p < 0.05 else "  (ok)"
                print(f"{metric:<9} {model:<8} {name:<17} {t['mean_a']:>9.3f} "
                      f"{t['mean_b']:>8.3f} {t['delta']:>+8.3f} "
                      f"{t['cliffs_delta']:>+8.2f} {fmt_p(p):>12}  {note}")

    # ── Figure ───────────────────────────────────────────────────────────────
    plotted = [m for m in args.models
               if any(data[k][m]["baseline"] for k in args.metrics)]
    nrow, ncol = len(args.metrics), len(plotted)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.1 * ncol, 4.4 * nrow),
                             sharey="row", facecolor=SURFACE, squeeze=False)
    for row, metric in enumerate(args.metrics):
        vals = [s for m in plotted for c in ("baseline", "direct")
                for s, _ in data[metric][m][c].values()]
        hi = 1.0 if metric in ("lexical", "semantic") else max(
            (s.max() for s in vals), default=1.0)
        lo = (np.floor(min(s.min() for s in vals) * 20) / 20
              if metric == "semantic" and vals else 0.0)
        edges = np.linspace(lo, hi, args.bins + 1)

        for col, model in enumerate(plotted):
            ax = axes[row][col]
            style_axes(ax, grid_axis="y")
            # gen 0 — same by construction in both arms; drawn from baseline.
            cell = data[metric][model]["baseline"].get(g0)
            if cell is not None:
                ax.hist(cell[0], bins=edges, density=True, color=GEN0_COLOR,
                        alpha=0.42, zorder=2)
            # gen 10 under LLM evolution.
            cell = data[metric][model]["baseline"].get(g10)
            if cell is not None:
                ax.hist(cell[0], bins=edges, density=True,
                        color=MODEL_COLORS[model], alpha=0.62, zorder=3)
            # gen 10 under direct evolution — outline so it reads over the fill.
            cell = data[metric][model]["direct"].get(g10)
            if cell is not None:
                ax.hist(cell[0], bins=edges, density=True, histtype="step",
                        color=DIRECT_COLOR, linewidth=1.7, zorder=4)

            if row == 0:
                ax.set_title(MODEL_LABELS[model], fontsize=11.5, color=INK)
            ax.set_xlabel(METRIC_AXIS[metric], fontsize=9.5, color=INK_2)
            ax.set_xlim(lo, hi)
        axes[row][0].set_ylabel("Density of within-run pairs",
                               fontsize=9.5, color=INK_2)

    handles = [
        Patch(facecolor=GEN0_COLOR, alpha=0.42,
              label=f"{gen_label(g0)} — initial strategies"),
        Patch(facecolor="#8a8a8a", alpha=0.62,
              label=f"{gen_label(g10)} — {args.baseline_legend}"),
        Line2D([0], [0], color=DIRECT_COLOR, linewidth=1.7,
               label=f"{gen_label(g10)} — {args.direct_legend}"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9.5, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Within-run strategy homogeneity: LLM evolution rewrites, "
                 "direct evolution collapses", fontsize=13, color=INK)
    fig.text(0.99, -0.012,
             "Every unordered pair of the 12 agents' strategies within a run, "
             "pooled across runs · gen-10 fill coloured per model",
             ha="right", fontsize=7, color=MUTED)
    plt.tight_layout(rect=(0, 0.03, 1, 1))

    os.makedirs(args.output_dir, exist_ok=True)
    tag = f"{args.baseline_label}_vs_{args.direct_label}"
    out_png = os.path.join(args.output_dir, f"{tag}_similarity_hist.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close()
    print(f"\nSaved: {out_png}")

    out_json = os.path.join(args.output_dir, f"{tag}_similarity_stats.json")
    with open(out_json, "w") as f:
        json.dump({
            "conditions": {
                "baseline": {"label": args.baseline_label,
                             "dir": args.baseline_dir,
                             "source": sources["baseline"],
                             "legend": args.baseline_legend},
                "direct": {"label": args.direct_label,
                           "dir": args.direct_dir,
                           "source": sources["direct"],
                           "legend": args.direct_legend},
            },
            "generations": args.generations,
            "note": "generation 1 in the data = initial state (t=0). Pairs are "
                    "formed within a run only. All tests use one value per run "
                    "(the run's mean) because the 66 pairs within a run share "
                    "agents and are not independent.",
            "metrics": {m: FORMULAS[m] for m in args.metrics},
            "summary": summary, "cross_condition_tests": tests,
        }, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
