#!/usr/bin/env python3
"""
Within-run strategy homogeneity, measured as pairwise edit distance.

For one run at one generation, the 12 agents hold a set S of strategy texts.
This takes every unordered pair in S (66 pairs at 12 agents), computes the
word-level Levenshtein distance between the two texts, and histograms the
result. A tight histogram near zero means the population converged on one
strategy; a broad one means the agents are still saying different things.

Pairs are only ever formed WITHIN a run — two agents from different runs are
never compared, since cross-run distance measures nothing about convergence.
Pairs from all runs of a model are then pooled into one histogram.

Two metrics:
  normalised  Levenshtein / max(len_a, len_b) in words, range [0, 1]. The
              default, because raw distance grows with strategy length and
              gen-10 strategies are not the same length as gen-1 ones.
  words       raw word-level Levenshtein (number of word insertions,
              deletions, and substitutions).

Reads figures/reactivity_plots/{variant}/{model}/strategy_analysis_{model}.json
(the same per-variant analysis JSONs the reactivity figures use). All records
are used, including is_specified=False ones — an uncodeable strategy is still
a strategy for the purpose of measuring what the population is saying.

Outputs (to --output_dir, default figures/strategy_diversity/):
  edit_distance_hist_{metric}.png    one panel per model, gen 0 vs gen 10
  edit_distance_stats_{metric}.json  per-cell stats + gen 0 vs gen 10 test

Usage:
    python figures/analyze_strategy_edit_distance.py \
        [--analysis_dir figures/reactivity_plots/neutral_baseline] \
        [--models claude gpt gemini] \
        [--generations 1 10] \
        [--metric normalised|words] \
        [--output_dir figures/strategy_diversity]
"""

import argparse
import json
import os
import re
import sys
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_gen0_variants import (style_axes, MODEL_LABELS,
                                   SURFACE, INK, INK_2, MUTED)

MODELS = ["claude", "gpt", "gemini"]
MODEL_COLORS = {"claude": "#FF6B35", "gpt": "#10A37F", "gemini": "#4285F4"}
GEN0_COLOR = "#888888"

_WORD_RE = re.compile(r"[a-z0-9']+")

METRIC_LABELS = {
    "normalised": "Normalised word-level edit distance",
    "words": "Word-level edit distance (words)",
}


def gen_label(g):
    """Data generation number -> the label used in the paper (gen 1 = t=0)."""
    return "Generation 0" if g == 1 else f"Generation {g}"


def tokenize(text):
    return _WORD_RE.findall(text.lower())


def levenshtein(a, b):
    """Edit distance between two token sequences (two-row DP)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, tok_a in enumerate(a, 1):
        cur = [i]
        for j, tok_b in enumerate(b, 1):
            cur.append(min(prev[j] + 1,            # deletion
                           cur[j - 1] + 1,          # insertion
                           prev[j - 1] + (tok_a != tok_b)))
        prev = cur
    return prev[-1]


def load_strategies_by_run(analysis_dir, model, generation):
    """Return {run_id: [token_list, ...]} for one model at one generation."""
    path = os.path.join(analysis_dir, model, f"strategy_analysis_{model}.json")
    if not os.path.exists(path):
        print(f"  [{model}] not found: {path}")
        return {}
    with open(path) as f:
        records = json.load(f).get("agent_records", [])

    by_run = {}
    for rec in records:
        if rec.get("generation") != generation:
            continue
        toks = tokenize(rec.get("strategy") or "")
        if not toks:
            continue
        by_run.setdefault(rec["run"], []).append(toks)
    return by_run


def pairwise_distances(by_run, metric):
    """All within-run pairwise distances, plus the per-run mean of each run."""
    all_d, run_means, n_pairs_per_run = [], [], []
    for _run, strategies in sorted(by_run.items()):
        if len(strategies) < 2:
            continue
        ds = []
        for a, b in combinations(strategies, 2):
            d = levenshtein(a, b)
            if metric == "normalised":
                d /= max(len(a), len(b))
            ds.append(d)
        all_d.extend(ds)
        run_means.append(float(np.mean(ds)))
        n_pairs_per_run.append(len(ds))
    return np.array(all_d), np.array(run_means), n_pairs_per_run


def cell_summary(d, run_means, n_runs, n_pairs_per_run):
    return {
        "n_runs": int(n_runs),
        "n_pairs": int(d.size),
        "pairs_per_run": sorted(set(n_pairs_per_run)),
        "mean": float(d.mean()) if d.size else None,
        "median": float(np.median(d)) if d.size else None,
        "std": float(d.std()) if d.size else None,
        "p10": float(np.percentile(d, 10)) if d.size else None,
        "p90": float(np.percentile(d, 90)) if d.size else None,
        "frac_identical": float((d == 0).mean()) if d.size else None,
        "run_mean_of_means": float(run_means.mean()) if run_means.size else None,
        "run_means": [float(m) for m in run_means],
    }


def fmt_p(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def main():
    parser = argparse.ArgumentParser(
        description="Within-run pairwise strategy edit distance, gen 0 vs gen 10.")
    parser.add_argument("--analysis_dir",
                        default="figures/reactivity_plots/neutral_baseline",
                        help="Directory with {model}/strategy_analysis_{model}.json")
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--generations", nargs="+", type=int, default=[1, 10],
                        help="Two generations to contrast (data numbering: 1 = t=0)")
    parser.add_argument("--metric", default="normalised",
                        choices=["normalised", "words"])
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--output_dir", default="figures/strategy_diversity")
    args = parser.parse_args()

    data, summary, tests = {}, {}, {}

    print(f"\nmetric: {args.metric}   source: {args.analysis_dir}")
    print(f"\n{'model':<8} {'gen':>4} {'runs':>5} {'pairs':>6} {'mean':>7} "
          f"{'median':>7} {'%identical':>11}")
    for model in args.models:
        data[model], summary[model] = {}, {}
        for gen in args.generations:
            by_run = load_strategies_by_run(args.analysis_dir, model, gen)
            d, run_means, n_pp = pairwise_distances(by_run, args.metric)
            if not d.size:
                print(f"{model:<8} {gen:>4}   no strategies — skipped")
                continue
            data[model][gen] = (d, run_means)
            s = cell_summary(d, run_means, len(by_run), n_pp)
            summary[model][gen] = s
            print(f"{model:<8} {gen:>4} {s['n_runs']:>5} {s['n_pairs']:>6} "
                  f"{s['mean']:>7.3f} {s['median']:>7.3f} "
                  f"{100 * s['frac_identical']:>10.1f}%")

        # Run-level test: pairs within a run are not independent, so the
        # significance test uses one number (the mean distance) per run.
        g_a, g_b = args.generations[0], args.generations[-1]
        if g_a in data[model] and g_b in data[model]:
            rm_a, rm_b = data[model][g_a][1], data[model][g_b][1]
            t = {}
            if rm_a.size > 1 and rm_b.size > 1:
                stat, p = sps.mannwhitneyu(rm_a, rm_b, alternative="two-sided")
                t["run_mean_mannwhitney"] = {"U": float(stat), "p": float(p)}
                stat, p = sps.ttest_ind(rm_a, rm_b, equal_var=False)
                t["run_mean_welch_t"] = {"t": float(stat), "p": float(p)}
            stat, p = sps.mannwhitneyu(data[model][g_a][0], data[model][g_b][0],
                                       alternative="two-sided")
            t["pair_level_mannwhitney"] = {"U": float(stat), "p": float(p)}
            tests[model] = t
            if "run_mean_mannwhitney" in t:
                delta = rm_b.mean() - rm_a.mean()
                print(f"         -> gen {g_a} vs {g_b}: mean run distance "
                      f"{rm_a.mean():.3f} -> {rm_b.mean():.3f} "
                      f"({delta:+.3f}), runs MW "
                      f"{fmt_p(t['run_mean_mannwhitney']['p'])}")

    plotted_models = [m for m in args.models if data.get(m)]
    if not plotted_models:
        print("\nNo data to plot.")
        return

    # Shared bin edges so the panels are directly comparable.
    hi = max(d.max() for m in plotted_models for d, _ in data[m].values())
    if args.metric == "normalised":
        hi = 1.0
    edges = np.linspace(0, hi, args.bins + 1)

    fig, axes = plt.subplots(1, len(plotted_models),
                             figsize=(5.0 * len(plotted_models), 4.6),
                             sharey=True, facecolor=SURFACE, squeeze=False)
    for ax, model in zip(axes[0], plotted_models):
        style_axes(ax, grid_axis="y")
        for i, gen in enumerate(args.generations):
            if gen not in data[model]:
                continue
            d, _ = data[model][gen]
            color = GEN0_COLOR if i == 0 else MODEL_COLORS[model]
            ax.hist(d, bins=edges, density=True, color=color,
                    alpha=0.45 if i == 0 else 0.65, zorder=2,
                    label=f"{gen_label(gen)} (n={d.size} pairs)")
            ax.axvline(np.median(d), color=color, linestyle="--",
                       linewidth=1.4, zorder=3)
        ax.set_title(MODEL_LABELS[model], fontsize=11.5, color=INK)
        ax.set_xlabel(METRIC_LABELS[args.metric], fontsize=9.5, color=INK_2)
        ax.set_xlim(0, hi)
        ax.legend(fontsize=8, loc="upper right", frameon=False, labelcolor=INK_2)

    axes[0][0].set_ylabel("Density of within-run strategy pairs",
                          fontsize=9.5, color=INK_2)
    fig.suptitle("Within-run strategy diversity, before and after evolution",
                 fontsize=13, color=INK)
    fig.text(0.99, 0.005,
             "Every unordered pair of the 12 agents' strategies within a run, "
             "pooled across runs · dashed line: median",
             ha="right", fontsize=7, color=MUTED)
    plt.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    out_png = os.path.join(args.output_dir,
                           f"edit_distance_hist_{args.metric}.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close()
    print(f"\nSaved: {out_png}")

    out_json = os.path.join(args.output_dir,
                            f"edit_distance_stats_{args.metric}.json")
    with open(out_json, "w") as f:
        json.dump({"metric": args.metric,
                   "analysis_dir": args.analysis_dir,
                   "note": "generation 1 in the data = initial state (t=0); "
                           "pairs are formed within a run only",
                   "generations": args.generations,
                   "summary": summary, "tests": tests}, f, indent=2)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
