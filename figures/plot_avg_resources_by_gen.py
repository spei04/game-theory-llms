#!/usr/bin/env python3
"""
Cultural evolution of cooperation: average final resources per generation.

One figure per prompt variant; all three executor models on the same axes.
Line = mean across runs of the generation's average final resources
(generation_stats[].average_resources in each Final_Accum file, i.e. the
mean agent endowment 240 - donated + received at generation end); shaded
band = 95% confidence interval of the mean across runs (t-based).

Output: figures/avg_resources/<experiment>_avg_resources.png

Usage:
    python figures/plot_avg_resources_by_gen.py \
        [--sim_root simulations] \
        [--experiments neutral_baseline neutral_claude ...] \
        [--models claude gpt gemini] \
        [--output_dir figures/avg_resources] [--runs] [--grid NAME]

--runs draws each run as a thin line under the mean (output *_runs.png).
--grid NAME also writes every experiment into one 2-column figure.
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

MODEL_LABELS = {"claude": "Claude", "gpt": "GPT", "gemini": "Gemini"}
MODEL_COLORS = {"claude": "#FF6B35", "gpt": "#10A37F", "gemini": "#4285F4"}

EXPERIMENT_TITLES = {
    "neutral_baseline": "Neutral (baseline)",
    "neutral_claude":   "Neutral (Claude-phrased)",
    "neutral_gpt":      "Neutral (GPT-phrased)",
    "neutral_gemini":   "Neutral (Gemini-phrased)",
    "neutral_deepseek": "Neutral (DeepSeek-phrased)",
    "collaborative":    "Collaborative",
    "aggressive":       "Aggressive",
}

SURFACE, INK, INK_2, MUTED, GRID, BASE = ("#fcfcfb", "#0b0b0b", "#52514e",
                                          "#898781", "#e1e0d9", "#c3c2b7")


def run_avg_resources(path):
    """gen -> average_resources for one run."""
    stats = json.load(open(path)).get("generation_stats", [])
    return {s["generation"]: float(s["average_resources"]) for s in stats
            if s.get("average_resources") is not None}


def draw_experiment(ax, sim_root, exp, models, runs=False, fs=1.0):
    """Draw one prompt variant on ax. Returns the generations plotted, or None.

    runs=False: mean line + 95% CI band. runs=True: one thin line per run
    with the mean drawn on top, so the spread between runs is visible."""
    gens = None
    for model in models:
        series = []
        for p in sorted(glob.glob(os.path.join(sim_root, exp, model,
                                               "Final_Accum*.json"))):
            r = run_avg_resources(p)
            if r:
                series.append(r)
        if not series:
            continue
        gens = sorted({g for r in series for g in r})
        mat = np.array([[r.get(g, np.nan) for g in gens] for r in series])
        mean = np.nanmean(mat, axis=0)
        color = MODEL_COLORS.get(model, "#555555")
        label = MODEL_LABELS.get(model, model)
        if runs:
            for row in mat:
                ax.plot(gens, row, color=color, lw=0.8, alpha=0.35, zorder=2)
            ax.plot(gens, mean, color=color, lw=2.6, zorder=3,
                    label=f"{label} (mean of {len(series)} runs)")
        else:
            n = np.sum(~np.isnan(mat), axis=0)
            sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(n)
            ci = sps.t.ppf(0.975, np.maximum(n - 1, 1)) * sem
            ax.plot(gens, mean, color=color, lw=2.2, marker="o", ms=4,
                    label=label, zorder=3)
            ax.fill_between(gens, mean - ci, mean + ci, color=color,
                            alpha=0.18, linewidth=0, zorder=2)
    if gens is None:
        return None
    ax.set_facecolor(SURFACE)
    ax.set_xlabel("Generation", fontsize=11 * fs, color=INK)
    ax.set_ylabel("Average final resources", fontsize=11 * fs, color=INK)
    ax.set_title(EXPERIMENT_TITLES.get(exp, exp), fontsize=12.5 * fs, color=INK, pad=10)
    ax.set_xticks(gens)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9.5 * fs)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.legend(fontsize=9.5 * fs, frameon=False, labelcolor=INK_2)
    return gens


def bottom_legend(fig, models, fs, reserve):
    """One legend of model names below the axes, so it never covers the lines."""
    handles = [plt.Line2D([], [], color=MODEL_COLORS.get(m, "#555555"), lw=2.6)
               for m in models]
    fig.legend(handles, [MODEL_LABELS.get(m, m) for m in models], loc="lower center",
               ncol=len(models), fontsize=9.5 * fs, frameon=False, labelcolor=INK_2,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, reserve, 1, 1))


def save(fig, out_path):
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_experiment(sim_root, exp, models, out_path, runs=False):
    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor=SURFACE)
    if draw_experiment(ax, sim_root, exp, models, runs=runs, fs=1.4 if runs else 1.0) is None:
        plt.close(fig)
        return False
    if runs:
        ax.get_legend().remove()
        bottom_legend(fig, models, fs=1.4, reserve=0.1)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
        plt.close(fig)
        return True
    fig.text(0.99, 0.01, "line: mean across runs · band: 95% CI of the mean (t-based)",
             ha="right", fontsize=7, color=MUTED)
    save(fig, out_path)
    return True


def plot_grid(sim_root, exps, models, out_path, runs=False):
    """All variants in one 2-column figure (the paper's appendix trajectory panel)."""
    nrows = (len(exps) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(13, 4.6 * nrows),
                             facecolor=SURFACE, sharey=True, squeeze=False)
    for ax, exp in zip(axes.flat, exps):
        draw_experiment(ax, sim_root, exp, models, runs=runs, fs=1.9)
        ax.get_legend().remove()
    for ax in list(axes.flat)[len(exps):]:
        ax.set_visible(False)
    bottom_legend(fig, models, fs=1.9, reserve=0.05)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Average final resources by generation.")
    ap.add_argument("--sim_root", default="simulations")
    ap.add_argument("--experiments", nargs="+",
                    default=["neutral_baseline", "neutral_claude", "neutral_gemini",
                             "neutral_gpt", "neutral_deepseek"])
    ap.add_argument("--models", nargs="+", default=["claude", "gpt", "gemini"])
    ap.add_argument("--output_dir", default="figures/avg_resources")
    ap.add_argument("--runs", action="store_true",
                    help="Draw every run as a thin line under the mean instead of a CI band.")
    ap.add_argument("--grid", metavar="NAME",
                    help="Also save all experiments in one 2-column figure as NAME.png.")
    args = ap.parse_args()
    suffix = "_runs" if args.runs else ""

    for exp in args.experiments:
        out = os.path.join(args.output_dir, f"{exp}_avg_resources{suffix}.png")
        if plot_experiment(args.sim_root, exp, args.models, out, runs=args.runs):
            print(f"Saved: {out}")
        else:
            print(f"  [{exp}] no data — skipped")
    if args.grid:
        out = os.path.join(args.output_dir, f"{args.grid}{suffix}.png")
        plot_grid(args.sim_root, args.experiments, args.models, out, runs=args.runs)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
