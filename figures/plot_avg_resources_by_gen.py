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
        [--output_dir figures/avg_resources]
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


def plot_experiment(sim_root, exp, models, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    plotted = False

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
        n = np.sum(~np.isnan(mat), axis=0)
        sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(n)
        ci = sps.t.ppf(0.975, np.maximum(n - 1, 1)) * sem
        color = MODEL_COLORS.get(model, "#555555")
        ax.plot(gens, mean, color=color, lw=2.2, marker="o", ms=4,
                label=f"{MODEL_LABELS.get(model, model)}", zorder=3)
        ax.fill_between(gens, mean - ci, mean + ci, color=color,
                        alpha=0.18, linewidth=0, zorder=2)
        plotted = True

    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel("Generation", fontsize=11, color=INK)
    ax.set_ylabel("Average final resources", fontsize=11, color=INK)
    ax.set_title(EXPERIMENT_TITLES.get(exp, exp), fontsize=12.5, color=INK, pad=10)
    ax.set_xticks(gens)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.legend(fontsize=9.5, frameon=False, labelcolor=INK_2, title="executor",
              title_fontsize=9.5)
    fig.text(0.99, 0.01, "line: mean across runs · band: 95% CI of the mean (t-based)",
             ha="right", fontsize=7, color=MUTED)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description="Average final resources by generation.")
    ap.add_argument("--sim_root", default="simulations")
    ap.add_argument("--experiments", nargs="+",
                    default=["neutral_baseline", "neutral_claude", "neutral_gemini",
                             "neutral_gpt", "neutral_deepseek"])
    ap.add_argument("--models", nargs="+", default=["claude", "gpt", "gemini"])
    ap.add_argument("--output_dir", default="figures/avg_resources")
    args = ap.parse_args()

    for exp in args.experiments:
        out = os.path.join(args.output_dir, f"{exp}_avg_resources.png")
        if plot_experiment(args.sim_root, exp, args.models, out):
            print(f"Saved: {out}")
        else:
            print(f"  [{exp}] no data — skipped")


if __name__ == "__main__":
    main()
