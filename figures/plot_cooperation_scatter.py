#!/usr/bin/env python3
"""
Final-generation cooperation rate, aggregated across runs.

One figure: x = prompt variant (experiment), three model strips per group.
Each dot is one run's LAST-generation cooperation rate; a horizontal tick
marks the cell mean with a 95% CI (t-based) whisker across runs. Runs that
ended before generation 10 are drawn as open circles.

This replaces the per-experiment run-number scatter: run numbers are
arbitrary (run N of one model is independent of run N of another), so runs
are shown as distributions rather than aligned columns, and no runs are
dropped for being missing in another model.

Cooperation rate = mean donation % across all donor rounds in a generation
(same definition as plot_cooperation_by_generation.py).

Output: figures/run_variation/cooperation_final_gen_by_experiment.png

Usage:
    python figures/plot_cooperation_scatter.py \
        [--sim_root simulations] \
        [--experiments neutral_baseline neutral_claude ...] \
        [--models claude gpt gemini] \
        [--output_dir figures/run_variation]
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

_TS_RE = re.compile(r'_(\d{8}_\d{6})\.json$')

MODEL_LABELS = {"claude": "Claude 3.5 Sonnet", "gpt": "GPT-4o", "gemini": "Gemini 2.5 Flash"}
MODEL_COLORS = {"claude": "#FF6B35", "gpt": "#10A37F", "gemini": "#4285F4"}

EXPERIMENT_LABELS = {
    "neutral_baseline": "Neutral\n(baseline)",
    "neutral_claude":   "Neutral\n(Claude-phrased)",
    "neutral_gpt":      "Neutral\n(GPT-phrased)",
    "neutral_gemini":   "Neutral\n(Gemini-phrased)",
    "neutral_deepseek": "Neutral\n(DeepSeek-phrased)",
    "collaborative":    "Collaborative",
    "aggressive":       "Aggressive",
}

# Chart chrome (light mode, consistent with the gen-0 figure suite)
SURFACE  = "#fcfcfb"
INK      = "#0b0b0b"
INK_2    = "#52514e"
MUTED    = "#898781"
GRID     = "#e1e0d9"
BASELINE = "#c3c2b7"


def run_cooperation_by_gen(accum_path):
    """gen -> mean donation % across all donor rounds in that generation."""
    with open(accum_path) as f:
        data = json.load(f)
    donation_pcts = defaultdict(list)
    for r in data.get("agents_data", []):
        if not r.get("is_donor"):
            continue
        donated = r.get("donated", 0) or 0
        remaining = r.get("resources", 0) or 0
        denom = donated + remaining
        if denom > 0:
            donation_pcts[r["current_generation"]].append(100.0 * donated / denom)
    return {g: float(np.mean(v)) for g, v in donation_pcts.items() if v}


def final_gen_values(sim_dir):
    """[(final_gen_cooperation_pct, last_gen), ...] — one entry per run."""
    out = []
    for path in sorted(glob.glob(os.path.join(sim_dir, "Final_Accum*.json")),
                       key=lambda p: (_TS_RE.search(p).group(1) if _TS_RE.search(p) else p)):
        by_gen = run_cooperation_by_gen(path)
        if by_gen:
            last = max(by_gen)
            out.append((by_gen[last], last))
    return out


def plot_summary(data, experiments, models, out_path):
    """data[exp][model] = [(value, last_gen), ...]"""
    fig, ax = plt.subplots(figsize=(1.9 * len(experiments) + 3.5, 5.2),
                           facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    n_models = len(models)
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * 0.27

    for e_idx, exp in enumerate(experiments):
        for m_idx, model in enumerate(models):
            vals = data.get(exp, {}).get(model, [])
            if not vals:
                continue
            x0 = e_idx + offsets[m_idx]
            max_gen = max(lg for _, lg in vals)
            complete = np.array([v for v, lg in vals if lg == max_gen])
            early = np.array([v for v, lg in vals if lg != max_gen])
            color = MODEL_COLORS[model]

            # discrete points on the vertical line at each x position;
            # translucency makes coincident runs read as darker stacks
            ax.vlines(x0, 0, 100, color=GRID, linewidth=0.6, zorder=1)
            ax.scatter(np.full(complete.shape, x0), complete, s=26, color=color,
                       alpha=0.45, linewidths=0, zorder=2)
            if early.size:
                ax.scatter(np.full(early.shape, x0), early, s=26,
                           facecolors="none", edgecolors=color, alpha=0.7,
                           linewidths=1.0, zorder=2)

            all_vals = np.array([v for v, _ in vals])
            mean = all_vals.mean()
            if all_vals.size > 1:
                ci = sps.t.ppf(0.975, all_vals.size - 1) * all_vals.std(ddof=1) / np.sqrt(all_vals.size)
                ax.vlines(x0, mean - ci, mean + ci, color=INK, linewidth=1.2, zorder=4)
            ax.hlines(mean, x0 - 0.11, x0 + 0.11, color=INK, linewidth=2.2, zorder=5)
            if all_vals.size < 10:
                ax.annotate(f"n={all_vals.size}", (x0, 104.5), ha="center",
                            fontsize=6.5, color=INK_2, annotation_clip=False)

    ax.set_xticks(np.arange(len(experiments)))
    ax.set_xticklabels([EXPERIMENT_LABELS.get(e, e) for e in experiments],
                       fontsize=9, color=INK_2)
    ax.set_ylim(-3, 103)
    ax.set_xlim(-0.6, len(experiments) - 0.4)
    ax.set_ylabel("Final-generation cooperation rate\n(mean donation %)",
                  fontsize=9.5, color=INK_2)
    ax.set_title("Final-generation cooperation by prompt variant",
                 fontsize=12, color=INK, pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)

    handles = [plt.Line2D([], [], marker="o", linestyle="", color=MODEL_COLORS[m],
                          markersize=6, label=MODEL_LABELS[m]) for m in models]
    ax.legend(handles=handles, fontsize=8.5, frameon=False, loc="upper right",
              labelcolor=INK_2)
    fig.text(0.99, 0.01,
             "Dot: one run's final generation · tick: mean · whisker: 95% CI "
             "across runs · open circle: run ended before gen 10",
             ha="right", fontsize=7, color=MUTED)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Final-generation cooperation, aggregated across runs.")
    ap.add_argument("--sim_root", default="simulations")
    ap.add_argument("--experiments", nargs="+",
                    default=["neutral_baseline", "neutral_claude", "neutral_gemini",
                             "neutral_gpt", "neutral_deepseek"])
    ap.add_argument("--models", nargs="+", default=["claude", "gpt", "gemini"])
    ap.add_argument("--output_dir", default="figures/run_variation")
    args = ap.parse_args()

    data = {}
    print(f"{'experiment':<18} {'model':<8} {'runs':>4} {'mean':>6} {'min':>6} {'max':>6}")
    for exp in args.experiments:
        data[exp] = {}
        for model in args.models:
            vals = final_gen_values(os.path.join(args.sim_root, exp, model))
            if not vals:
                print(f"{exp:<18} {model:<8}   no data")
                continue
            data[exp][model] = vals
            v = np.array([x for x, _ in vals])
            print(f"{exp:<18} {model:<8} {len(v):>4} {v.mean():>6.1f} {v.min():>6.1f} {v.max():>6.1f}")

    out_path = os.path.join(args.output_dir, "cooperation_final_gen_by_experiment.png")
    plot_summary(data, args.experiments, args.models, out_path)


if __name__ == "__main__":
    main()
