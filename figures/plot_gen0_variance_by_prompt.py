#!/usr/bin/env python3
"""
Gen-0 cooperation variance across prompt variants — the variance_by_prompt
layout (small dots = runs, marker = mean ± std across runs, executor colors),
with each dot a run's GENERATION-0 self-cooperation rate (run-level mean AUC
of the coded initial strategies, as %) instead of the played run-mean
cooperation.

Reads the per-cell run means saved by gen0_auc_prompt_vs_model.py
(figures/gen0_prompt_vs_model/gen0_auc_stats.json — July strategies-only
grid, ~200 runs per cell).

Output: figures/run_variation/gen0_variance_by_prompt.png (+ .json)

Usage:
    python figures/plot_gen0_variance_by_prompt.py \
        [--experiments neutral_baseline_200 neutral_claude_200 ...] \
        [--stats figures/gen0_prompt_vs_model/gen0_auc_stats.json]
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

MODELS = ["claude", "gpt", "gemini"]
MODEL_COLORS = {"claude": "#FF6B35", "gpt": "#10A37F", "gemini": "#4285F4"}

# short x labels in the style of plot_variance_by_prompt.py
SHORT = {
    "neutral_baseline_200": "baseline", "neutral_claude_200": "claude",
    "neutral_gpt_200": "gpt", "neutral_gemini_200": "gemini",
    "neutral_deepseek_200": "deepseek", "collaborative_200": "collaborative",
    "aggressive_200": "aggressive",
}


def main():
    ap = argparse.ArgumentParser(
        description="Gen-0 cooperation variance across prompt variants.")
    ap.add_argument("--stats", default="figures/gen0_prompt_vs_model/gen0_auc_stats.json")
    ap.add_argument("--experiments", nargs="+",
                    default=["neutral_baseline_200", "neutral_claude_200",
                             "neutral_gpt_200", "neutral_gemini_200",
                             "neutral_deepseek_200"])
    ap.add_argument("--output_dir", default="figures/run_variation")
    args = ap.parse_args()

    cells = json.load(open(args.stats))["cells"]

    fig, ax = plt.subplots(figsize=(12, 6.2))
    offsets = {"claude": -0.24, "gpt": 0.0, "gemini": 0.24}
    rng = np.random.default_rng(0)
    summary = {}

    print(f"{'variant':<24} {'model':<8} {'runs':>5} {'mean%':>7} {'std%':>6}")
    for e_idx, exp in enumerate(args.experiments):
        for model in MODELS:
            s = cells.get(exp, {}).get(model)
            if not s or not s.get("run_means"):
                print(f"{exp:<24} {model:<8}   no data")
                continue
            vals = 100.0 * np.array(s["run_means"])   # AUC -> %
            x0 = e_idx + offsets[model]
            color = MODEL_COLORS[model]

            jit = rng.uniform(-0.07, 0.07, size=vals.size)
            ax.scatter(x0 + jit, vals, s=12, color=color, alpha=0.25,
                       linewidths=0, zorder=2)
            mean, std = vals.mean(), vals.std(ddof=1)
            ax.errorbar(x0, mean, yerr=std, fmt="o", color=color,
                        markersize=9, markeredgecolor="white",
                        markeredgewidth=1.0, elinewidth=2.0, capsize=4,
                        zorder=4)
            summary.setdefault(exp, {})[model] = {
                "n_runs": int(vals.size), "mean_pct": float(mean),
                "between_run_std_pct": float(std),
            }
            print(f"{exp:<24} {model:<8} {vals.size:>5} {mean:>7.1f} {std:>6.2f}")

    ax.set_xticks(range(len(args.experiments)))
    ax.set_xticklabels([SHORT.get(e, e) for e in args.experiments], fontsize=11)
    ax.set_xlabel("prompt variant (neutral_*)", fontsize=12)
    ax.set_ylabel("gen-0 self-cooperation rate (%)\n(run-level mean AUC of coded strategies)",
                  fontsize=11)
    ax.set_title("Gen-0 cooperation variance across prompt variants", fontsize=14, pad=16)
    ax.text(0.0, 1.02, "Gen-0 cooperation per run (small dots = runs, "
            "marker = mean ± std across runs)", transform=ax.transAxes, fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=MODEL_COLORS[m],
                          markersize=8, label=m) for m in MODELS]
    ax.legend(handles=handles, title="executor", fontsize=10, title_fontsize=10,
              frameon=False, loc="upper left")
    plt.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    out_png = os.path.join(args.output_dir, "gen0_variance_by_prompt.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(args.output_dir, "gen0_variance_by_prompt.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
