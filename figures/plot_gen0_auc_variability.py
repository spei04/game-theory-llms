#!/usr/bin/env python3
"""
Run-to-run variability of initial (gen-0) cooperation, by prompt variant.

Reads the per-cell stats saved by gen0_auc_prompt_vs_model.py and plots the
SD of run-level mean AUC across runs for each (variant, model) — i.e. how
much the initial cooperation rate varies from run to run under each prompt.
Bootstrap 95% CI on the SD (resampling runs) as whiskers.

Output: figures/gen0_prompt_vs_model/gen0_auc_run_variability.png

Usage:
    python figures/plot_gen0_auc_variability.py \
        [--stats figures/gen0_prompt_vs_model/gen0_auc_stats.json]
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

MODELS = ["claude", "gpt", "gemini"]
MODEL_LABELS = {"claude": "Claude 3.5 Sonnet", "gpt": "GPT-4o", "gemini": "Gemini 2.5 Flash"}
MODEL_COLORS = {"claude": "#FF6B35", "gpt": "#10A37F", "gemini": "#4285F4"}

LABELS = {
    "neutral_baseline": "Neutral\n(baseline)", "neutral_claude": "Neutral\n(Claude-phrased)",
    "neutral_gpt": "Neutral\n(GPT-phrased)", "neutral_gemini": "Neutral\n(Gemini-phrased)",
    "neutral_deepseek": "Neutral\n(DeepSeek-phrased)", "collaborative": "Collaborative",
    "aggressive": "Aggressive",
}

SURFACE, INK, INK_2, MUTED, GRID, BASE = ("#fcfcfb", "#0b0b0b", "#52514e",
                                          "#898781", "#e1e0d9", "#c3c2b7")


def boot_sd_ci(run_means, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(run_means)
    sds = np.array([rng.choice(v, size=v.size, replace=True).std(ddof=1)
                    for _ in range(n_boot)])
    return np.percentile(sds, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description="Run-to-run variability of gen-0 AUC.")
    ap.add_argument("--stats", default="figures/gen0_prompt_vs_model/gen0_auc_stats.json")
    ap.add_argument("--output", default="figures/gen0_prompt_vs_model/gen0_auc_run_variability.png")
    args = ap.parse_args()

    cells = json.load(open(args.stats))["cells"]
    variants = list(cells.keys())

    fig, ax = plt.subplots(figsize=(12, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    bar_w = 0.78 / len(MODELS)
    offsets = (np.arange(len(MODELS)) - 1) * bar_w
    print(f"{'variant':<24} {'model':<8} {'SD':>7} {'95% CI':>18} {'runs':>5}")
    for m_idx, model in enumerate(MODELS):
        xs, sds, err_lo, err_hi = [], [], [], []
        for v_idx, variant in enumerate(variants):
            s = cells.get(variant, {}).get(model)
            if not s or s.get("run_std") is None:
                continue
            rm = s["run_means"]
            lo, hi = boot_sd_ci(rm)
            xs.append(v_idx + offsets[m_idx])
            sds.append(s["run_std"])
            err_lo.append(s["run_std"] - lo)
            err_hi.append(hi - s["run_std"])
            print(f"{variant:<24} {model:<8} {s['run_std']:>7.4f} "
                  f"[{lo:>7.4f},{hi:>7.4f}] {len(rm):>5}")
        ax.bar(xs, sds, width=bar_w * 0.92, color=MODEL_COLORS[model],
               alpha=0.9, edgecolor=SURFACE, linewidth=0.8, zorder=2,
               label=MODEL_LABELS[model])
        ax.errorbar(xs, sds, yerr=[err_lo, err_hi], fmt="none", ecolor=INK_2,
                    elinewidth=1, capsize=2.5, zorder=4)

    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels([LABELS.get(v.replace("_200", ""), v) for v in variants],
                       fontsize=8.5, color=INK_2)
    ax.set_ylabel("SD of run-level gen-0 cooperation (AUC)\nacross runs",
                  fontsize=9.5, color=INK_2)
    ax.set_title("Run-to-run variability of initial cooperation by prompt variant",
                 fontsize=12, color=INK, pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left", labelcolor=INK_2)
    fig.text(0.99, 0.01,
             "Bars: SD of per-run mean AUC across runs · whiskers: bootstrap 95% CI on the SD",
             ha="right", fontsize=7, color=MUTED)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
