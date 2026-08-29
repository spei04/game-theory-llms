#!/usr/bin/env python3
"""
One-figure summary of where gen-0 cooperation variability comes from.

Left panel : stacked bar of eta^2 shares from the run-level two-way ANOVA
             (prompt, prompt x model interaction, model, run-to-run residual),
             read from figures/gen0_prompt_vs_model/gen0_auc_stats.json.
Right panel: the raw picture behind those numbers — every run's mean gen-0
             AUC as a dot, grouped by prompt variant and colored by model.
             Spread between variant groups = prompt variance; spread between
             colors within a group = model (+ interaction) variance; spread
             of dots within one color = run-to-run noise.

Only the variants actually included in the ANOVA are shown, so the two
panels describe the same data.

Usage:
    python figures/plot_variability_sources.py \
        [--stats figures/gen0_prompt_vs_model/gen0_auc_stats.json] \
        [--output figures/variability_sources.png]
"""

import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

MODELS = ["claude", "gpt", "gemini"]
MODEL_LABELS = {"claude": "Claude 3.5 Sonnet", "gpt": "GPT-4o", "gemini": "Gemini 2.5 Flash"}
MODEL_COLORS = {"claude": "#FF6B35", "gpt": "#10A37F", "gemini": "#4285F4"}

VARIANT_LABELS = {
    "neutral_baseline": "Neutral\n(baseline)",
    "neutral_claude":   "Neutral\n(Claude)",
    "neutral_gpt":      "Neutral\n(GPT)",
    "neutral_gemini":   "Neutral\n(Gemini)",
    "neutral_deepseek": "Neutral\n(DeepSeek)",
    "collaborative":    "Collab.",
    "aggressive":       "Aggr.",
}

SOURCES = [   # (anova key, label, color)
    ("prompt",         "Prompt",           "#2a78d6"),
    ("prompt_x_model", "Prompt × model",   "#4a3aa7"),
    ("model",          "Model",            "#1baf7a"),
    ("residual",       "Run-to-run noise", "#b0aea6"),
]


def variant_label(v):
    base = v[:-4] if v.endswith("_200") else v
    return VARIANT_LABELS.get(base, base)


def main():
    parser = argparse.ArgumentParser(description="Plot sources of gen-0 cooperation variability.")
    parser.add_argument("--stats", default="figures/gen0_prompt_vs_model/gen0_auc_stats.json")
    parser.add_argument("--output", default="figures/variability_sources.png")
    args = parser.parse_args()

    with open(args.stats) as f:
        d = json.load(f)
    anova = d["anova_run_level"]
    cells = d["cells"]
    variants = anova.get("variants_included", list(cells))

    fig, (ax_bar, ax_dots) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 2.4]})

    # ── Left: eta^2 stacked bar ──────────────────────────────────────────────
    left = 0.0
    for key, label, color in SOURCES:
        share = anova[key]["eta_sq"]
        ax_bar.barh(0, share, left=left, height=0.45, color=color,
                    edgecolor="white", linewidth=1,
                    label=f"{label} ({share * 100:.0f}%)")
        if share > 0.05:
            ax_bar.text(left + share / 2, 0.32, f"{share * 100:.0f}%",
                        ha="center", fontsize=10, color=color, fontweight="bold")
        left += share
    ax_bar.legend(fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 0.02),
                  frameon=False)

    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(-0.8, 0.8)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Share of run-level variance (η²)", fontsize=10)
    ax_bar.set_title(f"Variance decomposition\n(two-way ANOVA, n={anova['n_obs_runs']} runs)",
                     fontsize=11)
    for side in ("top", "right", "left"):
        ax_bar.spines[side].set_visible(False)

    # ── Right: run-level AUC dots per variant x model ────────────────────────
    rng = np.random.default_rng(0)
    group_w = 0.75
    offsets = {m: (i - 1) * group_w / 3 for i, m in enumerate(MODELS)}

    for gx, variant in enumerate(variants):
        for m in MODELS:
            cell = cells.get(variant, {}).get(m)
            if not cell:
                continue
            runs = np.array(cell["run_means"])
            x = gx + offsets[m] + rng.uniform(-0.08, 0.08, size=runs.size)
            ax_dots.scatter(x, runs, s=6, color=MODEL_COLORS[m], alpha=0.18,
                            linewidths=0, zorder=2)
            ax_dots.errorbar(gx + offsets[m], runs.mean(), yerr=runs.std(ddof=1),
                             fmt="o", color=MODEL_COLORS[m], markersize=6,
                             markeredgecolor="white", capsize=3, linewidth=1.6,
                             zorder=3)

    ax_dots.set_xticks(range(len(variants)))
    ax_dots.set_xticklabels([variant_label(v) for v in variants], fontsize=9)
    ax_dots.set_ylabel("Run-mean gen-0 AUC (cooperation, 0–1)", fontsize=10)
    ax_dots.set_title("Each dot = one run  (mean ± 1 SD across runs)", fontsize=11)
    ax_dots.grid(axis="y", alpha=0.3)
    for side in ("top", "right"):
        ax_dots.spines[side].set_visible(False)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=MODEL_COLORS[m],
                          label=MODEL_LABELS[m]) for m in MODELS]
    ax_dots.legend(handles=handles, fontsize=8.5, loc="upper left")

    fig.suptitle("Sources of variability in initial (gen-0) cooperation", fontsize=13)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
