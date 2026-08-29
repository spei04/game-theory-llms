#!/usr/bin/env python3
"""
Plot the original-vs-current-setup comparison of gen-1 coded-strategy
self-cooperation AUC (see compare_original_vs_current_auc.py).

Gen 0 here = the initial, pre-evolution strategies (generation==1 in the
raw data files).

Outputs (figures/compare_setups/):
  gen0_auc_by_setup.png       grouped bars: mean AUC ± std per model x setup,
                              with Welch p (and Cohen's d) vs the original
  gen0_auc_distributions.png  box plots of the underlying AUC distributions

Usage:
    python figures/plot_compare_setups.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_original_vs_current_auc import (
    load_gen1_aucs, MODELS, ORIGINAL_BATCHES, CURRENT_BATCHES)

OUTPUT_DIR = "figures/compare_setups"

MODEL_LABELS = {"claude": "Claude 3.5 Sonnet", "gpt": "GPT-4o",
                "gemini": "Gemini 2.5 Flash"}

CONDITIONS = [
    ("Original (Apr, pooled)", None),
    ("Current — May full runs", "neutral_baseline (May, full runs)"),
    ("Current — Jul 200-run batch", "neutral_baseline_200 (Jul, strategies-only)"),
]
# Validated categorical palette (dataviz slots 1-3, light mode)
COND_COLORS = ["#2a78d6", "#008300", "#e87ba4"]

# Chart chrome / ink tokens (light mode)
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"


def collect():
    """Return {model: [(label, values), ...]} in CONDITIONS order."""
    data = {}
    for model in MODELS:
        orig = [load_gen1_aucs(d, model) for d in ORIGINAL_BATCHES.values()]
        orig = [v for v in orig if v is not None and len(v)]
        sets = [(CONDITIONS[0][0], np.concatenate(orig))]
        for label, key in CONDITIONS[1:]:
            vals = load_gen1_aucs(CURRENT_BATCHES[key], model)
            sets.append((label, vals if vals is not None else np.array([])))
        data[model] = sets
    return data


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def _p_text(p):
    if p < 1e-4:
        return "p<10⁻⁴"
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.2f}" if p >= 0.01 else f"p={p:.3f}"


def plot_bars(data, path):
    fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor=SURFACE)
    _style_axes(ax)

    n_cond = len(CONDITIONS)
    slot_w = 0.26
    offsets = (np.arange(n_cond) - (n_cond - 1) / 2) * slot_w

    n_caption = []
    for mi, model in enumerate(MODELS):
        sets = data[model]
        orig_vals = sets[0][1]
        n_caption.append(
            f"{MODEL_LABELS[model]}: " +
            "/".join(str(len(v)) for _, v in sets))
        for ci, (label, vals) in enumerate(sets):
            x = mi + offsets[ci]
            mean, std = np.mean(vals), np.std(vals)
            ax.bar(x, mean, width=slot_w * 0.88, color=COND_COLORS[ci],
                   label=label if mi == 0 else None, zorder=3)
            ax.errorbar(x, mean, yerr=std, fmt="none", ecolor=INK_2,
                        elinewidth=1.2, capsize=3, zorder=4)
            # value on the cap (above the whisker)
            ax.text(x, mean + std + 0.012, f"{mean:.3f}", ha="center",
                    va="bottom", fontsize=8.5, color=INK, zorder=5)
            # stats vs original for the two current-setup bars
            if ci > 0:
                _, p = sps.ttest_ind(orig_vals, vals, equal_var=False)
                pooled_sd = np.sqrt((np.var(orig_vals, ddof=1)
                                     + np.var(vals, ddof=1)) / 2)
                d = (np.mean(vals) - np.mean(orig_vals)) / pooled_sd
                note = _p_text(p)
                if p < 0.05:
                    note += f"\nd={d:+.2f}"
                ax.text(x, mean + std + 0.045, note, ha="center",
                        va="bottom", fontsize=7.5, color=MUTED, zorder=5)

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)
    ax.set_ylim(0, 0.72)
    ax.set_yticks(np.arange(0, 0.71, 0.1))
    ax.set_ylabel("Self-cooperation AUC (normalised, 0–1)",
                  fontsize=9.5, color=INK_2)
    ax.set_title("Gen-0 (initial) coded-strategy cooperation: original vs current setup",
                 fontsize=12, color=INK, pad=48, loc="left")
    ax.text(0, 1.13, "Baseline prompt only · mean ± std · "
                     "Welch p and Cohen's d vs original",
            transform=ax.transAxes, fontsize=9, color=INK_2)
    ax.legend(fontsize=8.5, frameon=False, ncols=3, loc="lower left",
              bbox_to_anchor=(0, 1.0), labelcolor=INK_2)
    fig.text(0.01, 0.008, "n per bar — " + " ;  ".join(n_caption),
             fontsize=7.5, color=MUTED)

    plt.tight_layout(rect=(0, 0.03, 1, 1))
    plt.savefig(path, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close()
    print(f"Saved: {path}")


def plot_distributions(data, path):
    fig, ax = plt.subplots(figsize=(9.5, 5.2), facecolor=SURFACE)
    _style_axes(ax)

    n_cond = len(CONDITIONS)
    slot_w = 0.26
    offsets = (np.arange(n_cond) - (n_cond - 1) / 2) * slot_w

    for mi, model in enumerate(MODELS):
        for ci, (label, vals) in enumerate(data[model]):
            x = mi + offsets[ci]
            bp = ax.boxplot(vals, positions=[x], widths=slot_w * 0.72,
                            patch_artist=True, whis=(5, 95),
                            showfliers=False, zorder=3)
            c = COND_COLORS[ci]
            bp["boxes"][0].set(facecolor=c, alpha=0.15, edgecolor=c,
                               linewidth=1.6)
            for part in ("whiskers", "caps"):
                for artist in bp[part]:
                    artist.set(color=c, linewidth=1.4)
            bp["medians"][0].set(color=INK, linewidth=1.6)
            # A collapsed box means the quartiles coincide (a point mass —
            # e.g. Gemini's many flat "donate 50%" strategies). Say so.
            q25, q50, q75 = np.percentile(vals, [25, 50, 75])
            if q75 - q25 < 0.005:
                share = np.mean(np.isclose(vals, q50, atol=0.005)) * 100
                ax.text(x, q50 - 0.045, f"{share:.0f}%\n= {q50:.2f}",
                        ha="center", va="top", fontsize=7, color=MUTED)
            if mi == 0:
                ax.bar(0, 0, color=c, alpha=0.6, label=label)  # legend proxy

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=10)
    ax.set_xlim(-0.55, len(MODELS) - 0.45)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Self-cooperation AUC (normalised, 0–1)",
                  fontsize=9.5, color=INK_2)
    ax.set_title("Gen-0 (initial) coded-strategy AUC distributions by setup",
                 fontsize=12, color=INK, pad=30, loc="left")
    ax.text(0, 1.06, "Baseline prompt only · box = IQR, whiskers = "
                     "5th–95th percentile, line = median",
            transform=ax.transAxes, fontsize=9, color=INK_2)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right",
              labelcolor=INK_2)

    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close()
    print(f"Saved: {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = collect()
    plot_bars(data, os.path.join(OUTPUT_DIR, "gen0_auc_by_setup.png"))
    plot_distributions(data,
                       os.path.join(OUTPUT_DIR, "gen0_auc_distributions.png"))


if __name__ == "__main__":
    main()
