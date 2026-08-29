#!/usr/bin/env python3
"""
Coded strategy curves (my donation % vs pair's previous donation %) for the
first vs final generation of the coded_new_* runs (STRATEGY_MODE='coded').

Source: simulations/coded_new_{model}/all_strategies_*.json — one file per
run, list[generation][agent] records carrying the classifier-emitted
strategy_code. Curves are evaluated with the same conventions as
compute_auc.py.

One panel per model, two conditions per panel:
  gen 1 (initial)  — gray, thin individual curves + bold mean
  gen 10 (final)   — model color, thin individual curves + bold mean

Output: figures/coded_strats/coded_new_gen1_vs_gen10.png (+ stats JSON)

Usage:
    python figures/plot_coded_new_gen_comparison.py \
        [--sim_root simulations] [--models claude gpt gemini] \
        [--gens 1 10] [--output_dir figures/coded_strats]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_auc import eval_strategy, auc_normalised, X_VALUES

MODEL_LABELS = {"claude": "Claude 3.5 Sonnet", "gpt": "GPT-4o", "gemini": "Gemini 2.5 Flash"}
MODEL_COLORS = {"claude": "#FF6B35", "gpt": "#10A37F", "gemini": "#4285F4"}
GEN1_COLOR = "#888888"


def load_gen_curves(sim_dir, generation):
    """All evaluable curves for one generation across every run in sim_dir."""
    curves = []
    for path in sorted(glob.glob(os.path.join(sim_dir, "all_strategies_*.json"))):
        gens = json.load(open(path))
        if len(gens) < generation:
            continue
        for rec in gens[generation - 1]:
            code = rec.get("strategy_code")
            if not code:
                continue
            ys = eval_strategy(code)
            if ys is not None:
                curves.append(ys)
    return curves


def main():
    ap = argparse.ArgumentParser(description="coded_new gen-1 vs gen-10 strategy curves.")
    ap.add_argument("--sim_root", default="simulations")
    ap.add_argument("--models", nargs="+", default=["claude", "gpt", "gemini"])
    ap.add_argument("--gens", nargs=2, type=int, default=[1, 10])
    ap.add_argument("--output_dir", default="figures/coded_strats")
    args = ap.parse_args()
    g_first, g_last = args.gens

    fig, axes = plt.subplots(1, len(args.models), figsize=(5.2 * len(args.models), 4.6),
                             sharey=True)
    axes = np.atleast_1d(axes)
    stats = {}

    for ax, model in zip(axes, args.models):
        sim_dir = os.path.join(args.sim_root, f"coded_new_{model}")
        color = MODEL_COLORS.get(model, "#555555")
        stats[model] = {}
        for gen, c, z in ((g_first, GEN1_COLOR, 2), (g_last, color, 3)):
            curves = load_gen_curves(sim_dir, gen)
            if not curves:
                print(f"  [{model}] gen {gen}: no curves")
                continue
            arr = np.vstack(curves)
            for ys in arr:
                ax.plot(X_VALUES, ys, color=c, alpha=0.10, lw=0.8, zorder=z)
            ax.plot(X_VALUES, arr.mean(axis=0), color=c, lw=2.6, zorder=z + 2,
                    label=f"gen {gen} mean (n={len(arr)})")
            aucs = [auc_normalised(ys) for ys in arr]
            stats[model][f"gen{gen}"] = {
                "n_strategies": len(arr),
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs, ddof=1)),
                "first_response_at_0": float(arr[:, 0].mean()),
                "response_at_100": float(arr[:, -1].mean()),
            }
            print(f"  [{model}] gen {gen}: n={len(arr)}  AUC {np.mean(aucs):.3f} ± {np.std(aucs, ddof=1):.3f}")

        ax.set_title(f"{MODEL_LABELS.get(model, model)} (coded_new)", fontsize=11)
        ax.set_xlabel("pair's donation % in previous round", fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(-2, 102)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, frameon=False, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("my donation %", fontsize=10)

    fig.suptitle(f"Coded strategies: generation {g_first} vs generation {g_last} "
                 "(mode-coded runs, all runs pooled)", fontsize=12)
    fig.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"coded_new_gen{g_first}_vs_gen{g_last}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(args.output_dir,
                           f"coded_new_gen{g_first}_vs_gen{g_last}_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
