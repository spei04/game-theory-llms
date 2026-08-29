#!/usr/bin/env python3
"""
Layer B of the variability-prediction analysis: which factors make runs SPREAD?

Two dispersion targets, one row per run (from the run feature table):
  between-run dispersion : |final_coop - cell mean|   (Brown-Forsythe target)
  within-run volatility  : log10 detrended std of the cooperation trajectory

Each target gets the same type-II decomposition as Layer A (model / prompt /
model x prompt / residual), in two variants:
  raw            : factors only
  level-adjusted : the cell's mean cooperation level enters first as a
                   covariate, absorbing the mechanical variance compression
                   near the 0% floor (collapsed cells can't spread). The
                   factor shares then reflect dispersion differences beyond
                   what the cooperation level itself explains.

Permutation p-values: target values shuffled across runs (1000x).

Input : figures/run_variation/run_feature_table.csv
Output: figures/run_variation/dispersion_regression.png / .json

Usage:
    python figures/dispersion_regression.py [--table ...] [--n_perm 1000]
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variance_decomposition import design_matrices, sse

C_LEVEL  = "#4a3aa7"   # violet — cooperation-level covariate
C_MODEL  = "#2a78d6"
C_PROMPT = "#1baf7a"
C_INTER  = "#eda100"
C_RUN    = "#c3c2b7"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def decompose_with_cov(y, M, P, I, cov=None):
    """Type-II SS with an optional covariate entering every fit first."""
    n = len(y)
    C = (np.empty((n, 0)) if cov is None else np.asarray(cov, float).reshape(n, -1))
    sse0     = sse(C, y)
    sse_a    = sse(np.column_stack([C, M]), y)
    sse_b    = sse(np.column_stack([C, P]), y)
    sse_ab   = sse(np.column_stack([C, M, P]), y)
    sse_full = sse(np.column_stack([C, I]), y)
    ss_total = float(((y - y.mean()) ** 2).sum())
    comp = {
        "level":       max(ss_total - sse0, 0.0) if cov is not None else 0.0,
        "model":       max(sse_b - sse_ab, 0.0),
        "prompt":      max(sse_a - sse_ab, 0.0),
        "interaction": max(sse_ab - sse_full, 0.0),
        "run":         sse_full,
    }
    return comp


def shares(comp):
    keys = ["level", "model", "prompt", "interaction", "run"]
    s = sum(comp[k] for k in keys)
    return {k: comp[k] / s for k in keys}


def perm_p(y, M, P, I, cov, n_perm, rng):
    obs = decompose_with_cov(y, M, P, I, cov)
    counts = {k: 0 for k in ("model", "prompt", "interaction")}
    for _ in range(n_perm):
        yp = rng.permutation(y)
        pm = decompose_with_cov(yp, M, P, I, cov)
        for k in counts:
            if pm[k] >= obs[k]:
                counts[k] += 1
    return {k: (v + 1) / (n_perm + 1) for k, v in counts.items()}


def main():
    ap = argparse.ArgumentParser(description="Dispersion regression (Layer B).")
    ap.add_argument("--table", default="figures/run_variation/run_feature_table.csv")
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--output_dir", default="figures/run_variation")
    args = ap.parse_args()

    df = pd.read_csv(args.table).dropna(subset=["volatility_detrended"])
    M, P, I = design_matrices(df)
    cov = df["cell_mean_final"].to_numpy(float)
    rng = np.random.default_rng(0)

    targets = {
        "between-run dispersion\n|final − cell mean| (pp)":
            df["abs_dev_from_cell"].to_numpy(float),
        "within-run volatility\nlog10 detrended std":
            np.log10(np.clip(df["volatility_detrended"].to_numpy(float), 0.1, None)),
    }

    results, bars = {}, []
    for tname, y in targets.items():
        key = tname.split("\n")[0]
        results[key] = {}
        for variant, c in (("raw", None), ("level-adjusted", cov)):
            comp = decompose_with_cov(y, M, P, I, c)
            sh = shares(comp)
            pv = perm_p(y, M, P, I, c, args.n_perm, rng)
            results[key][variant] = {"shares": sh, "perm_p": pv}
            bars.append((f"{tname}\n[{variant}]", sh, pv))
            print(f"[{key} | {variant}] "
                  + " ".join(f"{k}={sh[k]*100:.1f}%" for k in
                             ("level", "model", "prompt", "interaction", "run"))
                  + "  p: " + " ".join(f"{k}={pv[k]:.3f}" for k in pv))

    # ---- horizontal stacked bars
    order = [("level", C_LEVEL, "cooperation level (covariate)"),
             ("model", C_MODEL, "executor model"),
             ("prompt", C_PROMPT, "prompt variant"),
             ("interaction", C_INTER, "model × prompt"),
             ("run", C_RUN, "run stochasticity (residual)")]
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ys = np.arange(len(bars))[::-1]
    for (label, sh, pv), ypos in zip(bars, ys):
        left = 0.0
        for k, color, _ in order:
            w = sh[k] * 100
            if w <= 0:
                continue
            ax.barh(ypos, w, left=left, color=color, height=0.62, zorder=2)
            if w > 6:
                star = ""
                if k in pv and pv[k] < 0.05:
                    star = "*"
                ax.text(left + w / 2, ypos, f"{w:.0f}{star}", ha="center",
                        va="center", fontsize=8,
                        color="white" if k != "run" else INK)
            left += w
    ax.set_yticks(ys)
    ax.set_yticklabels([b[0] for b in bars], fontsize=8.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of dispersion-target variance (%)   (* = permutation p<0.05)",
                  fontsize=9.5, color=INK)
    ax.set_title("What makes runs spread? — dispersion variance shares (Layer B)",
                 fontsize=12, color=INK)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in order]
    ax.legend(handles, [l for _, _, l in order], fontsize=8, frameon=False,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "dispersion_regression.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    with open(os.path.join(args.output_dir, "dispersion_regression.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
