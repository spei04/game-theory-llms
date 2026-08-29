#!/usr/bin/env python3
"""
Layer A of the variability-prediction analysis: where does the run-to-run
variability in cooperation live?

For each generation, decompose the variance of per-run cooperation into:
    executor model | prompt variant | model x prompt interaction |
    run-level stochasticity (residual within cell)
using type-II sums of squares from nested OLS fits on the run feature table
(one observation per run per generation). Because each run contributes one
value per generation, the within-cell residual IS the run-stochasticity
component. Factor p-values come from a permutation test (cooperation values
shuffled across runs).

Input : figures/run_variation/run_feature_table.csv (build_run_feature_table.py)
Output: figures/run_variation/variance_decomposition.png
        figures/run_variation/variance_decomposition.json

Usage:
    python figures/variance_decomposition.py \
        [--table figures/run_variation/run_feature_table.csv] \
        [--n_perm 1000]
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# component colors (validated palette): identity by component, fixed order
C_MODEL  = "#2a78d6"   # blue
C_PROMPT = "#1baf7a"   # aqua
C_INTER  = "#eda100"   # yellow
C_RUN    = "#c3c2b7"   # gray — run-level stochasticity (residual)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def design_matrices(df):
    """Full-rank dummy blocks for model, prompt, and their interaction."""
    M = pd.get_dummies(df["model"], prefix="m", drop_first=True).to_numpy(float)
    P = pd.get_dummies(df["experiment"], prefix="p", drop_first=True).to_numpy(float)
    inter_labels = df["model"] + ":" + df["experiment"]
    I = pd.get_dummies(inter_labels, prefix="i", drop_first=True).to_numpy(float)
    # interaction block = cell dummies minus main-effect span; using cell
    # dummies directly in the "full" model spans main effects + interaction.
    return M, P, I


def sse(X, y):
    """Residual sum of squares of OLS y ~ [1, X]."""
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    r = y - X1 @ beta
    return float(r @ r)


def decompose(y, M, P, I):
    """Type-II variance components. Returns dict of SS by component."""
    ss_total = float(((y - y.mean()) ** 2).sum())
    sse_full  = sse(I, y)                       # cell dummies = A+B+AB
    sse_ab    = sse(np.column_stack([M, P]), y) # main effects only
    sse_a     = sse(M, y)
    sse_b     = sse(P, y)
    return {
        "model":       max(sse_b - sse_ab, 0.0),   # SS(A | B)
        "prompt":      max(sse_a - sse_ab, 0.0),   # SS(B | A)
        "interaction": max(sse_ab - sse_full, 0.0),
        "run":         sse_full,                    # within-cell residual
        "total":       ss_total,
    }


def shares(comp):
    parts = ["model", "prompt", "interaction", "run"]
    s = sum(comp[k] for k in parts)
    return {k: comp[k] / s for k in parts}


def permutation_pvals(y, M, P, I, n_perm, rng):
    obs = decompose(y, M, P, I)
    counts = {k: 0 for k in ("model", "prompt", "interaction")}
    for _ in range(n_perm):
        yp = rng.permutation(y)
        perm = decompose(yp, M, P, I)
        for k in counts:
            if perm[k] >= obs[k]:
                counts[k] += 1
    return {k: (v + 1) / (n_perm + 1) for k, v in counts.items()}


def main():
    ap = argparse.ArgumentParser(description="Variance decomposition of cooperation.")
    ap.add_argument("--table", default="figures/run_variation/run_feature_table.csv")
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--output_dir", default="figures/run_variation")
    args = ap.parse_args()

    df = pd.read_csv(args.table)
    gens = [g for g in range(1, 11) if f"coop_gen{g}" in df.columns]
    rng = np.random.default_rng(0)

    results = {}
    print(f"{'gen':>3s} {'model%':>7s} {'prompt%':>8s} {'inter%':>7s} {'run%':>6s}"
          f"   (permutation p at this gen)")
    for g in gens:
        sub = df.dropna(subset=[f"coop_gen{g}"])
        y = sub[f"coop_gen{g}"].to_numpy(float)
        M, P, I = design_matrices(sub)
        comp = decompose(y, M, P, I)
        sh = shares(comp)
        pv = permutation_pvals(y, M, P, I, args.n_perm, rng)
        results[g] = {"n_runs": len(sub), "shares": sh, "ss": comp, "perm_p": pv}
        print(f"{g:3d} {sh['model']*100:6.1f}% {sh['prompt']*100:7.1f}% "
              f"{sh['interaction']*100:6.1f}% {sh['run']*100:5.1f}%"
              f"   p_model={pv['model']:.3f} p_prompt={pv['prompt']:.3f} "
              f"p_inter={pv['interaction']:.3f}")

    # ---- stacked-area figure over generations
    fig, ax = plt.subplots(figsize=(8.5, 5.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    order = [("model", C_MODEL, "executor model"),
             ("prompt", C_PROMPT, "prompt variant"),
             ("interaction", C_INTER, "model × prompt"),
             ("run", C_RUN, "run stochasticity (within cell)")]
    xs = gens
    stack = np.array([[results[g]["shares"][k] * 100 for g in xs]
                      for k, _, _ in order])
    ax.stackplot(xs, stack, colors=[c for _, c, _ in order],
                 labels=[l for _, _, l in order], alpha=0.9)
    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(0, 100)
    ax.set_xticks(xs)
    ax.set_xlabel("generation", fontsize=11, color=INK)
    ax.set_ylabel("share of across-run cooperation variance (%)", fontsize=10, color=INK)
    ax.set_title("What explains cooperation variability? — variance shares by generation",
                 fontsize=12, color=INK)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=9, frameon=True,
              facecolor=SURFACE, edgecolor=GRID, loc="center left",
              bbox_to_anchor=(1.01, 0.5))
    fig.text(0.99, 0.01, "Type-II SS on per-run cooperation, 283 runs "
             "(5 neutral prompts × 3 executors × ~20 runs)",
             ha="right", fontsize=7, color=MUTED)
    fig.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "variance_decomposition.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    with open(os.path.join(args.output_dir, "variance_decomposition.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
    print(f"Saved: {os.path.join(args.output_dir, 'variance_decomposition.json')}")


if __name__ == "__main__":
    main()
