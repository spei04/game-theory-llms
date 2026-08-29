#!/usr/bin/env python3
"""
Source-of-variance analysis for the DIRECT EVOLUTION runs.

Same type-II decomposition as figures/variance_decomposition.py (cultural
evolution), with the selection-noise parameter beta taking the prompt
variant's slot:

    executor model | beta | model x beta | run stochasticity (residual)

Data: simulations/direct_evo_{beta}_old/{model}/Final_Accum*.json
(model-matched original executors; betas 0.01, 1, 5, 50, 100, 500).
In direct evolution the LLM only writes the gen-1 strategies — generations
2+ are verbatim softmax(score/beta) copying — so any growth of the run
share over generations is pure selection/drift lottery, with no LLM
rewriting in the loop.

Output: figures/run_variation/variance_decomposition_direct_evo.png / .json

Usage:
    python figures/variance_decomposition_direct_evo.py [--n_perm 1000]
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variance_decomposition import (design_matrices, decompose, shares,
                                    permutation_pvals)

BETAS = [("001", "0.01"), ("1", "1"), ("5", "5"),
         ("50", "50"), ("100", "100"), ("500", "500")]
MODELS = ["claude", "gpt", "gemini"]
_TS_RE = re.compile(r'_(\d{8}_\d{6})\.json$')

C_MODEL  = "#2a78d6"
C_BETA   = "#1baf7a"
C_INTER  = "#eda100"
C_RUN    = "#c3c2b7"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def run_cooperation_by_gen(accum_path):
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


def main():
    ap = argparse.ArgumentParser(description="Variance decomposition, direct evolution.")
    ap.add_argument("--sim_root", default="simulations")
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--output_dir", default="figures/run_variation")
    args = ap.parse_args()

    rows = []
    for bdir, blabel in BETAS:
        for model in MODELS:
            paths = sorted(glob.glob(os.path.join(
                args.sim_root, f"direct_evo_{bdir}_old", model, "Final_Accum*.json")))
            for p in paths:
                coop = run_cooperation_by_gen(p)
                if not coop:
                    continue
                row = {"experiment": f"beta={blabel}", "model": model}
                for g in range(1, 11):
                    row[f"coop_gen{g}"] = coop.get(g, np.nan)
                rows.append(row)
    df = pd.DataFrame(rows)
    print(f"{len(df)} direct-evolution runs "
          f"({df.groupby(['experiment','model']).size().min()}-"
          f"{df.groupby(['experiment','model']).size().max()} per cell)")

    rng = np.random.default_rng(0)
    gens = list(range(1, 11))
    results = {}
    print(f"{'gen':>3s} {'model%':>7s} {'beta%':>7s} {'inter%':>7s} {'run%':>6s}"
          f"   (permutation p)")
    for g in gens:
        sub = df.dropna(subset=[f"coop_gen{g}"])
        y = sub[f"coop_gen{g}"].to_numpy(float)
        M, P, I = design_matrices(sub)
        comp = decompose(y, M, P, I)
        sh = shares(comp)
        pv = permutation_pvals(y, M, P, I, args.n_perm, rng)
        results[g] = {"n_runs": len(sub), "shares": sh, "ss": comp, "perm_p": pv}
        print(f"{g:3d} {sh['model']*100:6.1f}% {sh['prompt']*100:6.1f}% "
              f"{sh['interaction']*100:6.1f}% {sh['run']*100:5.1f}%"
              f"   p_model={pv['model']:.3f} p_beta={pv['prompt']:.3f} "
              f"p_inter={pv['interaction']:.3f}")

    fig, ax = plt.subplots(figsize=(8.5, 5.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    order = [("model", C_MODEL, "executor model (gen-1 pool author)"),
             ("prompt", C_BETA, "selection noise β"),
             ("interaction", C_INTER, "model × β"),
             ("run", C_RUN, "run stochasticity (within cell)")]
    stack = np.array([[results[g]["shares"][k] * 100 for g in gens]
                      for k, _, _ in order])
    ax.stackplot(gens, stack, colors=[c for _, c, _ in order],
                 labels=[l for _, _, l in order], alpha=0.9)
    ax.set_xlim(gens[0], gens[-1])
    ax.set_ylim(0, 100)
    ax.set_xticks(gens)
    ax.set_xlabel("generation", fontsize=11, color=INK)
    ax.set_ylabel("share of across-run cooperation variance (%)", fontsize=10, color=INK)
    ax.set_title("Direct evolution: variance shares by generation "
                 "(LLM only writes generation 1)", fontsize=12, color=INK)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.7)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], fontsize=9, frameon=True,
              facecolor=SURFACE, edgecolor=GRID, loc="center left",
              bbox_to_anchor=(1.01, 0.5))
    fig.text(0.99, 0.01, f"Type-II SS on per-run cooperation, {len(df)} runs "
             "(6 β levels × 3 executors × ~20 runs; baseline prompt only)",
             ha="right", fontsize=7, color=MUTED)
    fig.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "variance_decomposition_direct_evo.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    with open(os.path.join(args.output_dir,
                           "variance_decomposition_direct_evo.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
