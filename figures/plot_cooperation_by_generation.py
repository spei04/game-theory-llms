#!/usr/bin/env python3
"""
Cooperation distribution BY GENERATION, pooling all setups × models × runs.

For each generation 1..10, gather every agent's total_final_score across every
run in every (setup, model) cell we have data for. Then plot one distribution
per generation on the x-axis.

Answers: does cooperation systematically evolve upward over generations,
independent of prompt or model?

Two panels:
  Left  — per-agent total_final_score  (finest granularity, ~thousands of pts/gen)
  Right — per-run mean donation %      (one number per run, ~500 pts/gen)

Usage:
    python figures/plot_cooperation_by_generation.py [--output_dir ...]
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

BASE_POOL = 240   # 10 units × 12 rounds × 2 games — see donor_game_baseline.py


def find_all_accum_files():
    """Every Final_Accum*.json under simulations/, recursively."""
    return sorted(glob.glob("simulations/**/Final_Accum*.json", recursive=True))


def per_gen_data(accum_path):
    """Returns two dicts keyed by generation:
        agent_scores[gen] = [total_final_score per agent in gen]
        run_donation_pct[gen] = single mean donation % across all rounds in gen"""
    with open(accum_path) as f:
        data = json.load(f)

    # Per-agent aggregation → total_final_score = 240 - Σdonated + Σreceived
    per_agent = defaultdict(lambda: defaultdict(lambda: {"donated": 0.0, "received": 0.0}))
    # Donation % across every donor round in the generation
    donation_pcts = defaultdict(list)

    for r in data.get("agents_data", []):
        g = r["current_generation"]
        a = r["agent_name"]
        per_agent[g][a]["donated"]  += r.get("donated", 0) or 0
        per_agent[g][a]["received"] += r.get("received", 0) or 0
        if r.get("is_donor"):
            donated   = r.get("donated", 0) or 0
            remaining = r.get("resources", 0) or 0
            denom = donated + remaining
            if denom > 0:
                donation_pcts[g].append(100.0 * donated / denom)

    agent_scores = {g: [BASE_POOL - v["donated"] + v["received"]
                        for v in agents.values()]
                    for g, agents in per_agent.items()}
    run_donation = {g: float(np.mean(vs)) for g, vs in donation_pcts.items() if vs}
    return agent_scores, run_donation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="figures/cooperation_distributions")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    files = find_all_accum_files()
    print(f"Scanning {len(files)} Final_Accum files ...")

    scores_by_gen   = defaultdict(list)   # gen -> list of agent-level scores
    donation_by_gen = defaultdict(list)   # gen -> list of run-level donation %

    for i, f in enumerate(files):
        try:
            agent_scores, run_donation = per_gen_data(f)
        except Exception as e:
            print(f"  skip {f}: {e}")
            continue
        for g, scs in agent_scores.items():
            scores_by_gen[g].extend(scs)
        for g, dp in run_donation.items():
            donation_by_gen[g].append(dp)
        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1} / {len(files)}")

    gens = sorted(scores_by_gen)
    print(f"\nGenerations found: {gens}")
    print(f"Samples per gen (agent-level scores): "
          f"{[len(scores_by_gen[g]) for g in gens]}")
    print(f"Samples per gen (run-level donation %): "
          f"{[len(donation_by_gen[g]) for g in gens]}")

    # ── Plot: overlapping KDE ("bell curves") — one per generation ──
    from scipy.stats import gaussian_kde
    cmap = plt.get_cmap("viridis")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left: per-agent total_final_score
    x1 = np.linspace(min(min(scores_by_gen[g]) for g in gens),
                     max(max(scores_by_gen[g]) for g in gens), 400)
    for i, g in enumerate(gens):
        vals = scores_by_gen[g]
        kde = gaussian_kde(vals, bw_method=0.35)
        color = cmap(i / max(len(gens) - 1, 1))
        ax1.plot(x1, kde(x1), color=color, lw=1.8,
                 label=f"gen {g}")
        ax1.fill_between(x1, kde(x1), color=color, alpha=0.08)
    ax1.set_xlabel("Per-agent total_final_score", fontsize=11)
    ax1.set_ylabel("density", fontsize=10)
    ax1.set_title(f"Score bell curves per generation\n"
                  f"(pooling every agent × run × setup × model — "
                  f"{sum(len(scores_by_gen[g]) for g in gens):,} points)",
                  fontsize=10)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.9,
               title="generation")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: per-run mean donation %
    x2 = np.linspace(0, 100, 400)
    for i, g in enumerate(gens):
        vals = donation_by_gen[g]
        if len(vals) < 2:
            continue
        kde = gaussian_kde(vals, bw_method=0.35)
        color = cmap(i / max(len(gens) - 1, 1))
        ax2.plot(x2, kde(x2), color=color, lw=1.8,
                 label=f"gen {g}")
        ax2.fill_between(x2, kde(x2), color=color, alpha=0.08)
    ax2.set_xlabel("Per-run mean donation % in gen", fontsize=11)
    ax2.set_ylabel("density", fontsize=10)
    ax2.set_title(f"Donation bell curves per generation\n"
                  f"(pooling every run × setup × model — "
                  f"{sum(len(donation_by_gen[g]) for g in gens):,} runs)",
                  fontsize=10)
    ax2.set_xlim(0, 100)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.9,
               title="generation")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    out = os.path.join(args.output_dir, "cooperation_by_generation_bellcurves.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")

    # ── Numerical summary ──
    print("\n" + "=" * 62)
    print(f"{'gen':>4s}  {'n_agents':>9s}  {'mean score':>12s}  {'n_runs':>7s}  {'mean donation %':>16s}")
    print("=" * 62)
    for g in gens:
        s = scores_by_gen[g]; d = donation_by_gen[g]
        print(f"{g:>4d}  {len(s):>9d}  {np.mean(s):>10.2f}    {len(d):>7d}  {np.mean(d):>13.2f} %")


if __name__ == "__main__":
    main()
