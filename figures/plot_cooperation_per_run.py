#!/usr/bin/env python3
"""
Cooperation rate per generation, one line per run, one color per run.

Cooperation rate = mean donation % across all donor rounds in a generation
(same definition as plot_cooperation_by_generation.py). One Final_Accum
file = one run; runs are numbered by sorted timestamp, matching the run
numbering in figures/run_variation_analysis.py output.

One plot per (experiment, model), saved alongside the run-variation figures:
    figures/run_variation/<experiment>/<model>/cooperation_per_run.png

Usage:
    python figures/plot_cooperation_per_run.py \
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

_TS_RE = re.compile(r'_(\d{8}_\d{6})\.json$')


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


def plot_experiment_model(sim_dir, out_path, title):
    files = sorted(glob.glob(os.path.join(sim_dir, "Final_Accum*.json")),
                   key=lambda p: (_TS_RE.search(p).group(1) if _TS_RE.search(p) else p))
    if not files:
        return None

    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    finals = []
    all_gens = set()
    for i, path in enumerate(files):
        coop = run_cooperation_by_gen(path)
        gens = sorted(coop)
        all_gens.update(gens)
        color = cmap(i % 20)
        ax.plot(gens, [coop[g] for g in gens], color=color, lw=1.8,
                marker="o", ms=3.5, label=f"run {i + 1}")
        if gens:
            finals.append(coop[gens[-1]])

    ax.set_xlabel("generation", fontsize=11)
    ax.set_ylabel("cooperation rate (mean donation %)", fontsize=11)
    ax.set_xticks(sorted(all_gens))
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return len(files), (float(np.mean(finals)) if finals else float("nan")), \
        (float(np.std(finals)) if finals else float("nan"))


def main():
    ap = argparse.ArgumentParser(description="Cooperation rate per generation per run.")
    ap.add_argument("--sim_root", default="simulations")
    ap.add_argument("--experiments", nargs="+",
                    default=["neutral_baseline", "neutral_claude", "neutral_gemini",
                             "neutral_gpt", "neutral_deepseek"])
    ap.add_argument("--models", nargs="+", default=["claude", "gpt", "gemini"])
    ap.add_argument("--output_dir", default="figures/run_variation")
    args = ap.parse_args()

    for exp in args.experiments:
        for model in args.models:
            sim_dir = os.path.join(args.sim_root, exp, model)
            if not os.path.isdir(sim_dir):
                print(f"  [{exp}/{model}] {sim_dir} not found — skipping")
                continue
            out_path = os.path.join(args.output_dir, exp, model, "cooperation_per_run.png")
            res = plot_experiment_model(sim_dir, out_path,
                                        f"{exp}/{model} — cooperation rate per run")
            if res:
                n, mean_f, std_f = res
                print(f"  [{exp}/{model}] {n} runs | final-gen cooperation "
                      f"{mean_f:.1f} ± {std_f:.1f} % | saved {out_path}")


if __name__ == "__main__":
    main()
