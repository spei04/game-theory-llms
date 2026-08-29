#!/usr/bin/env python3
"""
Variance of cooperation rate across prompt variants.

Two components of variability, per (prompt variant, executor model):
  between-run  — std across run-mean cooperation rates (how differently do
                 independent runs of the same setup end up?)
  within-run   — detrended std of each run's per-generation cooperation
                 (round-to-round volatility; distribution shown across runs)

Cooperation rate = mean donation % across all donor rounds in a generation
(same definition as plot_cooperation_by_generation.py). Detrending as in
analyze_run_variance.py (residuals of a linear fit vs generation).

Output:
  figures/run_variation/variance_by_prompt.png
  figures/run_variation/variance_by_prompt.json

Usage:
    python figures/plot_variance_by_prompt.py \
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

MODEL_COLORS = {          # same as plot_coded_strategies.py
    "claude": "#FF6B35",
    "gpt":    "#10A37F",
    "gemini": "#4285F4",
}


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


def collect(sim_dir):
    """Per run: mean cooperation + detrended std. Returns (means, det_stds)."""
    files = sorted(glob.glob(os.path.join(sim_dir, "Final_Accum*.json")),
                   key=lambda p: (_TS_RE.search(p).group(1) if _TS_RE.search(p) else p))
    means, det_stds = [], []
    for path in files:
        coop = run_cooperation_by_gen(path)
        gens = np.array(sorted(coop))
        if len(gens) < 4:
            continue
        vals = np.array([coop[g] for g in gens])
        slope, intercept = np.polyfit(gens, vals, 1)
        resid = vals - (slope * gens + intercept)
        means.append(float(vals.mean()))
        det_stds.append(float(resid.std(ddof=2)))
    return means, det_stds


def main():
    ap = argparse.ArgumentParser(description="Cooperation variance across prompt variants.")
    ap.add_argument("--sim_root", default="simulations")
    ap.add_argument("--experiments", nargs="+",
                    default=["neutral_baseline", "neutral_claude", "neutral_gemini",
                             "neutral_gpt", "neutral_deepseek"])
    ap.add_argument("--models", nargs="+", default=["claude", "gpt", "gemini"])
    ap.add_argument("--output_dir", default="figures/run_variation")
    args = ap.parse_args()

    results = {}
    for exp in args.experiments:
        for model in args.models:
            means, det_stds = collect(os.path.join(args.sim_root, exp, model))
            if len(means) < 3:
                continue
            results[(exp, model)] = {
                "n_runs": len(means),
                "between_run_std": float(np.std(means, ddof=1)),
                "mean_cooperation": float(np.mean(means)),
                "run_means": means,
                "within_run_det_stds": det_stds,
                "within_run_median": float(np.median(det_stds)),
            }

    exps = [e for e in args.experiments if any((e, m) in results for m in args.models)]
    x_base = np.arange(len(exps))
    offsets = np.linspace(-0.25, 0.25, len(args.models))
    rng = np.random.default_rng(0)

    fig, ax1 = plt.subplots(figsize=(9.5, 5.0))

    for mi, model in enumerate(args.models):
        color = MODEL_COLORS.get(model, "#555555")
        first = True
        for ei, exp in enumerate(exps):
            r = results.get((exp, model))
            if not r:
                continue
            x = x_base[ei] + offsets[mi]
            means = np.array(r["run_means"])
            ax1.scatter(x + rng.uniform(-0.06, 0.06, len(means)), means,
                        color=color, s=14, alpha=0.4, zorder=2)
            ax1.errorbar(x, means.mean(), yerr=means.std(ddof=1),
                         fmt="o", color=color, ms=8, capsize=4, lw=1.8,
                         markeredgecolor="white", markeredgewidth=0.8,
                         zorder=3, label=model if first else None)
            first = False

    ax1.set_ylabel("run-mean cooperation (%)", fontsize=10)
    ax1.set_title("Run-mean cooperation (small dots = runs, marker = mean ± std across runs)",
                  fontsize=10, loc="left")
    ax1.legend(fontsize=9, frameon=False, title="executor", title_fontsize=9)
    ax1.set_ylim(-3, 103)
    ax1.set_xticks(x_base)
    ax1.set_xticklabels([e.replace("neutral_", "") for e in exps], fontsize=10)
    ax1.set_xlabel("prompt variant (neutral_*)", fontsize=11)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    fig.suptitle("Cooperation-rate variance across prompt variants", fontsize=12)
    fig.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "variance_by_prompt.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    json_out = {f"{e}/{m}": {k: v for k, v in r.items()
                             if k not in ("within_run_det_stds", "run_means")}
                for (e, m), r in results.items()}
    json_path = os.path.join(args.output_dir, "variance_by_prompt.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)

    print(f"{'experiment/model':30s} {'n':>3s} {'mean coop':>10s} "
          f"{'between-run std':>16s} {'median within-run std':>22s}")
    for (e, m), r in results.items():
        print(f"{e + '/' + m:30s} {r['n_runs']:3d} {r['mean_cooperation']:10.1f} "
              f"{r['between_run_std']:16.2f} {r['within_run_median']:22.2f}")
    print(f"\nSaved: {out}\nSaved: {json_path}")


if __name__ == "__main__":
    main()
