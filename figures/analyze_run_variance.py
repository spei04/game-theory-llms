#!/usr/bin/env python3
"""
Is there a lot of variability in variance across runs?

For each run, take its per-generation cooperation rates (mean donation %,
same definition as plot_cooperation_by_generation.py) and compute:
  std_raw       — std across generations (ddof=1)
  std_detrended — std of residuals from a linear fit vs generation
                  (a run that drifts steadily has high raw std but low
                  detrended std; detrended std = round-to-round volatility)

Per (experiment, model) we then ask how much the per-run variances differ:
  - spread of run stds (min / median / max, max/min ratio)
  - Brown-Forsythe (Levene, center='median') test across runs
  - Spearman correlation between run mean and run std

Figure per experiment: models stacked; each panel shows runs sorted by
detrended std, solid dot = detrended std, open circle = raw std. Run
colors/numbers match the other run_variation figures.

Output:
  figures/run_variation/<experiment>/run_variance_models.png
  figures/run_variation/run_variance_stats.json

Usage:
    python figures/analyze_run_variance.py \
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
from scipy import stats as sps

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


def sorted_run_files(sim_dir):
    return sorted(glob.glob(os.path.join(sim_dir, "Final_Accum*.json")),
                  key=lambda p: (_TS_RE.search(p).group(1) if _TS_RE.search(p) else p))


def run_metrics(files):
    """Per run: id, mean, std_raw, std_detrended, residuals, values."""
    out = []
    for run_id, path in enumerate(files, start=1):
        coop = run_cooperation_by_gen(path)
        gens = np.array(sorted(coop))
        if len(gens) < 4:
            continue
        vals = np.array([coop[g] for g in gens])
        slope, intercept = np.polyfit(gens, vals, 1)
        resid = vals - (slope * gens + intercept)
        out.append({
            "run_id": run_id,
            "mean": float(vals.mean()),
            "std_raw": float(vals.std(ddof=1)),
            "std_detrended": float(resid.std(ddof=2)),
            "slope": float(slope),
            "values": vals,
            "resid": resid,
        })
    return out


def summarize(runs):
    """Variance-of-variance summary + Brown-Forsythe across runs."""
    stds_d = np.array([r["std_detrended"] for r in runs])
    stds_r = np.array([r["std_raw"] for r in runs])
    bf_raw = sps.levene(*[r["values"] for r in runs], center="median")
    bf_det = sps.levene(*[r["resid"] for r in runs], center="median")
    rho, rho_p = sps.spearmanr([r["mean"] for r in runs], stds_d)
    eps = 1e-9
    return {
        "n_runs": len(runs),
        "std_detrended": {"min": float(stds_d.min()), "median": float(np.median(stds_d)),
                          "max": float(stds_d.max()),
                          "max_over_min": float(stds_d.max() / max(stds_d.min(), eps))},
        "std_raw": {"min": float(stds_r.min()), "median": float(np.median(stds_r)),
                    "max": float(stds_r.max()),
                    "max_over_min": float(stds_r.max() / max(stds_r.min(), eps))},
        "brown_forsythe_raw_p": float(bf_raw.pvalue),
        "brown_forsythe_detrended_p": float(bf_det.pvalue),
        "spearman_mean_vs_std": {"rho": float(rho), "p": float(rho_p)},
    }


def plot_experiment(exp, panels, out_path):
    cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(len(panels), 1, figsize=(9.5, 3.0 * len(panels)),
                             sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (model, runs, summ) in zip(axes, panels):
        order = sorted(runs, key=lambda r: r["std_detrended"])
        for x, r in enumerate(order):
            color = cmap((r["run_id"] - 1) % 20)
            ax.vlines(x, 0, r["std_detrended"], color=color, lw=1.2, alpha=0.6)
            ax.scatter(x, r["std_detrended"], color=color, s=42, zorder=3,
                       edgecolors="white", linewidths=0.5)
            ax.scatter(x, r["std_raw"], facecolors="none", edgecolors=color,
                       s=42, zorder=3, linewidths=1.2)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([str(r["run_id"]) for r in order], fontsize=7)
        ax.set_xlim(-0.7, len(order) - 0.3)
        ax.set_ylabel("std of cooperation\nacross gens (pp)", fontsize=9)
        ax.set_title(f"{model}  — detrended std spread "
                     f"{summ['std_detrended']['min']:.1f}-{summ['std_detrended']['max']:.1f} pp "
                     f"(x{summ['std_detrended']['max_over_min']:.1f}), "
                     f"Brown-Forsythe p={summ['brown_forsythe_detrended_p']:.3f}",
                     fontsize=9.5, loc="left")
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color="#555555", label="detrended std"),
        plt.Line2D([], [], marker="o", ls="", markerfacecolor="none",
                   color="#555555", label="raw std"),
    ]
    axes[0].legend(handles=handles, fontsize=8, frameon=False, loc="upper left")
    axes[-1].set_xlabel("run (columns sorted by detrended std; label = run number)",
                        fontsize=10)
    fig.suptitle(f"{exp} — how volatile is each run, and do runs differ in volatility?",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Variability of within-run variance.")
    ap.add_argument("--sim_root", default="simulations")
    ap.add_argument("--experiments", nargs="+",
                    default=["neutral_baseline", "neutral_claude", "neutral_gemini",
                             "neutral_gpt", "neutral_deepseek"])
    ap.add_argument("--models", nargs="+", default=["claude", "gpt", "gemini"])
    ap.add_argument("--output_dir", default="figures/run_variation")
    args = ap.parse_args()

    all_stats = {}
    print(f"{'experiment/model':30s} {'med det-std':>11s} {'range (pp)':>12s} "
          f"{'max/min':>8s} {'BF p(det)':>10s} {'BF p(raw)':>10s} {'rho(mean,std)':>14s}")
    for exp in args.experiments:
        panels = []
        for model in args.models:
            sim_dir = os.path.join(args.sim_root, exp, model)
            files = sorted_run_files(sim_dir)
            if not files:
                continue
            runs = run_metrics(files)
            if len(runs) < 3:
                continue
            summ = summarize(runs)
            summ["per_run"] = [{k: r[k] for k in
                                ("run_id", "mean", "std_raw", "std_detrended", "slope")}
                               for r in runs]
            all_stats[f"{exp}/{model}"] = summ
            panels.append((model, runs, summ))
            sd = summ["std_detrended"]
            print(f"{exp + '/' + model:30s} {sd['median']:11.2f} "
                  f"{sd['min']:5.2f}-{sd['max']:5.2f} {sd['max_over_min']:8.1f} "
                  f"{summ['brown_forsythe_detrended_p']:10.4f} "
                  f"{summ['brown_forsythe_raw_p']:10.4f} "
                  f"{summ['spearman_mean_vs_std']['rho']:14.2f}")
        if panels:
            out_path = os.path.join(args.output_dir, exp, "run_variance_models.png")
            plot_experiment(exp, panels, out_path)

    stats_path = os.path.join(args.output_dir, "run_variance_stats.json")
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved stats: {stats_path}")


if __name__ == "__main__":
    main()
