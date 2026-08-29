#!/usr/bin/env python3
"""
Compare gen-1 (first-generation) coded-strategy self-cooperation AUC between
the original April setup and the current pipeline, per model.

Original setup batches (baseline prompt, claude-3-5-sonnet / gpt-4o /
gemini-1.5-flash, 12 agents):
  0409  figures/reactivity_plots/0409   (runs 2026-04-09, 5 runs/model)
  0417  figures/reactivity_plots/0417   (runs 2026-04-16, 5 runs/model)
  0424  figures/reactivity_plots/gen1   (runs 2026-04-23/24, 5 runs/model)

Current setup (baseline prompt, same models):
  neutral_baseline      figures/reactivity_plots/neutral_baseline
                        (full runs, 2026-05-07, 20 runs/model)
  neutral_baseline_200  figures/reactivity_plots/neutral_baseline_200
                        (strategies-only runs, 2026-07-16, ~200 runs/model)

AUC = trapezoidal area under strategy_fn(pair_prev_pct) over [0, 100],
normalised to [0, 1] (see compute_auc.py). Gen-1 records with
is_specified=True only.

Usage:
    python figures/compare_original_vs_current_auc.py
"""

import json
import os
import sys

import numpy as np
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_auc import eval_strategy, auc_normalised

MODELS = ["claude", "gpt", "gemini"]

ORIGINAL_BATCHES = {
    "0409": "figures/reactivity_plots/0409",
    "0417": "figures/reactivity_plots/0417",
    "0424": "figures/reactivity_plots/gen1",
}
CURRENT_BATCHES = {
    "neutral_baseline (May, full runs)": "figures/reactivity_plots/neutral_baseline",
    "neutral_baseline_200 (Jul, strategies-only)": "figures/reactivity_plots/neutral_baseline_200",
}


def load_gen1_aucs(analysis_dir, model):
    """Return list of AUC values for gen-1 specified strategies of one model."""
    for name in (f"strategy_analysis_{model}.json", "strategy_analysis.json"):
        path = os.path.join(analysis_dir, model, name)
        if os.path.exists(path):
            break
    else:
        return None
    with open(path) as f:
        records = json.load(f)["agent_records"]
    aucs = []
    for rec in records:
        if rec.get("generation") != 1 or not rec.get("is_specified"):
            continue
        code = rec.get("strategy_code")
        if not code:
            continue
        ys = eval_strategy(code)
        if ys is None:
            continue
        aucs.append(auc_normalised(ys))
    return np.array(aucs)


def describe(vals):
    return (f"n={len(vals):4d}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
            f"median={np.median(vals):.4f}")


def main():
    results = {}
    for model in MODELS:
        print(f"\n{'='*74}\n{model.upper()}\n{'='*74}")
        orig_parts = {}
        for label, d in ORIGINAL_BATCHES.items():
            vals = load_gen1_aucs(d, model)
            if vals is None or len(vals) == 0:
                print(f"  original {label:<12}: no data")
                continue
            orig_parts[label] = vals
            print(f"  original {label:<12}: {describe(vals)}")
        orig = np.concatenate(list(orig_parts.values()))
        print(f"  original POOLED       : {describe(orig)}")

        cur_sets = {}
        for label, d in CURRENT_BATCHES.items():
            vals = load_gen1_aucs(d, model)
            if vals is None or len(vals) == 0:
                print(f"  current  {label}: no data")
                continue
            cur_sets[label] = vals
            print(f"  current  {label:<43}: {describe(vals)}")

        model_res = {"original_pooled": {"n": int(len(orig)),
                                         "mean": float(np.mean(orig)),
                                         "std": float(np.std(orig))},
                     "original_batches": {k: {"n": int(len(v)),
                                              "mean": float(np.mean(v)),
                                              "std": float(np.std(v))}
                                          for k, v in orig_parts.items()},
                     "comparisons": {}}

        # Across-batch heterogeneity within the original setup (context for
        # judging whether original-vs-current gaps exceed batch noise).
        if len(orig_parts) > 1:
            f_stat, f_p = sps.f_oneway(*orig_parts.values())
            kw_stat, kw_p = sps.kruskal(*orig_parts.values())
            print(f"  [original batches heterogeneity] ANOVA p={f_p:.4f}  "
                  f"Kruskal p={kw_p:.4f}")
            model_res["original_batch_anova_p"] = float(f_p)

        for label, cur in cur_sets.items():
            t, t_p = sps.ttest_ind(orig, cur, equal_var=False)
            u, u_p = sps.mannwhitneyu(orig, cur, alternative="two-sided")
            pooled_sd = np.sqrt((np.var(orig, ddof=1) + np.var(cur, ddof=1)) / 2)
            d_eff = (np.mean(cur) - np.mean(orig)) / pooled_sd if pooled_sd > 0 else 0.0
            print(f"  original pooled vs {label}:")
            print(f"      diff={np.mean(cur)-np.mean(orig):+.4f}  Welch t p={t_p:.4f}  "
                  f"Mann-Whitney p={u_p:.4f}  Cohen's d={d_eff:+.3f}")
            model_res["comparisons"][label] = {
                "n": int(len(cur)), "mean": float(np.mean(cur)),
                "std": float(np.std(cur)),
                "diff_vs_original": float(np.mean(cur) - np.mean(orig)),
                "welch_p": float(t_p), "mannwhitney_p": float(u_p),
                "cohens_d": float(d_eff),
            }
        results[model] = model_res

    out = "figures/original_vs_current_gen1_auc.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
