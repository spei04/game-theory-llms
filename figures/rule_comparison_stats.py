#!/usr/bin/env python3
"""Cultural vs direct evolution under the baseline prompt: the remaining numbers.

Everything here is read from the Final_Accum run files (plus the direct-evolution
strategy-code cache for coverage); no LLM calls.

  cooperation     Generation-10 cooperation per model and rule, mean ± SE across runs
  variance split  model vs run share of Generation-10 cooperation variance per rule
                  (same computation as plot_variance_diversity_summary.py), with a
                  bootstrap 95% interval for the difference in model share
                  (direct - cultural), resampling runs within each model
  beta sweep      that split plus the model spread (max - min of model means) and the
                  pooled within-model SD for every condition in the appendix table
  distinct texts  distinct strategy texts per run at Generation 10, mean ± SE
  coded coverage  share of direct-evolution Generation-10 strategies that could be
                  coded; the L1 norm distance (strategy_curve_spread.py) uses only these

Output: figures/rule_comparison_stats.json

Usage: python figures/rule_comparison_stats.py
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_variance_diversity_summary import MODELS, load_setting, model_run_shares
from strategy_curve_spread import DIRECT_CACHE

RULES = {"cultural": "simulations/neutral_baseline",
         "direct": "simulations/direct_evo_50_neutral"}
# Appendix table: the beta sweep used an earlier wording of the baseline prompt.
BETA_SWEEP = {"0.01": "simulations/direct_evo_001_old", "1": "simulations/direct_evo_1_old",
              "5": "simulations/direct_evo_5_old", "50": "simulations/direct_evo_50_old",
              "100": "simulations/direct_evo_100_old", "500": "simulations/direct_evo_500_old"}
N_BOOT, SEED = 5000, 0


def mean_se(x):
    x = np.asarray(x, float)
    return {"mean": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(len(x))), "n_runs": len(x)}


def split(df, col="g10"):
    """Variance split plus the absolute spreads reported next to it."""
    model, run, n = model_run_shares(df, col)
    groups = [g[col].dropna().to_numpy(float) for _, g in df.groupby("model")]
    means = [g.mean() for g in groups]
    within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    return {"model_pct": model, "run_pct": run, "n_runs": n,
            "model_spread_pp": float(max(means) - min(means)),
            "within_model_sd_pp": float(np.sqrt(within / sum(len(g) - 1 for g in groups)))}


def boot_model_share_diff(settings, col="g10"):
    """95% interval for direct - cultural model share, resampling runs within each model."""
    rng = np.random.default_rng(SEED)
    by_model = {rule: [df[df["model"] == m] for m in MODELS] for rule, df in settings.items()}
    diffs = []
    for _ in range(N_BOOT):
        share = {}
        for rule in ("cultural", "direct"):
            parts = [g.iloc[rng.integers(0, len(g), len(g))] for g in by_model[rule]]
            share[rule] = model_run_shares(pd.concat(parts), col)[0]
        diffs.append(share["direct"] - share["cultural"])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"lo95": float(lo), "hi95": float(hi), "n_boot": N_BOOT, "seed": SEED}


def gen_texts(sim_dir, model, generation):
    """One list of strategy texts per run at the given generation."""
    runs = []
    for f in sorted(glob.glob(f"{sim_dir}/{model}/Final_Accum*.json")):
        recs = json.load(open(f))["agents_data"]
        texts = {r["agent_name"]: r["strategy"] for r in recs if r["current_generation"] == generation}
        if texts:
            runs.append(list(texts.values()))
    return runs


def main():
    out = {}
    settings = {rule: load_setting(d + "/{model}") for rule, d in RULES.items()}

    print("Generation-10 cooperation (%), mean ± SE across runs")
    out["cooperation_gen10"] = {}
    for rule, df in settings.items():
        for m in MODELS:
            s = mean_se(df.loc[df["model"] == m, "g10"].dropna())
            out["cooperation_gen10"][f"{rule}|{m}"] = s
            print(f"  {rule:9s} {m:7s} {s['mean']:5.1f} ± {s['se']:3.1f}  (n={s['n_runs']})")

    print("\nVariance split, model vs run (%)")
    out["variance_split"] = {}
    for rule, df in settings.items():
        for g in ("g1", "g10"):
            s = split(df, g)
            out["variance_split"][f"{rule}|{g}"] = s
            print(f"  {rule:9s} {g:4s} model {s['model_pct']:5.1f}  run {s['run_pct']:5.1f}")
    ci = boot_model_share_diff(settings)
    out["model_share_diff_direct_minus_cultural_gen10"] = ci
    print(f"  direct - cultural model share, 95% interval [{ci['lo95']:+.1f}, {ci['hi95']:+.1f}]")

    print("\nAppendix table: Generation-10 variance sources")
    out["beta_table"] = {}
    rows = {"cultural": settings["cultural"], "direct beta=50": settings["direct"]}
    rows.update({f"earlier wording, beta={b}": load_setting(d + "/{model}") for b, d in BETA_SWEEP.items()})
    for label, df in rows.items():
        s = split(df)
        out["beta_table"][label] = s
        print(f"  {label:27s} model {s['model_pct']:5.1f}  run {s['run_pct']:5.1f}  "
              f"model spread {s['model_spread_pp']:5.1f}  within SD {s['within_model_sd_pp']:5.1f}")

    print("\nDistinct strategy texts per run at Generation 10, mean ± SE")
    out["distinct_texts_gen10"] = {}
    for rule, d in RULES.items():
        for m in MODELS:
            s = mean_se([len(set(t)) for t in gen_texts(d, m, 10)])
            out["distinct_texts_gen10"][f"{rule}|{m}"] = s
            print(f"  {rule:9s} {m:7s} {s['mean']:5.2f} ± {s['se']:4.2f}  (n={s['n_runs']})")

    print("\nCoded coverage of direct-evolution Generation-10 strategies")
    cache = json.load(open(DIRECT_CACHE))
    out["direct_coded_coverage_gen10"] = {}
    for m in MODELS:
        texts = [t for run in gen_texts(RULES["direct"], m, 10) for t in run]
        coded = 0
        for t in texts:
            e = cache.get(hashlib.sha1(t.encode("utf-8")).hexdigest())
            coded += bool(e and e["is_specified"] and e["strategy_code"])
        out["direct_coded_coverage_gen10"][m] = {"coded": coded, "total": len(texts), "pct": 100 * coded / len(texts)}
        print(f"  {m:7s} {coded}/{len(texts)} = {100 * coded / len(texts):4.1f}%")

    json.dump(out, open("figures/rule_comparison_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
