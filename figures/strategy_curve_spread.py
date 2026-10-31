#!/usr/bin/env python3
"""Within-run strategy spread from coded response curves.

For each run and generation, take the coded response curves of its agents
(same curves as Figure 1) and compute the mean pairwise area between them:

    D_r = 1/C(n_r, 2) * sum_{i<j} (1/100) * integral_0^100 |f_i(x) - f_j(x)| dx

i.e. the average vertical gap, in percentage points, between two agents' curves
in the same run. n_r counts agents (identical copies included), restricted to
those whose strategy could be coded. 0 pp means every agent plays the same
strategy. Generation 1 comes before any evolution, so it pools both rules' runs.

Usage: python figures/strategy_curve_spread.py
"""
import glob
import hashlib
import json
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_gen0_variants import eval_strategy
from compute_auc import X_VALUES

CONDITIONS = {"Cultural evolution": "neutral_baseline",
              "Direct evolution (beta=50)": "direct_evo_50_neutral"}
# Direct evolution copies strategies verbatim, so every Generation-10 text is one
# of its run's Generation-1 texts. Those were coded once with
#   python figures/code_gen0_strategies.py submit|collect \
#       --sim_dir simulations/direct_evo_50_neutral --out_dir figures/reactivity_plots/direct_evo_50_neutral
# and are looked up here through that cache. direct_evo_50_neutral uses the same
# neutral baseline prompt as the cultural runs (direct_evo_50_old used an older
# prompt; direct_evo_50_new used newer models).
DIRECT_SIM = "simulations/direct_evo_50_neutral"
DIRECT_CACHE = "figures/reactivity_plots/direct_evo_50_neutral/_strategy_code_cache.json"
MODELS = ["claude", "gpt", "gemini"]
_trapz = getattr(np, "trapezoid", None) or np.trapz


def spread(curves):
    """Mean over agent pairs of the area between their curves, / 100 -> pp."""
    return float(np.mean([_trapz(np.abs(a - b), X_VALUES) / 100
                          for a, b in combinations(curves, 2)]))


def run_curves(path, generation):
    recs = json.load(open(path)).get("agent_records", [])
    by_run = {}
    for r in recs:
        if r.get("generation") != generation or not r.get("is_specified") or not r.get("strategy_code"):
            continue
        ys = eval_strategy(r["strategy_code"])
        if ys is not None:
            by_run.setdefault(r["run"], []).append(ys)
    return [c for c in by_run.values() if len(c) >= 2]


def direct_run_curves(model, generation):
    cache = json.load(open(DIRECT_CACHE))
    out, missing = [], 0
    for f in sorted(glob.glob(f"{DIRECT_SIM}/{model}/Final_Accum*.json")):
        recs = json.load(open(f))["agents_data"]
        texts = {r["agent_name"]: r["strategy"] for r in recs if r["current_generation"] == generation}
        curves = []
        for t in texts.values():
            e = cache.get(hashlib.sha1(t.encode("utf-8")).hexdigest())
            if e is None:
                missing += 1
                continue
            if e["is_specified"] and e["strategy_code"]:
                ys = eval_strategy(e["strategy_code"])
                if ys is not None:
                    curves.append(ys)
        if len(curves) >= 2:
            out.append(curves)
    if missing:
        print(f"  [{model} g{generation}] {missing} strategies not in cache")
    return out


def main():
    out = {}

    def record(label, m, g, runs):
        s = [spread(c) for c in runs]
        sd = float(np.std(s, ddof=1))  # across runs
        se = sd / np.sqrt(len(s))      # standard error of the mean (table error bars)
        out[f"{label}|{m}|gen{g}"] = {"mean_pp": float(np.mean(s)), "sd_pp": sd, "se_pp": se, "n_runs": len(s)}
        print(f"{label:28s} {m:7s} gen{g:<3d} spread {np.mean(s):5.1f} ± {se:4.1f} pp SE  (SD {sd:4.1f}, n={len(s)} runs)")

    for m in MODELS:
        cells = {}
        for label, variant in CONDITIONS.items():
            p = f"figures/reactivity_plots/{variant}/{m}/strategy_analysis_{m}.json"
            for g in (1, 10):
                cells[label, g] = direct_run_curves(m, g) if variant == DIRECT_SIM.split("/")[-1] else run_curves(p, g)
                record(label, m, g, cells[label, g])
        record("Generation 1 (both rules)", m, 1, sum((cells[label, 1] for label in CONDITIONS), []))
    json.dump(out, open("figures/strategy_curve_spread.json", "w"), indent=1)


if __name__ == "__main__":
    main()
