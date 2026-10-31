#!/usr/bin/env python3
"""How close is a cultural-evolution newcomer's strategy to the survivors it read?

For each run and generation g >= 2 under cultural evolution, take every
newcomer (agent named "<g>_<k>") and the survivors carried over into the same
generation, which are the strategies the newcomer saw before writing its own.
The distance between two agents is the same area-between-curves gap used for
the L1 norm distance in strategy_curve_spread.py:

    d(i, j) = (1/100) * integral_0^100 |f_i(x) - f_j(x)| dx   (pp)

and each newcomer is scored by its distance to the closest survivor. Only
coded strategies are used. 0 pp means the newcomer plays exactly like some
survivor.

Output: figures/strategy_closest_survivor.json

Usage: python figures/strategy_closest_survivor.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_auc import X_VALUES, eval_strategy

ANALYSIS = "figures/reactivity_plots/neutral_baseline/{m}/strategy_analysis_{m}.json"
MODELS = ["claude", "gpt", "gemini"]
_trapz = getattr(np, "trapezoid", None) or np.trapz


def gap(a, b):
    """Area between two curves / 100 -> average gap in pp."""
    return float(_trapz(np.abs(a - b), X_VALUES) / 100)


def coded_curves(model):
    """{(run, generation, agent): curve} for every coded strategy."""
    recs = json.load(open(ANALYSIS.format(m=model)))["agent_records"]
    out = {}
    for r in recs:
        if str(r.get("is_specified")) != "True" or not r.get("strategy_code"):
            continue
        ys = eval_strategy(r["strategy_code"])
        if ys is not None:
            out[(r["run"], int(r["generation"]), r["agent"])] = ys
    return out


def main():
    out = {}
    for m in MODELS:
        curves = coded_curves(m)
        nearest = []
        for (run, g, agent), ys in curves.items():
            if g < 2 or not agent.startswith(f"{g}_"):
                continue
            survivors = [c for (r2, g2, a2), c in curves.items()
                         if r2 == run and g2 == g and not a2.startswith(f"{g}_")]
            if survivors:
                nearest.append(min(gap(ys, s) for s in survivors))
        d = np.array(nearest)
        out[m] = {"n_newcomers": len(d), "median_pp": float(np.median(d)),
                  "q25_pp": float(np.percentile(d, 25)), "q75_pp": float(np.percentile(d, 75)),
                  "mean_pp": float(d.mean()), "frac_within_2pp": float((d <= 2).mean()),
                  "frac_exact": float((d < 1e-9).mean())}
        print(f"{m:7s} newcomers {len(d):4d}  median {np.median(d):4.2f} pp  "
              f"IQR [{np.percentile(d, 25):4.2f}, {np.percentile(d, 75):4.2f}]  "
              f"within 2 pp {100 * (d <= 2).mean():4.1f}%  exact {100 * (d < 1e-9).mean():4.1f}%")
    json.dump(out, open("figures/strategy_closest_survivor.json", "w"), indent=1)


if __name__ == "__main__":
    main()
