#!/usr/bin/env python3
"""
Report the fraction of text strategies retained in the text->code reactivity
pipeline (Appendix: "Text-to-code conversion").

A strategy is "retained" if the classifier judged it fully specified AND its
emitted Python code compiles and evaluates successfully (compute_auc.eval_strategy
returns non-None) — i.e. exactly the criterion described in the appendix text.

For each (prompt variant, executor model, generation) this reports:
    n_agents, n_specified, n_retained, retention %

and prints:
  - a full table (all generations) to console / CSV
  - the two-column (gen 1, gen 10) LaTeX table used in the appendix
  - summary retention stats per model, pooled across prompts and generations

Source: figures/reactivity_plots/<experiment>/<model>/strategy_analysis_<model>.json
(produced by figures/strategy_analysis.py).

Output:
  figures/run_variation/retention_by_cell.csv        (full table, all generations)
  figures/run_variation/retention_summary.json        (per-model summary stats)
  LaTeX table printed to stdout (generations 1 and 10)

Usage:
    python figures/report_retention.py \
        [--analysis_root figures/reactivity_plots] \
        [--experiments neutral_baseline neutral_claude ...] \
        [--models claude gpt gemini] \
        [--output_dir figures/run_variation]
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_auc import eval_strategy

MODEL_LABELS = {"claude": "Claude", "gpt": "GPT-4o", "gemini": "Gemini"}
EXPERIMENT_LABELS = {
    "neutral_baseline": "Baseline",
    "neutral_claude":   "Claude-phrased",
    "neutral_gpt":      "GPT-phrased",
    "neutral_gemini":   "Gemini-phrased",
    "neutral_deepseek": "DeepSeek-phrased",
}


def retention_by_generation(analysis_root, exp, model, code_cache):
    """gen -> {n_agents, n_specified, n_retained}."""
    path = os.path.join(analysis_root, exp, model, f"strategy_analysis_{model}.json")
    if not os.path.exists(path):
        return {}
    recs = json.load(open(path)).get("agent_records", [])
    by_gen = defaultdict(lambda: {"n_agents": 0, "n_specified": 0, "n_retained": 0})
    for r in recs:
        g = r["generation"]
        by_gen[g]["n_agents"] += 1
        if not r.get("is_specified"):
            continue
        by_gen[g]["n_specified"] += 1
        code = r.get("strategy_code")
        if not code:
            continue
        if code not in code_cache:
            code_cache[code] = eval_strategy(code) is not None
        if code_cache[code]:
            by_gen[g]["n_retained"] += 1
    return dict(by_gen)


def main():
    ap = argparse.ArgumentParser(description="Report text->code retention rates.")
    ap.add_argument("--analysis_root", default="figures/reactivity_plots")
    ap.add_argument("--experiments", nargs="+",
                    default=["neutral_baseline", "neutral_claude", "neutral_gpt",
                             "neutral_gemini", "neutral_deepseek"])
    ap.add_argument("--models", nargs="+", default=["claude", "gpt", "gemini"])
    ap.add_argument("--output_dir", default="figures/run_variation")
    args = ap.parse_args()

    code_cache = {}
    all_rows = []   # (exp, model, gen, n_agents, n_specified, n_retained)
    for exp in args.experiments:
        for model in args.models:
            by_gen = retention_by_generation(args.analysis_root, exp, model, code_cache)
            for g, c in sorted(by_gen.items()):
                all_rows.append((exp, model, g, c["n_agents"], c["n_specified"],
                                 c["n_retained"]))

    if not all_rows:
        print("No strategy_analysis JSONs found under", args.analysis_root)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- full CSV (all generations) ----
    csv_path = os.path.join(args.output_dir, "retention_by_cell.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "model", "generation", "n_agents",
                    "n_specified", "n_retained", "retention_pct"])
        for exp, model, g, n, spec, ret in all_rows:
            w.writerow([exp, model, g, n, spec, ret, round(100 * ret / n, 1)])
    print(f"Saved: {csv_path}")

    # ---- console: full table ----
    print(f"\n{'experiment':<18}{'model':<8}{'gen':>4}{'n':>5}"
          f"{'specified':>10}{'retained':>9}{'retain%':>9}")
    for exp, model, g, n, spec, ret in all_rows:
        print(f"{exp:<18}{model:<8}{g:>4}{n:>5}{spec:>10}{ret:>9}{100*ret/n:>8.1f}%")

    # ---- per-model summary (pooled across prompts and generations) ----
    per_model = defaultdict(list)
    for exp, model, g, n, spec, ret in all_rows:
        per_model[model].append(100 * ret / n)
    summary = {}
    print(f"\n{'model':<8}{'mean%':>8}{'min%':>7}{'max%':>7}{'n_cells':>9}")
    for model, vals in per_model.items():
        summary[model] = {"mean_pct": float(np.mean(vals)), "min_pct": float(min(vals)),
                          "max_pct": float(max(vals)), "n_cells": len(vals)}
        print(f"{model:<8}{np.mean(vals):>7.1f}%{min(vals):>6.1f}%"
              f"{max(vals):>6.1f}%{len(vals):>9d}")

    json_path = os.path.join(args.output_dir, "retention_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {json_path}")

    # ---- LaTeX table: generation 1 and generation 10 ----
    lookup = {(exp, model, g): 100 * ret / n
             for exp, model, g, n, spec, ret in all_rows}
    print("\n% --- LaTeX table (generation 1 and generation 10) ---")
    print(r"\begin{table}[H]")
    print(r"  \caption{Fraction of strategies retained (fully specified and "
          r"successfully compiled) at generation 1 and generation 10, by "
          r"prompt variant and executor model.}")
    print(r"  \label{tab:retention}")
    print(r"  \centering")
    print(r"  \small")
    print(r"  \begin{tabular}{lccc ccc}")
    print(r"    \toprule")
    print(r"     & \multicolumn{3}{c}{Generation 1} & \multicolumn{3}{c}{Generation 10} \\")
    print(r"    \cmidrule(lr){2-4} \cmidrule(lr){5-7}")
    model_cols = " & ".join(MODEL_LABELS.get(m, m) for m in args.models)
    print(rf"    Prompt variant & {model_cols} & {model_cols} \\")
    print(r"    \midrule")
    for exp in args.experiments:
        label = EXPERIMENT_LABELS.get(exp, exp)
        g1 = [f"{lookup.get((exp, m, 1), float('nan')):.1f}\\%" for m in args.models]
        g10 = [f"{lookup.get((exp, m, 10), float('nan')):.1f}\\%" for m in args.models]
        print(f"    {label:<18} & " + " & ".join(g1) + " & " + " & ".join(g10) + r" \\")
    print(r"    \bottomrule")
    print(r"  \end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
