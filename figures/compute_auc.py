#!/usr/bin/env python3
"""
Compute average area under the coded-strategy curve for each model,
broken down by generation.

For each agent with a fully specified strategy, evaluates strategy_fn over
pair_prev_pct in [0, 100] and computes the area using the trapezoidal rule.
The AUC is normalised to [0, 1] by dividing by 100*100 (max possible area).

Reads strategy_analysis_{model}.json from --analysis_dir/{model}/.
Saves results to --output (default: figures/auc_stats.json).

Usage:
    python figures/compute_auc.py \
        [--analysis_dir figures/reactivity_plots/gen1] \
        [--output figures/auc_stats.json]
"""

import argparse
import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt

MODELS = ["claude", "gpt", "gemini"]

X_VALUES = np.linspace(0, 100, 500)   # pair_prev_pct grid for integration

# ---------------------------------------------------------------------------
# Unicode → ASCII so exec() doesn't choke on LLM-generated code
# ---------------------------------------------------------------------------
_UNICODE_OPS = [
    ('\u2265', '>='), ('\u2264', '<='), ('\u2260', '!='),
    ('\u00d7', '*'),  ('\u00f7', '/'),  ('\u2212', '-'),
]

def _sanitize(code):
    for uni, asc in _UNICODE_OPS:
        code = code.replace(uni, asc)
    return code

def _fix_empty_blocks(code):
    lines = code.splitlines(keepends=True)
    result = []
    for i, line in enumerate(lines):
        result.append(line)
        code_part = line.split('#')[0].rstrip()
        if not code_part.endswith(':'):
            continue
        indent = len(line) - len(line.lstrip())
        j = i + 1
        while j < len(lines) and (not lines[j].strip() or lines[j].strip().startswith('#')):
            j += 1
        if j >= len(lines) or (len(lines[j]) - len(lines[j].lstrip())) <= indent:
            result.append(' ' * (indent + 4) + 'pass\n')
    return ''.join(result)

def eval_strategy(code):
    """Return y-values over X_VALUES, or None if the code can't be evaluated."""
    ns = {}
    sanitized = _sanitize(code)
    for attempt in [sanitized, _fix_empty_blocks(sanitized)]:
        try:
            exec(attempt, ns)
            break
        except SyntaxError:
            continue
        except Exception:
            return None
    fn = ns.get('strategy_fn')
    if fn is None:
        return None
    try:
        ys = [float(np.clip(fn(x), 0, 100)) for x in X_VALUES]
        return np.array(ys)
    except Exception:
        return None

def auc_normalised(ys):
    """Trapezoidal AUC over [0,100], normalised to [0,1]."""
    raw = np.trapz(ys, X_VALUES)   # units: pct * pct
    return raw / (100.0 * 100.0)   # normalise by max area

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute average AUC of coded strategies per model and generation.")
    parser.add_argument("--analysis_dir", default="figures/reactivity_plots/gen1",
                        help="Directory containing {model}/strategy_analysis_{model}.json")
    parser.add_argument("--output", default="figures/auc_stats.json",
                        help="Path to write the output JSON")
    args = parser.parse_args()

    results = {}

    for model in MODELS:
        path = os.path.join(args.analysis_dir, model, f"strategy_analysis_{model}.json")
        if not os.path.exists(path):
            print(f"  [{model}] not found: {path} — skipping")
            continue

        with open(path) as f:
            data = json.load(f)

        records = data.get("agent_records", [])
        print(f"\n[{model}] {len(records)} records")

        # Collect AUC per generation
        by_gen = {}       # gen -> list of AUC values
        skipped = 0

        for rec in records:
            if not rec.get("is_specified", False):
                continue
            code = rec.get("strategy_code")
            if not code:
                continue
            ys = eval_strategy(code)
            if ys is None:
                skipped += 1
                continue
            gen = rec["generation"]
            by_gen.setdefault(gen, []).append(auc_normalised(ys))

        print(f"  skipped (eval error): {skipped}")

        gen_stats = {}
        for gen in sorted(by_gen):
            vals = by_gen[gen]
            gen_stats[gen] = {
                "n":      len(vals),
                "mean":   float(np.mean(vals)),
                "std":    float(np.std(vals)),
                "median": float(np.median(vals)),
                "min":    float(np.min(vals)),
                "max":    float(np.max(vals)),
            }
            print(f"  gen {gen:2d}: n={len(vals):3d}  mean={np.mean(vals):.4f}  "
                  f"std={np.std(vals):.4f}  median={np.median(vals):.4f}")

        # Overall (across all generations)
        all_vals = [v for vs in by_gen.values() for v in vs]
        overall = {
            "n":      len(all_vals),
            "mean":   float(np.mean(all_vals)),
            "std":    float(np.std(all_vals)),
            "median": float(np.median(all_vals)),
            "min":    float(np.min(all_vals)),
            "max":    float(np.max(all_vals)),
        }
        print(f"  overall: n={len(all_vals)}  mean={np.mean(all_vals):.4f}")

        results[model] = {
            "overall":     overall,
            "by_generation": gen_stats,
        }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.output}")

    plot_auc_by_generation(results, args.output)


def plot_auc_by_generation(results, json_output_path):
    MODEL_LABELS = {"claude": "Claude 3.5 Sonnet", "gpt": "GPT-4o", "gemini": "Gemini 2.5 Flash"}
    MODEL_COLORS = {"claude": "#FF6B35", "gpt": "#10A37F", "gemini": "#4285F4"}

    models   = [m for m in MODELS if m in results]
    all_gens = sorted({int(g) for m in models for g in results[m]["by_generation"]})

    n_models = len(models)
    bar_w    = 0.7 / n_models
    offsets  = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_w

    fig, ax = plt.subplots(figsize=(13, 5))

    for m_idx, model in enumerate(models):
        gen_stats = results[model]["by_generation"]
        xs, means, sems = [], [], []
        for gen in all_gens:
            s = gen_stats.get(str(gen)) or gen_stats.get(gen)
            if s is None:
                continue
            xs.append(gen + offsets[m_idx])
            means.append(s["mean"])
            sems.append(s["std"] / (s["n"] ** 0.5))

        ax.bar(xs, means, width=bar_w * 0.9,
               color=MODEL_COLORS[model], label=MODEL_LABELS[model],
               alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.errorbar(xs, means, yerr=sems,
                    fmt="none", color="#333333", capsize=3, linewidth=1)

    ax.set_xticks(all_gens)
    ax.set_xticklabels([f"Gen {g}" for g in all_gens], fontsize=9)
    ax.set_ylabel("Mean AUC (normalised, 0–1)", fontsize=10)
    ax.set_title("Average Area Under Coded-Strategy Curve by Generation", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    plot_path = os.path.splitext(json_output_path)[0] + "_by_generation.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
