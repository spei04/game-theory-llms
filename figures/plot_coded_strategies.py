#!/usr/bin/env python3
"""
Summarize coded strategies for fully specified agents across models.

For each model (claude, gpt, gemini), reads strategy_analysis.json,
filters to a given generation, collects all fully specified strategy
functions, evaluates them at x=0..100, and plots mean ± std per model.

Usage:
    python figures/plot_coded_strategies.py \
        [--base_dir figures/reactivity_plots/gen1] \
        [--generation 1] \
        [--output_dir figures/coded_strats]
"""

import json
import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt


MODEL_DIRS = {
    "Claude 3.5 Sonnet": "claude",
    "GPT-4o":            "gpt",
    "Gemini 2.5 Flash":  "gemini",
}

MODEL_COLORS = {
    "Claude 3.5 Sonnet": "#FF6B35",
    "GPT-4o":            "#10A37F",
    "Gemini 2.5 Flash":  "#4285F4",
}


_UNICODE_OPS = [
    ('\u2265', '>='),  # ≥
    ('\u2264', '<='),  # ≤
    ('\u2260', '!='),  # ≠
    ('\u00d7', '*'),   # ×
    ('\u00f7', '/'),   # ÷
    ('\u2212', '-'),   # − (minus sign)
]

def _sanitize_code(code):
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


def eval_strategy(code, x_values):
    """Evaluate strategy_fn(x) for each x. Returns list or None."""
    namespace = {}
    sanitized = _sanitize_code(code)
    try:
        exec(sanitized, namespace)
    except SyntaxError:
        try:
            exec(_fix_empty_blocks(sanitized), namespace)
        except SyntaxError:
            return None  # unfixable syntax in LLM-generated code
        except Exception as e:
            print(f"  Error evaluating strategy code: {e}")
            return None
    except Exception as e:
        print(f"  Error evaluating strategy code: {e}")
        return None
    fn = namespace.get('strategy_fn')
    if fn is None:
        return None
    try:
        return [float(np.clip(fn(x), 0, 100)) for x in x_values]
    except Exception as e:
        print(f"  Error evaluating strategy code: {e}")
        return None


def load_records(json_path, generation):
    """Return agent records for the given generation that are fully specified and have code."""
    with open(json_path) as f:
        data = json.load(f)
    records = [
        r for r in data.get("agent_records", [])
        if r.get("generation") == generation
        and r.get("is_specified", False)
        and r.get("strategy_code")
    ]
    return records


def collect_strategy_curves(records, x_values):
    """
    Evaluate each record's strategy_fn at x_values.
    Returns list of y-arrays (one per record that evaluates successfully).
    """
    curves = []
    seen_codes = set()
    for r in records:
        code = r["strategy_code"]
        # Deduplicate identical code strings within a model
        if code in seen_codes:
            # Still include — different agents/runs may have the same strategy
            pass
        seen_codes.add(code)
        ys = eval_strategy(code, x_values)
        if ys is not None:
            curves.append(ys)
    return curves


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean±std of coded strategies per model for a given generation.")
    parser.add_argument("--base_dir", default="figures/reactivity_plots/gen1",
                        help="Directory containing claude/, gpt/, gemini/ subdirs")
    parser.add_argument("--generation", type=int, default=1,
                        help="Generation to plot (default: 1)")
    parser.add_argument("--output_dir", default="figures/coded_strats",
                        help="Where to save the output plot")
    args = parser.parse_args()

    x_values = np.linspace(0, 100, 300)

    fig, ax = plt.subplots(figsize=(9, 6))

    any_plotted = False
    for model_name, subdir in MODEL_DIRS.items():
        json_path = os.path.join(args.base_dir, subdir, f"strategy_analysis_{subdir}.json")
        if not os.path.exists(json_path):
            print(f"  [{model_name}] Not found: {json_path} — skipping.")
            continue

        records = load_records(json_path, args.generation)
        if not records:
            print(f"  [{model_name}] No fully specified records with code for generation {args.generation}.")
            continue

        curves = collect_strategy_curves(records, x_values)
        if not curves:
            print(f"  [{model_name}] No curves evaluated successfully.")
            continue

        curves_arr = np.array(curves)  # shape (n_agents, n_x)
        mean_y = np.mean(curves_arr, axis=0)
        std_y  = np.std(curves_arr,  axis=0)
        n      = len(curves)

        color = MODEL_COLORS[model_name]
        ax.plot(x_values, mean_y, color=color, linewidth=2.5,
                label=f"{model_name} (n={n})")
        ax.fill_between(x_values,
                        np.clip(mean_y - std_y, 0, 100),
                        np.clip(mean_y + std_y, 0, 100),
                        alpha=0.18, color=color)

        print(f"  [{model_name}] {n} curves — mean donation at x=50: {mean_y[150]:.1f}%")
        any_plotted = True

    if not any_plotted:
        print("No data to plot. Make sure strategy_analysis.json files exist and contain strategy_code.")
        return

    ax.set_xlabel("Pair's donation % in previous round  (% of 10 units)", fontsize=12)
    ax.set_ylabel("My donation %  (% of 10 units)", fontsize=12)
    ax.set_title(f"Coded Strategy Summary — Generation {args.generation}", fontsize=13)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"gen{args.generation}_coded_strategies.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
