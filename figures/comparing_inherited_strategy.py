#!/usr/bin/env python3
"""
Compare a new agent's coded strategy vs the rank-1 strategy from the previous generation.

For each new agent in a given generation that:
  1. Declared an inspiration via "Inherited from: [agent_name]"
  2. Has a fully specified coded strategy in strategy_analysis.json
  3. The rank-1 agent from the previous generation also has a fully specified coded strategy

Produces one plot per qualifying agent showing both coded strategies on the
same axes, saved to figures/strategy_inherit_comparison/.

Usage:
    python figures/comparing_inherited_strategy.py \
        --generation 2 \
        --model claude \
        [--sim_dir simulations/new_baseline] \
        [--analysis_dir figures/reactivity_plots/gen1] \
        [--output_dir figures/strategy_inherit_comparison]

--model accepts the subdirectory name: claude, gpt, or gemini.
--analysis_dir should contain a {model}/strategy_analysis.json file.
"""

import json
import os
import re
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt


# Maps CLI model name → filename fragment used in all_strategies_*.json
MODEL_FILE_PATTERNS = {
    "claude":  "claude",
    "gpt":     "gpt",
    "gemini":  "gemini",
}

_UNICODE_OPS = [
    ('\u2265', '>='),
    ('\u2264', '<='),
    ('\u2260', '!='),
    ('\u00d7', '*'),
    ('\u00f7', '/'),
    ('\u2212', '-'),
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
            print(f"  Could not evaluate strategy code: {e}")
            return None
    except Exception as e:
        print(f"  Could not evaluate strategy code: {e}")
        return None
    fn = namespace.get('strategy_fn')
    if fn is None:
        return None
    try:
        return [float(np.clip(fn(x), 0, 100)) for x in x_values]
    except Exception as e:
        print(f"  Could not evaluate strategy code: {e}")
        return None


def load_strategy_analysis(analysis_dir, model):
    """Load strategy_analysis.json for the given model."""
    path = os.path.join(analysis_dir, model, f"strategy_analysis_{model}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"strategy_analysis_{model}.json not found at: {path}")
    with open(path) as f:
        data = json.load(f)
    # Build index: (run, generation, agent_name) -> strategy_code
    index = {}
    for record in data.get("agent_records", []):
        if record.get("strategy_code") and record.get("is_specified", False):
            key = (record["run"], record["generation"], record["agent"])
            index[key] = record["strategy_code"]
    return index


def extract_run_timestamp(filename):
    """Extract run timestamp (e.g. '20260416_195746') from an all_strategies filename."""
    m = re.search(r'_(\d{8}_\d{6})\.json$', filename)
    return m.group(1) if m else None


def find_sim_files(sim_dir, model):
    pattern = os.path.join(sim_dir, f"*all_strategies*{model}*.json")
    return sorted(glob.glob(pattern))


def make_safe_name(s):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', s)


def plot_comparison(agent_ys, top_ys, x_values,
                    new_agent, inspired_by,
                    top_agent, generation, run, output_dir, model):
    _, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x_values, agent_ys, color="#4C72B0", linewidth=2.5,
            label=f"Agent {new_agent} (inspired by {inspired_by})")
    ax.plot(x_values, top_ys, color="#DD8452", linewidth=2.5, linestyle="--",
            label=f"Rank-1 strategy (agent {top_agent}, gen {generation - 1})")

    ax.set_xlabel("Pair's donation % in previous round  (% of 10 units)", fontsize=12)
    ax.set_ylabel("My donation %  (% of 10 units)", fontsize=12)
    ax.set_title(
        f"Agent {new_agent} — Gen {generation} — {model}\n"
        f"Inspired by {inspired_by} vs rank-1 agent {top_agent}",
        fontsize=12
    )
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{make_safe_name(model)}_gen{generation}_agent{make_safe_name(new_agent)}_run{run}.png"
    out = os.path.join(output_dir, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot unmodified-inherited vs rank-1 coded strategies per qualifying agent.")
    parser.add_argument("--generation", type=int, required=True,
                        help="Generation whose new agents to inspect")
    parser.add_argument("--model", required=True, choices=list(MODEL_FILE_PATTERNS),
                        help="Model subdirectory: claude, gpt, or gemini")
    parser.add_argument("--sim_dir", default="simulations/new_baseline",
                        help="Directory containing all_strategies_*.json files")
    parser.add_argument("--analysis_dir", default="figures/reactivity_plots/gen1",
                        help="Directory containing {model}/strategy_analysis.json")
    parser.add_argument("--output_dir", default="figures/strategy_inherit_comparison",
                        help="Base output directory; plots are saved under {output_dir}/{model}/")
    args = parser.parse_args()

    # Load coded strategy index
    try:
        code_index = load_strategy_analysis(args.analysis_dir, args.model)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    print(f"Loaded {len(code_index)} coded strategy records for '{args.model}'.")

    sim_files = find_sim_files(args.sim_dir, args.model)
    if not sim_files:
        print(f"No all_strategies files found for model '{args.model}' in {args.sim_dir}")
        return
    sim_files = sim_files[:1]
    print(f"Using simulation file: {os.path.basename(sim_files[0])}")

    x_values = np.linspace(0, 100, 300)
    total_plots = 0

    for sim_file in sim_files:
        run = extract_run_timestamp(sim_file)
        if run is None:
            print(f"  Skipping (no timestamp in filename): {sim_file}")
            continue

        with open(sim_file) as f:
            all_strategies = json.load(f)

        # Find the target generation's agent list
        gen_agents = None
        for gen_list in all_strategies:
            if gen_list and gen_list[0].get("generation") == args.generation:
                gen_agents = gen_list
                break
        if gen_agents is None:
            print(f"  [{run}] Generation {args.generation} not found — skipping.")
            continue

        # Build rank-1 fallback: scan gen_agents for the agent cited with inherited_rank == 1
        rank1_fallback = None
        for a in gen_agents:
            if a.get("inherited_rank") == 1 and a.get("inherited_from"):
                rank1_fallback = a["inherited_from"]
                break

        # Collect qualifying new agents:
        # - declared an inspiration via "Inherited from:" (inspired_by is set)
        # - both the agent's own coded strategy and the rank-1 agent's coded strategy
        #   are fully specified (present in strategy_analysis.json)
        for agent_rec in gen_agents:
            inspired_by = agent_rec.get("inspired_by")
            # Fallback: extract from strategy field for data generated before inspired_by was added
            if not inspired_by:
                m = re.search(r'Inherited from:\s*(\S+)', agent_rec.get("strategy", ""))
                if m:
                    inspired_by = m.group(1).strip().rstrip('.')
            top_ranked_agent = agent_rec.get("top_ranked_agent") or rank1_fallback
            agent_name = agent_rec["agent_name"]

            if not inspired_by:
                continue  # agent did not declare an inspiration
            if not top_ranked_agent:
                print(f"  [{run}] Agent {agent_name}: missing top_ranked_agent — skipping.")
                continue

            prev_gen = args.generation - 1

            # Look up the new agent's own coded strategy (current generation)
            agent_key = (run, args.generation, agent_name)
            agent_code = code_index.get(agent_key)
            if not agent_code:
                print(f"  [{run}] Agent {agent_name}: no fully specified coded strategy — skipping.")
                continue

            # Look up the rank-1 agent's coded strategy (from generation G-1)
            top_key = (run, prev_gen, top_ranked_agent)
            top_code = code_index.get(top_key)
            if not top_code:
                print(f"  [{run}] Agent {agent_name}: no coded strategy for top agent "
                      f"{top_ranked_agent} (gen {prev_gen}) — skipping.")
                continue

            agent_ys = eval_strategy(agent_code, x_values)
            top_ys = eval_strategy(top_code, x_values)

            if agent_ys is None or top_ys is None:
                print(f"  [{run}] Agent {agent_name}: strategy evaluation failed — skipping.")
                continue

            plot_comparison(
                agent_ys=agent_ys,
                top_ys=top_ys,
                x_values=x_values,
                new_agent=agent_name,
                inspired_by=inspired_by,
                top_agent=top_ranked_agent,
                generation=args.generation,
                run=run,
                output_dir=os.path.join(args.output_dir, args.model),
                model=args.model,
            )
            total_plots += 1

    print(f"\nDone. {total_plots} plot(s) saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
