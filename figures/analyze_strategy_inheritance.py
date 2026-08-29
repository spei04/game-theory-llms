#!/usr/bin/env python3
"""
Analyze strategy inheritance patterns from a simulation run.

Reads all_strategies_<model>_<timestamp>.json produced by donor_game_baseline.py.

Outputs:
  1. Most commonly picked (inherited) strategies across all generations
  2. How often rank-1 was picked vs lower ranks
  3. Histogram of inherited strategy ranks saved to --output_dir

Usage:
    python figures/analyze_strategy_inheritance.py <strategies_json> \
        [--output_dir figures/strategy_inheritance]
"""

import json
import argparse
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def load_data(json_path):
    with open(json_path) as f:
        return json.load(f)


def extract_new_agents(all_strategies_by_generation):
    """
    Return list of agent dicts for generation 2+ new agents only
    (those with inherited_from set). Survivors carry inherited_from=None.
    """
    new_agents = []
    for gen_strategies in all_strategies_by_generation:
        for agent in gen_strategies:
            if agent.get("inherited_from"):
                new_agents.append(agent)
    return new_agents


def print_most_common_picked(new_agents, top_n=10):
    """Print the most frequently inherited strategy texts."""
    counter = Counter(a["inherited_strategy"] for a in new_agents if a.get("inherited_strategy"))
    print(f"\n{'='*60}")
    print(f"Most commonly inherited strategies (top {top_n})")
    print(f"{'='*60}")
    for i, (strategy, count) in enumerate(counter.most_common(top_n), 1):
        print(f"\n#{i} — picked {count}x")
        print(f"  {strategy}")


def print_rank_summary(new_agents):
    """Print summary of how often each rank was picked."""
    ranks = [a["inherited_rank"] for a in new_agents if a.get("inherited_rank")]
    if not ranks:
        print("\nNo inherited_rank data found (requires new-format JSON from updated simulation).")
        return

    total = len(ranks)
    rank_counter = Counter(ranks)
    max_rank = max(ranks)

    print(f"\n{'='*60}")
    print(f"Inherited strategy rank distribution  (n={total} new agents)")
    print(f"{'='*60}")
    for rank in range(1, max_rank + 1):
        count = rank_counter.get(rank, 0)
        pct = count / total * 100
        bar = '█' * int(pct / 2)
        print(f"  Rank {rank:2d}: {count:3d}  ({pct:5.1f}%)  {bar}")

    rank1_count = rank_counter.get(1, 0)
    print(f"\nRank 1 (top scorer) picked: {rank1_count}/{total} = {rank1_count/total*100:.1f}%")
    print(f"Mean rank picked: {np.mean(ranks):.2f}  |  Median: {np.median(ranks):.1f}")


def plot_rank_histogram(new_agents, output_dir, label):
    """Plot histogram of inherited strategy ranks."""
    ranks = [a["inherited_rank"] for a in new_agents if a.get("inherited_rank")]
    if not ranks:
        print("No rank data to plot.")
        return

    max_rank = max(ranks)
    bins = list(range(1, max_rank + 2))  # [1, 2, ..., max_rank+1]
    rank_counter = Counter(ranks)
    counts = [rank_counter.get(r, 0) for r in range(1, max_rank + 1)]
    x = list(range(1, max_rank + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(x, counts, width=0.7, color='steelblue', edgecolor='white', linewidth=0.8)

    # Highlight rank 1
    bars[0].set_color('#e07b39')

    ax.set_xlabel("Rank of inherited strategy (1 = best scorer in previous generation)", fontsize=11)
    ax.set_ylabel("Number of times picked", fontsize=11)
    ax.set_title(f"Inherited Strategy Rank Distribution\n{label}", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in x])
    ax.grid(axis='y', alpha=0.3)

    # Annotate counts
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    str(count), ha='center', va='bottom', fontsize=9)

    total = len(ranks)
    rank1_pct = rank_counter.get(1, 0) / total * 100
    ax.text(0.98, 0.97,
            f"n = {total} new agents\nRank 1 picked: {rank1_pct:.1f}%\nMean rank: {np.mean(ranks):.2f}",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.85, edgecolor='#cccccc'))

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"inherited_rank_histogram_{label}.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nHistogram saved: {out}")


def find_most_recent_json(folder):
    """Return the most recently modified *all_strategies_*.json in folder."""
    import glob
    pattern = os.path.join(folder, "**", "*all_strategies_*.json")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


_MODEL_KEYWORDS = {
    "claude": "claude",
    "gpt":    "gpt",
    "gemini": "gemini",
}

def find_model_jsons(folder, model):
    """Return all *all_strategies_*.json files in folder whose name contains the model keyword."""
    import glob
    keyword = _MODEL_KEYWORDS.get(model.lower(), model.lower())
    pattern = os.path.join(folder, "*all_strategies_*.json")
    matches = glob.glob(pattern)
    return [p for p in matches if keyword in os.path.basename(p).lower()]


def main():
    parser = argparse.ArgumentParser(description="Analyze strategy inheritance from simulation JSON.")
    parser.add_argument("json_path", nargs="?", default=None,
                        help="Path to all_strategies_<model>_<ts>.json. "
                             "Omit to auto-select the most recent file in --search_dir.")
    parser.add_argument("--search_dir", default="simulations/new_baseline",
                        help="Folder to search for strategies JSON files "
                             "(used when json_path is omitted, default: simulations/new_baseline)")
    parser.add_argument("--model", default=None, choices=["claude", "gpt", "gemini"],
                        help="Filter to a specific model: claude, gpt, or gemini. "
                             "Aggregates all matching files in --search_dir.")
    parser.add_argument("--output_dir", default="figures/strategy_inheritance",
                        help="Directory to save plots (default: figures/strategy_inheritance)")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Number of most-common strategies to print (default: 10)")
    args = parser.parse_args()

    # --- Model-filtered mode: aggregate all files for the given model ---
    if args.model:
        paths = find_model_jsons(args.search_dir, args.model)
        if not paths:
            print(f"No all_strategies_*.json files found for model '{args.model}' in {args.search_dir}")
            return
        print(f"Found {len(paths)} file(s) for model '{args.model}':")
        for p in sorted(paths):
            print(f"  {p}")

        all_new_agents = []
        total_agents = 0
        total_gens = 0
        for path in paths:
            data = load_data(path)
            all_new_agents.extend(extract_new_agents(data))
            total_agents += sum(len(g) for g in data)
            total_gens += len(data)

        label = args.model
        print(f"\nLoaded: {total_gens} generation-blocks, {total_agents} total agent records")
        print(f"New agents (with unmodified inheritance): {len(all_new_agents)}")

        print_most_common_picked(all_new_agents, top_n=args.top_n)
        print_rank_summary(all_new_agents)
        plot_rank_histogram(all_new_agents, args.output_dir, label)
        return

    # --- Single-file mode ---
    json_path = args.json_path
    if json_path is None:
        json_path = find_most_recent_json(args.search_dir)
        if json_path is None:
            print(f"No all_strategies_*.json found in {args.search_dir}")
            return
        print(f"Auto-selected: {json_path}")

    data = load_data(json_path)
    label = os.path.splitext(os.path.basename(json_path))[0]

    new_agents = extract_new_agents(data)
    total_agents = sum(len(g) for g in data)
    print(f"\nLoaded: {len(data)} generations, {total_agents} total agent records")
    print(f"New agents (with unmodified inheritance): {len(new_agents)}")

    print_most_common_picked(new_agents, top_n=args.top_n)
    print_rank_summary(new_agents)
    plot_rank_histogram(new_agents, args.output_dir, label)


if __name__ == "__main__":
    main()
