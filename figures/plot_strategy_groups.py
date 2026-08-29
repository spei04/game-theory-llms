#!/usr/bin/env python3
"""
Classify agent strategies by shape and plot the percentage of each group
across generations, separately for two simulation batches.

is_specified is determined by:
  1. strategy_analysis.json files (authoritative when the run is covered)
  2. LLM classification for all other runs (LLM can also return "Unspecified")

LLM classifications are cached per (batch, model) in
  figures/strategy_groups/llm_cache_{batch}_{model}.json

Strategy shape categories:
  Unspecified       – no clear quantitative rule
  Unconditional     – fixed donation regardless of partner behaviour
  Direct Reciprocity   – mirrors partner % exactly (pure TfT)
  Adjusted Reciprocity – mirrors partner % ± offset/cap/floor/bonus
  Binary Threshold  – 2 discrete donation levels separated by one threshold
  Multi-level Stepwise – 3+ discrete levels / multiple thresholds
  Proportional Formula – explicit arithmetic (min/max, division, linear)
  Other             – fully specified but doesn't fit the above

One figure per model (claude, gpt, gemini).
Each figure: two subplots side-by-side (dir1 | dir2).

Usage:
    python figures/plot_strategy_groups.py \\
        [--dir1 simulations/0417] \\
        [--dir2 simulations/0424] \\
        [--analysis_dir1 figures/reactivity_plots/gen10] \\
        [--analysis_dir2 figures/reactivity_plots/gen1] \\
        [--output_dir figures/strategy_groups]
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import anthropic
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

CATEGORIES = [
    "Unspecified",
    "Unconditional",
    "Proportional Formula",
    "Multi-level Stepwise",
    "Binary Threshold",
    "Adjusted Reciprocity",
    "Direct Reciprocity",
    "Other",
]

CATEGORY_COLORS = {
    "Unspecified":          "#999999",
    "Unconditional":        "#e377c2",
    "Direct Reciprocity":   "#1f77b4",
    "Adjusted Reciprocity": "#aec7e8",
    "Binary Threshold":     "#ff7f0e",
    "Multi-level Stepwise": "#2ca02c",
    "Proportional Formula": "#9467bd",
    "Other":                "#d62728",
}

CATEGORY_DESCRIPTIONS = {
    "Unspecified":          "no clear quantitative rules — vague language, missing numbers, or garbled/incomplete text",
    "Unconditional":        "fixed donation regardless of partner behavior (e.g. always donate 50%)",
    "Direct Reciprocity":   "mirrors partner's exact donation % — pure tit-for-tat, no modifier",
    "Adjusted Reciprocity": "mirrors partner's % with a modifier: cap, floor, bonus, penalty, or fixed offset",
    "Binary Threshold":     "two discrete donation levels separated by exactly one threshold",
    "Multi-level Stepwise": "three or more discrete donation levels / multiple thresholds",
    "Proportional Formula": "explicit arithmetic formula (min/max, halving, linear e.g. X/2 or (X+N)/D)",
    "Other":                "fully specified with clear numbers but doesn't fit any category above",
}


# ---------------------------------------------------------------------------
# Strategy text cleaning
# ---------------------------------------------------------------------------

def clean_strategy(s: str) -> str:
    s = re.sub(r'^My strategy will be\s*[\*:]+\s*', '', s, flags=re.I).strip()
    s = re.sub(r'\n*\s*Inherited from:\s*\S[^\n]*', '', s, flags=re.I).strip()
    # Strip trailing punctuation/formatting artefacts (**, ***, trailing quotes, etc.)
    s = re.sub(r'[\*"\'\.]+\s*$', '', s).strip()
    return s


# ---------------------------------------------------------------------------
# Analysis index — is_specified lookup
# ---------------------------------------------------------------------------

MODEL_KEYWORDS = {
    "claude": "claude",
    "gpt":    "gpt",
    "gemini": "gemini",
}


def _ts_variants(run: str) -> list:
    """
    The Final_Accum and all_strategies files from the same simulation run are
    sometimes saved 1–2 seconds apart, giving different timestamps.
    Return the run string plus ±2-second variants so lookups are fuzzy.
    """
    from datetime import datetime, timedelta
    try:
        dt = datetime.strptime(run, "%Y%m%d_%H%M%S")
    except ValueError:
        return [run]
    variants = [run]
    for delta in [-2, -1, 1, 2]:
        variants.append((dt + timedelta(seconds=delta)).strftime("%Y%m%d_%H%M%S"))
    return variants


def load_analysis_index(analysis_dir: str) -> tuple:
    """
    Returns:
      record_index  – {(run, generation, agent_name): is_specified}
                      Keys are indexed under ALL ±2-second timestamp variants so
                      that Final_Accum and all_strategies files from the same run
                      match even when their timestamps differ by a second or two.
      unspec_texts  – set of cleaned strategy texts ever marked is_specified=False
    Reads all strategy_analysis*.json files under analysis_dir/{model}/.
    """
    record_index = {}
    unspec_texts = set()
    if not analysis_dir or not os.path.isdir(analysis_dir):
        return record_index, unspec_texts
    for model in MODEL_KEYWORDS:
        for path in sorted(glob.glob(
                os.path.join(analysis_dir, model, "strategy_analysis*.json"))):
            with open(path) as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue
            for r in data.get("agent_records", []):
                is_spec = r.get("is_specified", False)
                gen      = r["generation"]
                agent    = r["agent"]
                # Index under the canonical run AND all ±2s variants
                for ts in _ts_variants(r["run"]):
                    record_index[(ts, gen, agent)] = is_spec
                if not is_spec:
                    unspec_texts.add(clean_strategy(r.get("strategy", "")))
    return record_index, unspec_texts


# ---------------------------------------------------------------------------
# LLM classifier
# ---------------------------------------------------------------------------

def _build_classify_prompt(strategies: list) -> str:
    """strategies: list of (idx, cleaned_text).
    Only called for strategies already confirmed as fully specified by strategy_analysis.py.
    'Unspecified' is intentionally excluded — specification is determined upstream.
    """
    shape_categories = [c for c in CATEGORIES if c != "Unspecified"]
    cats = "\n".join(
        f"- {cat}: {CATEGORY_DESCRIPTIONS[cat]}"
        for cat in shape_categories
    )
    strats = "\n".join(f"{idx}. {text[:400]}" for idx, text in strategies)
    return (
        "Each strategy below has already been verified as fully specified (it contains "
        "clear numeric donation rules). Classify each one into exactly one of these "
        "shape categories:\n"
        f"{cats}\n\n"
        "Respond ONLY with a JSON object mapping each number to its category name.\n"
        'Example: {"1": "Direct Reciprocity", "2": "Binary Threshold"}\n\n'
        f"Strategies:\n{strats}"
    )


def _classify_batch(client: anthropic.Anthropic, strategies: list) -> dict:
    """Returns {idx: category_str} for the given batch."""
    prompt = _build_classify_prompt(strategies)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not m:
        return {}
    try:
        raw = json.loads(m.group())
        valid = set(CATEGORIES) - {"Unspecified"}
        return {int(k): v for k, v in raw.items() if v in valid}
    except (json.JSONDecodeError, ValueError):
        return {}


def _cache_path(output_dir: str, batch_label: str, model: str) -> str:
    return os.path.join(output_dir, f"llm_cache_{batch_label}_{model}.json")


def _load_cache(output_dir: str, batch_label: str, model: str) -> dict:
    path = _cache_path(output_dir, batch_label, model)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict, output_dir: str, batch_label: str, model: str):
    os.makedirs(output_dir, exist_ok=True)
    path = _cache_path(output_dir, batch_label, model)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)


def classify_with_llm(
    texts: list,
    batch_label: str,
    model: str,
    client: anthropic.Anthropic,
    output_dir: str,
    batch_size: int = 40,
) -> dict:
    """
    Returns {strategy_text: category} for every text in the list.
    Hits cache first; calls the LLM only for uncached entries.
    """
    cache = _load_cache(output_dir, batch_label, model)
    to_classify = [t for t in texts if t not in cache]

    if to_classify:
        unique = list(dict.fromkeys(to_classify))  # deduplicate, preserve order
        print(f"  [{model}/{batch_label}] LLM-classifying {len(unique)} unique strategies …")

        for start in range(0, len(unique), batch_size):
            chunk = unique[start:start + batch_size]
            indexed = [(i + 1, s) for i, s in enumerate(chunk)]
            result = _classify_batch(client, indexed)
            for i, s in enumerate(chunk):
                cache[s] = result.get(i + 1, "Other")

        _save_cache(cache, output_dir, batch_label, model)

    return cache


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_batch(
    sim_dir: str,
    analysis_index: tuple,
    client: anthropic.Anthropic,
    batch_label: str,
    output_dir: str,
) -> dict:
    """
    Returns {model_short: {generation: [category, ...]}}

    Specification is determined solely by strategy_analysis.py results:
    - is_specified=True  in analysis → LLM classifies into a shape category
    - is_specified=False in analysis → "Unspecified" (no LLM call)
    - not found in analysis          → "Unspecified" (no LLM call)
    """
    record_index, unspec_texts = analysis_index

    # Phase 1: collect raw records
    raw = defaultdict(lambda: defaultdict(list))
    # raw[model][gen] = list of (cleaned_text, forced_unspecified)

    for path in sorted(glob.glob(os.path.join(sim_dir, "*all_strategies*.json"))):
        if path.endswith(".txt"):
            continue
        with open(path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        bn = os.path.basename(path).lower()
        model_short = None
        for short, kw in MODEL_KEYWORDS.items():
            if kw in bn:
                model_short = short
                break
        if model_short is None:
            continue

        m = re.search(r'_(\d{8}_\d{6})\.json$', path)
        run = m.group(1) if m else None

        for gen_list in data:
            for agent in gen_list:
                gen        = agent.get("generation")
                strat      = agent.get("strategy", "")
                agent_name = agent.get("agent_name", "")
                if gen is None or not strat:
                    continue

                cleaned = clean_strategy(strat)

                # Only send to LLM if strategy_analysis explicitly confirmed is_specified=True.
                # Everything else (False or missing) → Unspecified.
                key = (run, gen, agent_name) if run is not None else None
                is_spec = record_index.get(key) if key else None
                forced_unspecified = (is_spec is not True)

                raw[model_short][gen].append((cleaned, forced_unspecified))

    # Phase 2: LLM-classify per model
    result = defaultdict(lambda: defaultdict(list))

    for model_short, gen_data in raw.items():
        # Gather texts that need LLM classification
        texts_to_classify = [
            text for gen_agents in gen_data.values()
            for text, forced in gen_agents
            if not forced
        ]

        llm_cache = classify_with_llm(
            texts_to_classify,
            batch_label=batch_label,
            model=model_short,
            client=client,
            output_dir=output_dir,
        )

        for gen in sorted(gen_data):
            for text, forced in gen_data[gen]:
                if forced:
                    result[model_short][gen].append("Unspecified")
                else:
                    result[model_short][gen].append(llm_cache.get(text, "Other"))

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def compute_fractions(gen_data: dict) -> dict:
    """gen_data: {generation: [category, ...]} → {category: {generation: pct}}"""
    fracs = defaultdict(dict)
    for gen, cats in sorted(gen_data.items()):
        total = len(cats)
        if total == 0:
            continue
        counts = defaultdict(int)
        for c in cats:
            counts[c] += 1
        for cat in CATEGORIES:
            fracs[cat][gen] = counts[cat] / total * 100.0
    return fracs


def plot_model(model: str, data_dir1: dict, data_dir2: dict, output_dir: str,
               label1: str = "dir1", label2: str = "dir2"):
    gen_data1 = data_dir1.get(model, {})
    gen_data2 = data_dir2.get(model, {})

    if not gen_data1 and not gen_data2:
        print(f"  No data for {model} — skipping.")
        return

    fracs1 = compute_fractions(gen_data1)
    fracs2 = compute_fractions(gen_data2)

    model_titles = {
        "claude": "Claude 3.5 Sonnet",
        "gpt":    "GPT-4o",
        "gemini": "Gemini 2.5 Flash",
    }
    title = model_titles.get(model, model)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(f"Strategy Shape over Generations — {title}", fontsize=13, y=1.01)

    for ax, fracs, label in zip(axes, [fracs1, fracs2], [label1, label2]):
        if not fracs:
            ax.set_title(f"{label}\n(no data)")
            ax.set_visible(False)
            continue

        all_gens = sorted({g for cat_d in fracs.values() for g in cat_d})
        x = np.array(all_gens)

        for cat in CATEGORIES:
            cat_fracs = fracs.get(cat, {})
            y = np.array([cat_fracs.get(g, 0.0) for g in all_gens])
            if y.sum() == 0:
                continue
            ax.plot(x, y,
                    color=CATEGORY_COLORS[cat],
                    linewidth=2,
                    marker='o', markersize=4,
                    label=cat)

        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Generation", fontsize=10)
        ax.set_ylabel("% of agents", fontsize=10)
        ax.set_xlim(x.min() - 0.3, x.max() + 0.3)
        ax.set_ylim(-3, 103)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"strategy_groups_{model}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot strategy shape group fractions over generations.")
    parser.add_argument("--dir1", default="simulations/0417")
    parser.add_argument("--dir2", default="simulations/0424")
    parser.add_argument("--analysis_dir1", default="figures/reactivity_plots/gen10",
                        help="strategy_analysis.json dir for dir1 (auto-detect not-specified)")
    parser.add_argument("--analysis_dir2", default="figures/reactivity_plots/gen1",
                        help="strategy_analysis.json dir for dir2")
    parser.add_argument("--output_dir", default="figures/strategy_groups")
    args = parser.parse_args()

    # Load API key from .env if not already in environment
    if not os.environ.get("ANTHROPIC_API_KEY"):
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        if os.path.exists(env_path):
            with open(env_path) as ef:
                for line in ef:
                    line = line.strip()
                    if line.startswith("ANTHROPIC_API_KEY="):
                        os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1]
                        break

    client = anthropic.Anthropic()

    label1 = os.path.basename(args.dir1.rstrip('/'))
    label2 = os.path.basename(args.dir2.rstrip('/'))

    print(f"Loading analysis index for dir1 ({args.analysis_dir1}) …")
    idx1 = load_analysis_index(args.analysis_dir1)
    print(f"  {len(idx1[0])} records indexed, {len(idx1[1])} known-unspecified texts.")

    print(f"Loading analysis index for dir2 ({args.analysis_dir2}) …")
    idx2 = load_analysis_index(args.analysis_dir2)
    print(f"  {len(idx2[0])} records indexed, {len(idx2[1])} known-unspecified texts.")

    print(f"\nClassifying dir1 ({args.dir1}) …")
    data_dir1 = load_batch(args.dir1, idx1, client, label1, args.output_dir)

    print(f"\nClassifying dir2 ({args.dir2}) …")
    data_dir2 = load_batch(args.dir2, idx2, client, label2, args.output_dir)

    print()
    for model in ["claude", "gpt", "gemini"]:
        n1 = sum(len(v) for v in data_dir1.get(model, {}).values())
        n2 = sum(len(v) for v in data_dir2.get(model, {}).values())
        print(f"  {model}: dir1={n1} agent-records, dir2={n2} agent-records")
        plot_model(model, data_dir1, data_dir2, args.output_dir,
                   label1=label1, label2=label2)

    print(f"\nDone. Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
