#!/usr/bin/env python3
"""
Trace the evolutionary lineage of the rank-1 strategy in a given generation.

For a chosen run and model:
  1. Ranks agents in the target generation by total accumulated resources.
  2. Identifies the rank-1 agent and walks back the inheritance chain.
  3. At each step, diffs the inherited (parent) strategy against the new strategy
     and classifies the step as:
       - SELECTION   : strategy text unchanged from parent
       - MUTATION    : single parent, strategy text changed
       - RECOMBINATION: agent drew on multiple ancestors (LLM rewrote substantially)
  4. Saves a plot showing the lineage and a text report.

Usage:
    python figures/trace_strategy_lineage.py \\
        [--model claude] \\
        [--batch 0424] \\
        [--run_idx 0]          # 0 = first matching run \\
        [--target_gen 10] \\
        [--output_dir figures/strategy_inheritance]
"""

import argparse
import glob
import json
import os
import re
import textwrap
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ENDOWMENT = 10.0

MODEL_PATTERNS = {
    "claude": "claude",
    "gpt":    "gpt",
    "gemini": "gemini",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def find_run_files(batch_dir, model):
    """Looks in batch_dir, batch_dir/<model>/, and batch_dir/array*<model>*/."""
    candidates = [batch_dir, os.path.join(batch_dir, model)]
    candidates += sorted(glob.glob(os.path.join(batch_dir, f"array*{model}*")))
    strat_files, accum_files = [], []
    for d in candidates:
        if not os.path.isdir(d):
            continue
        strat_files += sorted(glob.glob(
            os.path.join(d, f"*all_strategies*{MODEL_PATTERNS[model]}*.json")))
        accum_files += sorted(glob.glob(
            os.path.join(d, f"*Final_Accum*{MODEL_PATTERNS[model]}*.json")))
    return strat_files, accum_files


def _ts(path):
    m = re.search(r'_(\d{8}_\d{6})\.json$', path)
    return m.group(1) if m else ""


def match_files(strat_files, accum_files):
    """Pair strat and accum files by timestamp (±2 s)."""
    from datetime import datetime, timedelta
    def to_dt(ts):
        try: return datetime.strptime(ts, "%Y%m%d_%H%M%S")
        except: return None

    pairs = []
    used = set()
    for sf in strat_files:
        sf_dt = to_dt(_ts(sf))
        best = None
        for af in accum_files:
            if af in used: continue
            af_dt = to_dt(_ts(af))
            if sf_dt and af_dt and abs((sf_dt - af_dt).total_seconds()) <= 2:
                best = af
                break
        if best:
            pairs.append((sf, best))
            used.add(best)
    return pairs


def rank_agents(accum_path, generation):
    """Return [(agent_name, total_resources)] sorted best-first for a generation."""
    with open(accum_path) as f:
        data = json.load(f)
    agents_data = data.get("agents_data", [])

    gen_rows = [r for r in agents_data if r.get("current_generation") == generation]
    # Per agent: take resources at the final round of each game, sum games
    final_res = {}
    for r in gen_rows:
        key = (r["agent_name"], r["game_number"])
        rn = r["round_number"]
        res = r.get("resources", 0) or 0
        if key not in final_res or rn > final_res[key][0]:
            final_res[key] = (rn, res)

    totals = defaultdict(float)
    for (agent, _), (_, res) in final_res.items():
        totals[agent] += res
    return sorted(totals.items(), key=lambda x: -x[1])


def load_agent_index(strat_path):
    """Return {agent_name: record_dict} keyed to the agent's BIRTH generation.

    Agent names encode birth generation as the leading number (e.g. '7_4' was
    born in generation 7).  An agent may appear in multiple generation lists as
    a survivor; we always want the record from the generation it was born in so
    that inherited_from points to its actual parent, not None (survivors carry
    forward inherited_from=None in subsequent generations).
    """
    with open(strat_path) as f:
        data = json.load(f)

    def birth_gen(name):
        m = re.match(r'^(\d+)_', name)
        return int(m.group(1)) if m else None

    index = {}
    for gen_list in data:
        for a in gen_list:
            name = a["agent_name"]
            born = birth_gen(name)
            rec_gen = a.get("generation")
            # Keep this record only if it's from the agent's birth generation,
            # or if we haven't seen this agent yet (fallback)
            if name not in index or rec_gen == born:
                index[name] = a
    return index


# ---------------------------------------------------------------------------
# Strategy text cleaning
# ---------------------------------------------------------------------------

def clean_strat(s):
    s = re.sub(r'^My strategy will be\s*[\*:]+\s*', '', s or "", flags=re.I).strip()
    s = re.sub(r'\n*\s*Inherited from:\s*\S[^\n]*', '', s, flags=re.I).strip()
    s = re.sub(r'[\*"\'.]+\s*$', '', s).strip()
    return s


# ---------------------------------------------------------------------------
# Diff / classification
# ---------------------------------------------------------------------------

def token_similarity(a, b):
    """Jaccard similarity of word sets."""
    wa = set(re.findall(r'\w+', a.lower()))
    wb = set(re.findall(r'\w+', b.lower()))
    if not wa and not wb:
        return 1.0
    return len(wa & wb) / len(wa | wb)


def classify_step(parent_strat, child_strat):
    """
    Returns (label, similarity):
      SELECTION     – identical text
      MINOR TWEAK   – very high similarity (≥0.85), small numeric change
      MUTATION      – moderate similarity (≥0.40), strategy reshaped
      INNOVATION    – low similarity (<0.40), substantially new strategy
    """
    if parent_strat is None:
        return "FOUNDER", 0.0
    p = clean_strat(parent_strat)
    c = clean_strat(child_strat)
    if p == c:
        return "SELECTION", 1.0
    sim = token_similarity(p, c)
    if sim >= 0.85:
        return "MINOR TWEAK", sim
    elif sim >= 0.40:
        return "MUTATION", sim
    else:
        return "INNOVATION", sim


STEP_COLORS = {
    "FOUNDER":     "#888888",
    "SELECTION":   "#2ca02c",
    "MINOR TWEAK": "#aec7e8",
    "MUTATION":    "#ff7f0e",
    "INNOVATION":  "#d62728",
}


# ---------------------------------------------------------------------------
# Lineage tracing
# ---------------------------------------------------------------------------

def trace_lineage(rank1_agent, agent_index):
    """
    Walk inheritance chain back to the founder.
    Returns list of dicts (ordered root → leaf).
    """
    chain = []
    visited = set()
    name = rank1_agent

    while name and name not in visited:
        visited.add(name)
        rec = agent_index.get(name)
        if rec is None:
            break
        chain.append(rec)
        parent_name = rec.get("inherited_from")
        if not parent_name:
            break
        name = parent_name

    chain.reverse()   # root first
    return chain


def annotate_chain(chain):
    """Add step_type, similarity, parent_strategy to each record."""
    annotated = []
    for i, rec in enumerate(chain):
        parent_strat = chain[i - 1]["strategy"] if i > 0 else None
        step, sim = classify_step(parent_strat, rec["strategy"])
        annotated.append({
            **rec,
            "step_type": step,
            "similarity": sim,
            "parent_strategy": parent_strat,
        })
    return annotated


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def wrap(text, width=70):
    return "\n".join(textwrap.wrap(clean_strat(text or ""), width))


def plot_lineage(annotated, model, run_ts, target_gen, output_dir):
    n = len(annotated)
    fig_h = max(6, n * 2.2)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n - 0.5)
    ax.axis("off")

    ax.set_title(
        f"Strategy lineage — rank-1 agent in generation {target_gen}\n"
        f"Model: {model}   Run: {run_ts}",
        fontsize=12, pad=14,
    )

    y_positions = list(range(n))  # top = root (earliest)

    for idx, rec in enumerate(annotated):
        y = n - 1 - idx   # flip so gen increases downward
        step  = rec["step_type"]
        color = STEP_COLORS[step]
        gen   = rec["generation"]
        agent = rec["agent_name"]
        irank = rec.get("inherited_rank", "—")

        # Vertical connector
        if idx < n - 1:
            ax.plot([0.08, 0.08], [y - 0.45, y + 0.45], color="#cccccc", lw=1.5, zorder=0)

        # Node circle
        circle = plt.Circle((0.08, y), 0.12, color=color, zorder=2, clip_on=False)
        ax.add_patch(circle)
        ax.text(0.08, y, f"G{gen}", ha="center", va="center",
                fontsize=7.5, fontweight="bold", color="white", zorder=3)

        # Step label
        ax.text(0.19, y + 0.25, step, ha="left", va="center",
                fontsize=8, color=color, fontweight="bold")
        if rec["similarity"] > 0:
            ax.text(0.19, y + 0.08, f"sim={rec['similarity']:.2f}", ha="left", va="center",
                    fontsize=7, color="#666666")
        ax.text(0.19, y - 0.1, f"agent {agent}  |  inherited rank {irank}",
                ha="left", va="center", fontsize=7.5, color="#444444")

        # Strategy text box
        strat_text = wrap(rec["strategy"], width=75)
        ax.text(0.19, y - 0.3, strat_text,
                ha="left", va="top", fontsize=6.8, color="#111111",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f7f7f7",
                          edgecolor=color, alpha=0.85))

    # Legend
    legend_patches = [
        mpatches.Patch(color=v, label=k)
        for k, v in STEP_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8,
              framealpha=0.9, title="Step type", title_fontsize=8)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fname = f"lineage_{model}_run{run_ts}_gen{target_gen}.png"
    out = os.path.join(output_dir, fname)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {out}")
    return out


def print_report(annotated, ranked_agents, model, run_ts, target_gen):
    print(f"\n{'='*70}")
    print(f"STRATEGY LINEAGE  |  model={model}  run={run_ts}  target_gen={target_gen}")
    print(f"{'='*70}")
    print(f"\nGen {target_gen} rankings (top 5):")
    for rank, (agent, res) in enumerate(ranked_agents[:5], 1):
        print(f"  #{rank}  {agent}  resources={res:.1f}")

    print(f"\nLineage of rank-1 agent '{annotated[-1]['agent_name']}' "
          f"(tracing back {len(annotated)} generations):\n")

    for rec in annotated:
        step  = rec["step_type"]
        gen   = rec["generation"]
        agent = rec["agent_name"]
        irank = rec.get("inherited_rank", "—")
        sim   = rec["similarity"]
        strat = clean_strat(rec["strategy"])

        print(f"  ── Gen {gen:2d}  {agent:8s}  [{step:12s}  sim={sim:.2f}]"
              f"  ← inherited rank {irank}")
        print(f"     Strategy: {strat[:120]}")
        if rec["parent_strategy"] and step not in ("SELECTION", "FOUNDER"):
            parent_clean = clean_strat(rec["parent_strategy"])
            print(f"     Parent:   {parent_clean[:120]}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Trace rank-1 strategy lineage in a simulation run.")
    parser.add_argument("--model",      default="claude",
                        choices=list(MODEL_PATTERNS))
    parser.add_argument("--batch",      default="0424",
                        help="Subdirectory under simulations/ (e.g. 0417 or 0424)")
    parser.add_argument("--run_idx",    type=int, default=0,
                        help="Index of run to analyse (0 = first found)")
    parser.add_argument("--target_gen", type=int, default=10)
    parser.add_argument("--output_dir", default="figures/strategy_inheritance")
    args = parser.parse_args()

    batch_dir = os.path.join("simulations", args.batch)
    strat_files, accum_files = find_run_files(batch_dir, args.model)
    pairs = match_files(strat_files, accum_files)

    if not pairs:
        print(f"No matched strat+accum file pairs found in {batch_dir} for {args.model}.")
        return

    if args.run_idx >= len(pairs):
        print(f"run_idx={args.run_idx} out of range (only {len(pairs)} runs available).")
        return

    strat_path, accum_path = pairs[args.run_idx]
    run_ts = _ts(strat_path)
    print(f"Using run: {run_ts}")
    print(f"  strat: {os.path.basename(strat_path)}")
    print(f"  accum: {os.path.basename(accum_path)}")

    # Rank agents in target generation
    ranked = rank_agents(accum_path, args.target_gen)
    if not ranked:
        print(f"No agents found in generation {args.target_gen}.")
        return
    rank1_agent = ranked[0][0]
    print(f"\nRank-1 agent in gen {args.target_gen}: {rank1_agent}  "
          f"(resources={ranked[0][1]:.1f})")

    # Load all strategies and trace lineage
    agent_index = load_agent_index(strat_path)
    chain = trace_lineage(rank1_agent, agent_index)
    annotated = annotate_chain(chain)

    print_report(annotated, ranked, args.model, run_ts, args.target_gen)
    plot_lineage(annotated, args.model, run_ts, args.target_gen, args.output_dir)

    # Summary stats
    steps = [r["step_type"] for r in annotated]
    from collections import Counter
    counts = Counter(steps)
    print("Step type summary:")
    for k, v in counts.most_common():
        print(f"  {k:14s}: {v}")


if __name__ == "__main__":
    main()
