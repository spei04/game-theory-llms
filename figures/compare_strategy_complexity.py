#!/usr/bin/env python3
"""
Compare strategy text across three history-mode setups:
  - simple   (neutral_baseline)
  - 3traces  (history_3traces)
  - 3chain   (history_3chain)

For each (setup, model) pair:
  1. Picks one run.
  2. Walks generations 1..10, identifies the rank-1 agent in each generation,
     and prints that agent's strategy text.
  3. Computes complexity metrics (char length, word count, sentence count,
     numeric-token count) over ALL strategies in that run and reports
     mean/median/max.

Usage:
    python figures/compare_strategy_complexity.py [--model claude] [--run_idx 0]
"""

import argparse
import glob
import json
import os
import re
import statistics
import textwrap
from collections import defaultdict

SETUPS = {
    "simple":  "simulations/neutral_baseline",
    "3traces": "simulations/history_3traces",
    "3chain":  "simulations/history_3chain",
}

MODEL_PATTERNS = {
    "claude": "claude",
    "gpt":    "gpt",
    "gemini": "gemini",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def find_run_pair(setup_dir, model, run_idx):
    """Returns (strat_path, accum_path) for the run_idx-th run of model."""
    model_dir = os.path.join(setup_dir, model)
    strat = sorted(glob.glob(os.path.join(model_dir, f"all_strategies*{MODEL_PATTERNS[model]}*.json")))
    accum = sorted(glob.glob(os.path.join(model_dir, f"Final_Accum*{MODEL_PATTERNS[model]}*.json")))
    if not strat or not accum:
        return None, None
    return strat[min(run_idx, len(strat) - 1)], accum[min(run_idx, len(accum) - 1)]


def find_all_runs(setup_dir, model):
    """Returns [(strat_path, accum_path), ...] paired by timestamp suffix."""
    model_dir = os.path.join(setup_dir, model)
    strat = sorted(glob.glob(os.path.join(model_dir, f"all_strategies*{MODEL_PATTERNS[model]}*.json")))
    accum = sorted(glob.glob(os.path.join(model_dir, f"Final_Accum*{MODEL_PATTERNS[model]}*.json")))
    # Trust sort order — both lists share the same timestamp suffix and were
    # written by the same run.
    return list(zip(strat, accum))


def rank_agents_by_gen(accum_path, endowment=10, num_rounds=12):
    """Returns {gen: [(agent_name, savings), ...sorted best-first]}.

    True final score per agent per generation, matching the game's own
    top_final_score = savings + resources at end of game 2:

        score = (endowment * num_rounds * num_games) - Σ donated + Σ received
              = base_pool - total_donated + total_received

    Each agent gets `endowment` units every round. Donor sacrifices `donated`;
    recipient's savings grow by `received`. Since role alternates across the
    2 games, every agent is donor in `num_rounds` rounds and recipient in the
    other `num_rounds` rounds.
    """
    with open(accum_path) as f:
        data = json.load(f)
    rows = data.get("agents_data", [])

    totals = defaultdict(lambda: defaultdict(lambda: {"donated": 0.0, "received": 0.0}))
    for r in rows:
        g = r["current_generation"]
        a = r["agent_name"]
        totals[g][a]["donated"]  += r.get("donated", 0) or 0
        totals[g][a]["received"] += r.get("received", 0) or 0

    num_games = 2
    base_pool = endowment * num_rounds * num_games   # =240 with defaults

    out = {}
    for g, per_agent in totals.items():
        scored = []
        for a, d in per_agent.items():
            score = base_pool - d["donated"] + d["received"]
            scored.append((a, score))
        scored.sort(key=lambda x: -x[1])
        out[g] = scored
    return out


def load_strats_by_gen_agent(strat_path):
    """Returns {(gen, agent_name): strategy_text}."""
    with open(strat_path) as f:
        data = json.load(f)
    out = {}
    for gen_list in data:
        for a in gen_list:
            g = a.get("generation")
            n = a.get("agent_name")
            if (g, n) not in out:
                out[(g, n)] = a.get("strategy", "") or ""
    return out


def clean(s):
    s = re.sub(r'^My strategy will be\s*[\*:]+\s*', '', s or "", flags=re.I).strip()
    s = re.sub(r'\n*\s*Inherited from:\s*\S[^\n]*', '', s, flags=re.I).strip()
    return s


def complexity(text):
    t = clean(text)
    chars     = len(t)
    words     = len(re.findall(r'\w+', t))
    sentences = len([s for s in re.split(r'[.!?]+\s', t) if s.strip()])
    numbers   = len(re.findall(r'\b\d+(?:\.\d+)?\b', t))
    return chars, words, sentences, numbers


# ── main ─────────────────────────────────────────────────────────────────────

def print_lineage(setup, model, strat_by_ga, ranks_by_gen):
    print(f"\n{'='*78}")
    print(f"  {setup.upper()}  /  {model}  —  rank-1 strategy per generation")
    print(f"{'='*78}")
    for gen in sorted(ranks_by_gen):
        if not ranks_by_gen[gen]:
            continue
        top_agent, top_res = ranks_by_gen[gen][0]
        strat = strat_by_ga.get((gen, top_agent), "")
        clean_s = clean(strat)
        print(f"\n--- Gen {gen:2d}  agent {top_agent}  total_final_score={top_res:.1f} ---")
        wrapped = textwrap.fill(clean_s, width=92,
                                initial_indent="  ", subsequent_indent="  ")
        print(wrapped[:1400] + ("..." if len(wrapped) > 1400 else ""))


def setup_stats(strat_by_ga):
    cs, ws, ss, ns = [], [], [], []
    for s in strat_by_ga.values():
        c, w, ssent, nn = complexity(s)
        cs.append(c); ws.append(w); ss.append(ssent); ns.append(nn)
    return {
        "n":        len(cs),
        "chars":    (statistics.mean(cs),    statistics.median(cs),    max(cs)),
        "words":    (statistics.mean(ws),    statistics.median(ws),    max(ws)),
        "sents":    (statistics.mean(ss),    statistics.median(ss),    max(ss)),
        "numbers":  (statistics.mean(ns),    statistics.median(ns),    max(ns)),
    }


def print_one_strategy_per_run(setup, model, gen):
    """For every run of (setup, model), print the rank-1 strategy in `gen`.
    Lets you eyeball variability across replicates."""
    setup_dir = SETUPS[setup]
    pairs = find_all_runs(setup_dir, model)
    if not pairs:
        print(f"[skip] no runs found for {setup}/{model}")
        return

    print(f"\n{'='*78}")
    print(f"  {setup.upper()}  /  {model}  —  rank-1 strategy in gen {gen} across {len(pairs)} runs")
    print(f"{'='*78}")

    for run_i, (strat_path, accum_path) in enumerate(pairs):
        strat_by_ga = load_strats_by_gen_agent(strat_path)
        ranks = rank_agents_by_gen(accum_path).get(gen, [])
        if not ranks:
            continue
        top_agent, top_res = ranks[0]
        s = clean(strat_by_ga.get((gen, top_agent), ""))
        ts = re.search(r'(\d{8}_\d{6})', os.path.basename(strat_path))
        run_tag = ts.group(1) if ts else f"run{run_i}"
        print(f"\n--- run {run_i:2d} ({run_tag})  agent {top_agent}  res={top_res:.1f} ---")
        print(textwrap.fill(s, width=92, initial_indent="  ", subsequent_indent="  "))


def print_all_agents_at_gen(setup, model, run_idx, gen):
    """For one run of (setup, model), print ALL 12 agents' strategies in `gen`,
    sorted by savings (best first)."""
    setup_dir = SETUPS[setup]
    strat_path, accum_path = find_run_pair(setup_dir, model, run_idx)
    if not strat_path:
        print(f"[skip] no run found for {setup}/{model}")
        return

    print(f"\n{'='*78}")
    print(f"  {setup.upper()}  /  {model}  —  ALL agents in generation {gen}")
    print(f"  run: {os.path.basename(strat_path)}")
    print(f"{'='*78}")

    strat_by_ga = load_strats_by_gen_agent(strat_path)
    ranks = rank_agents_by_gen(accum_path).get(gen, [])
    if not ranks:
        print(f"  (no agents in gen {gen})")
        return

    for i, (agent, savings) in enumerate(ranks, 1):
        s = clean(strat_by_ga.get((gen, agent), "(no strategy text)"))
        print(f"\n  #{i:2d}  agent {agent}  total_final_score={savings:.1f}")
        print(textwrap.fill(s, width=90, initial_indent="       ", subsequent_indent="       "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default="claude", choices=list(MODEL_PATTERNS))
    ap.add_argument("--run_idx", type=int, default=0)
    ap.add_argument("--all_runs", action="store_true",
                    help="Print the rank-1 strategy from every run (one per run) "
                         "instead of the full gen-1..10 lineage of a single run.")
    ap.add_argument("--all_agents", action="store_true",
                    help="Print ALL agents' strategies at --gen for one run of each setup.")
    ap.add_argument("--gen", type=int, default=10,
                    help="Which generation to pick strategies from "
                         "in --all_runs / --all_agents mode (default: 10).")
    ap.add_argument("--setups", nargs="+", default=list(SETUPS),
                    help=f"Subset of setups to include (default: all). "
                         f"Choices: {list(SETUPS)}")
    args = ap.parse_args()

    if args.all_agents:
        for setup in args.setups:
            if setup not in SETUPS:
                print(f"[skip] unknown setup: {setup}")
                continue
            print_all_agents_at_gen(setup, args.model, args.run_idx, args.gen)
        return

    if args.all_runs:
        for setup in args.setups:
            if setup not in SETUPS:
                print(f"[skip] unknown setup: {setup}")
                continue
            print_one_strategy_per_run(setup, args.model, args.gen)
        return

    summary_rows = []

    for setup in args.setups:
        if setup not in SETUPS:
            print(f"[skip] unknown setup: {setup}")
            continue
        setup_dir = SETUPS[setup]
        strat_path, accum_path = find_run_pair(setup_dir, args.model, args.run_idx)
        if not strat_path:
            print(f"[skip] no run found for {setup}/{args.model}")
            continue

        print(f"\n>>> {setup}/{args.model}  run={os.path.basename(strat_path)}")
        strat_by_ga = load_strats_by_gen_agent(strat_path)
        ranks_by_gen = rank_agents_by_gen(accum_path)

        print_lineage(setup, args.model, strat_by_ga, ranks_by_gen)

        stats = setup_stats(strat_by_ga)
        summary_rows.append((setup, stats))

    # ── final comparison table ──
    print(f"\n\n{'='*78}")
    print(f"  COMPLEXITY COMPARISON  ({args.model})")
    print(f"{'='*78}")
    hdr = f"{'setup':10s} {'n':>4s}  {'chars (mean/med/max)':>26s}  {'words':>18s}  {'sents':>14s}  {'#nums':>14s}"
    print(hdr)
    print("-" * len(hdr))
    for setup, s in summary_rows:
        c, w, se, nn = s["chars"], s["words"], s["sents"], s["numbers"]
        print(f"{setup:10s} {s['n']:>4d}  "
              f"{c[0]:6.0f}/{c[1]:5.0f}/{c[2]:5.0f}    "
              f"{w[0]:5.0f}/{w[1]:4.0f}/{w[2]:4.0f}  "
              f"{se[0]:4.1f}/{se[1]:4.0f}/{se[2]:4.0f}  "
              f"{nn[0]:4.1f}/{nn[1]:4.0f}/{nn[2]:4.0f}")

    # Ratios vs simple
    if summary_rows and summary_rows[0][0] == "simple":
        base = summary_rows[0][1]
        print(f"\nRatio vs simple:")
        for setup, s in summary_rows[1:]:
            r_c = s["chars"][0] / base["chars"][0] if base["chars"][0] else 0
            r_w = s["words"][0] / base["words"][0] if base["words"][0] else 0
            r_s = s["sents"][0] / base["sents"][0] if base["sents"][0] else 0
            r_n = s["numbers"][0] / base["numbers"][0] if base["numbers"][0] else 0
            print(f"  {setup:10s} chars ×{r_c:.2f}  words ×{r_w:.2f}  sents ×{r_s:.2f}  nums ×{r_n:.2f}")


if __name__ == "__main__":
    main()
