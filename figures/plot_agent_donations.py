#!/usr/bin/env python3
"""
Plot donation reactivity for agents across all models in a folder.

X-axis: pair's donation % in their previous donor round (% of 10 units)
Y-axis: my donation % in round x (% of 10 units)

Plots are generated for --generation (default: 10) using the first run only.

Usage:
    python figures/plot_agent_donations.py <folder> \
        [--model claude|gpt|gemini|qwen] [--agent NAME] \
        [--generation N] [--output_dir DIR] [--cost_cap X]

For simulation-wide strategy metrics (strategy_analysis.json), use:
    python figures/strategy_analysis.py
"""

import glob
import json
import argparse
import os
import re
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ENDOWMENT = 10.0

_TS_RE = re.compile(r'_(\d{8}_\d{6})\.json$')


def _extract_timestamp(path):
    m = _TS_RE.search(os.path.basename(path))
    return m.group(1) if m else ""


def _model_label(basename):
    m = re.search(r'Final_Accum_(.+?)_\d{8}_\d{6}\.json$', basename)
    if m:
        return m.group(1)
    name = re.sub(r'^.*Final_Accum_', '', basename)
    name = re.sub(r'_\d{8}_\d{6}\.json$', '', name)
    return name or basename


_MODEL_KEYWORDS = {
    'claude': 'claude',
    'gpt':    'gpt',
    'gemini': 'gemini',
    'qwen':   'qwen',
}

def _short_name(label):
    """Map a full model label to a short directory name (claude/gpt/gemini/qwen)."""
    ll = label.lower()
    for short, kw in _MODEL_KEYWORDS.items():
        if kw in ll:
            return short
    return label  # fallback: use full label if no match


def discover_files(folder, model_filter=None):
    """
    Return {model_label: [path, ...]} sorted newest-first by timestamp.
    Optionally filter to one model family.
    """
    paths = glob.glob(os.path.join(folder, "*Final_Accum_*.json"))
    groups = {}
    for p in paths:
        label = _model_label(os.path.basename(p))
        groups.setdefault(label, []).append(p)
    for label in groups:
        groups[label].sort(key=_extract_timestamp, reverse=True)

    if model_filter:
        keyword = _MODEL_KEYWORDS.get(model_filter.lower(), model_filter.lower())
        groups = {k: v for k, v in groups.items() if keyword in k.lower()}

    return groups


def load_data(json_path):
    with open(json_path) as f:
        return json.load(f)


def build_lookup(agents_data, generation):
    lookup = {}
    for r in agents_data:
        if r['current_generation'] == generation:
            key = (r['game_number'], r['round_number'], r['agent_name'])
            lookup[key] = r
    return lookup


def collect_scatter_data(agents_data, agent_name, generation):
    """
    Return (pair_prev_pct, my_pct, global_round, game, is_first_round) for each donor round.

    global_round is 1–12 for game 1 and 13–24 for game 2.
    is_first_round=True for global rounds 1 and 13 (round_number==1 in each game),
    which have no prior partner history and are plotted as triangles at x=0.
    Each agent donates in either round 1 or round 13, never both.
    """
    lookup = build_lookup(agents_data, generation)
    points = []

    for r in agents_data:
        if r['current_generation'] != generation:
            continue
        if r['agent_name'] != agent_name:
            continue
        if not r.get('is_donor', False):
            continue

        game = r['game_number']
        round_num = r['round_number']
        my_pct = ((r.get('donated', 0) or 0) / ENDOWMENT) * 100.0
        global_round = round_num if game == 1 else round_num + 12

        # Round 1 and round 13 (round_number==1 in each game) have no prior partner
        # history — plot as triangles at x=0.
        if round_num == 1:
            points.append((None, my_pct, global_round, game, True))
            continue

        pair_name = r.get('paired_with')
        if not pair_name:
            points.append((None, my_pct, global_round, game, True))
            continue

        # Find the pair's most recent donor round before this one within this game.
        prev = None
        for prev_round in range(round_num - 1, 0, -1):
            candidate = lookup.get((game, prev_round, pair_name))
            if candidate and candidate.get('is_donor', False):
                prev = candidate
                break
        if prev is None:
            points.append((None, my_pct, global_round, game, True))
            continue

        pair_pct = ((prev.get('donated', 0) or 0) / ENDOWMENT) * 100.0
        points.append((pair_pct, my_pct, global_round, game, False))

    return points


def get_strategy(agents_data, agent_name, generation):
    for r in agents_data:
        if (r['current_generation'] == generation
                and r['agent_name'] == agent_name
                and r.get('strategy')):
            return r['strategy']
    return ""


def ask_claude(strategy_text):
    """
    Returns (is_specified: bool, code: str | None, explanation: str).
    """
    try:
        import anthropic
    except ImportError:
        print("anthropic package not installed — skipping strategy code generation.")
        return False, None, "anthropic not installed"

    import os as _os
    if not _os.environ.get("ANTHROPIC_API_KEY"):
        env_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), ".env")
        if _os.path.exists(env_path):
            with open(env_path) as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line.startswith("ANTHROPIC_API_KEY="):
                        _os.environ["ANTHROPIC_API_KEY"] = _line.split("=", 1)[1].strip().strip('"').strip("'")
                        break

    client = anthropic.Anthropic()

    prompt = (
        "You are analyzing an agent's strategy in a donor game.\n\n"
        f'The agent\'s strategy is:\n"{strategy_text}"\n\n'
        "In this game:\n"
        f"- Each round the agent receives a fixed new {int(ENDOWMENT)} units and donates 0–{int(ENDOWMENT)} of them.\n"
        "- Unspent units go to savings and cannot be used again.\n"
        "- Each round the agent is randomly paired with another agent. One is the donor, the other the recipient.\n"
        "- The recipient receives 2× the donated amount.\n"
        "- There are 2 games per generation, each up to 12 rounds (80% chance of continuing each round).\n"
        "- In round 1 of each game (rounds 1 and 13 overall), the agent has NO information about their "
        "current partner's previous behaviour — they cannot condition on partner history in those rounds.\n"
        "- From round 2 onwards, the agent can see what % of 10 units their CURRENT partner donated "
        "in the partner's immediately preceding round.\n\n"
        "Task:\n"
        "1. Is this strategy FULLY SPECIFIED? A fully specified strategy produces a specific donation amount "
        "(or unambiguous calculation resolving to a number) for every situation the agent can be in:\n"
        "   - It must specify what to donate in rounds 1 and 13 (no partner info available).\n"
        "   - It must specify what to donate from round 2 onwards given the partner's previous donation % (0–100).\n"
        "   - Referencing specific round numbers IS allowed — round numbers are observable.\n"
        "   - Any arithmetic or threshold rule that resolves to a concrete number IS fully specified "
        "(e.g. 'donate same % as partner', 'donate 5 in round 1 then mirror partner', 'donate 10 always').\n"
        "   - NOT fully specified: vague language with no resolvable number ('donate moderately', "
        "'be generous'), references to unobservable information (opponent identity, total savings of others, "
        "random chance), or any situation left genuinely ambiguous.\n"
        "2. If YES, write a Python function. pair_prev_pct is None for rounds 1 and 13 (no partner info):\n"
        "   def strategy_fn(pair_prev_pct):\n"
        '       """Returns donation % (0-100). pair_prev_pct is None in rounds 1 and 13."""\n'
        "       ...\n"
        "   Output only the function, no other code.\n"
        "3. If NO, explain why in one sentence.\n\n"
        "Respond in this EXACT format:\n"
        "FULLY_SPECIFIED: yes/no\n"
        "EXPLANATION: <one sentence>\n"
        "CODE:\n"
        "<python function or NONE>"
    )

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    response = msg.content[0].text
    print(f"\n--- Claude response ---\n{response}\n-----------------------\n")

    is_specified = bool(re.search(r'FULLY_SPECIFIED:\s*yes', response, re.IGNORECASE))

    explanation = ""
    m = re.search(r'EXPLANATION:\s*(.*?)(?:\nCODE:|$)', response, re.DOTALL)
    if m:
        explanation = m.group(1).strip()

    code = None
    m = re.search(r'CODE:\s*\n(.*)', response, re.DOTALL)
    if m:
        code_str = m.group(1).strip()
        code_str = re.sub(r'^```[a-z]*\n?', '', code_str)
        code_str = re.sub(r'\n?```\s*$', '', code_str).strip()
        if code_str and code_str.upper() != "NONE" and "def strategy_fn" in code_str:
            code = code_str

    return is_specified, code, explanation


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
    """Insert `pass` after block-opening lines whose body is empty or comment-only."""
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
            print(f"Error running strategy code: {e}")
            return None
    except Exception as e:
        print(f"Error running strategy code: {e}")
        return None
    fn = namespace.get('strategy_fn')
    if fn is None:
        return None
    results = []
    for x in x_values:
        try:
            results.append(float(np.clip(fn(x), 0, 100)))
        except Exception:
            results.append(None)
    return results if any(v is not None for v in results) else None


COST_PER_1K_INPUT  = 3.0 / 1_000_000
COST_PER_1K_OUTPUT = 15.0 / 1_000_000


def estimate_cost(strategy_text):
    prompt_chars = len(strategy_text) + 800
    output_chars = 600
    return (prompt_chars / 4) * COST_PER_1K_INPUT + (output_chars / 4) * COST_PER_1K_OUTPUT


def _classify_strategy(strategy, strategy_cache, total_cost, cost_cap):
    """
    Return (is_specified, code, explanation, total_cost).
    Uses strategy_cache to avoid duplicate Claude calls.
    """
    if strategy in strategy_cache:
        is_specified, code, explanation = strategy_cache[strategy]
        return is_specified, code, explanation, total_cost

    estimated = estimate_cost(strategy)
    if total_cost + estimated > cost_cap:
        print(f"  Cost cap reached (${total_cost:.2f} used). Skipping Claude call.")
        result = (False, None, "Cost cap reached — Claude not called.")
    else:
        result = ask_claude(strategy)
        total_cost += estimated
        print(f"  Fully specified: {result[0]}  |  {result[2]}")
        print(f"  Estimated cumulative cost: ${total_cost:.2f} / ${cost_cap:.2f}")

    strategy_cache[strategy] = result
    return result[0], result[1], result[2], total_cost


def plot_reactivity(agent_name, points, strategy, generation, output_dir,
                    is_specified, code, explanation):
    if not points:
        print(f"  No data points for agent {agent_name} gen {generation}.")
        return

    pair_pcts    = [p[0] for p in points]
    my_pcts      = [p[1] for p in points]
    round_keys   = [p[2] for p in points]
    games        = [p[3] for p in points]
    is_first_rnd = [p[4] for p in points]

    fig, ax = plt.subplots(figsize=(8, 7))

    def _group_points(xs, ys, keys, games):
        """Group points by (x, y) coordinate. Returns list of (x, y, label, color)."""
        groups = {}
        for x, y, k, g in zip(xs, ys, keys, games):
            coord = (round(x, 6), round(y, 6))
            groups.setdefault(coord, []).append((k, g))
        result = []
        for (x, y), items in groups.items():
            items_sorted = sorted(items, key=lambda t: t[0])
            label = ','.join(f'r{k}' for k, g in items_sorted)
            # Color: mixed if both games present, else by game
            gs = {g for _, g in items_sorted}
            if len(gs) == 2:
                color = '#9467bd'   # purple = both games
            elif 1 in gs:
                color = 'steelblue'
            else:
                color = 'tomato'
            result.append((x, y, label, color))
        return result

    # Reactive rounds — circles, grouped by coordinate
    rx = [x for x, f in zip(pair_pcts, is_first_rnd) if not f]
    ry = [y for y, f in zip(my_pcts,   is_first_rnd) if not f]
    rk = [k for k, f in zip(round_keys,is_first_rnd) if not f]
    rg = [g for g, f in zip(games,     is_first_rnd) if not f]
    if rx:
        for x, y, label, color in _group_points(rx, ry, rk, rg):
            ax.scatter([x], [y], c=[color], s=70, alpha=0.85, zorder=3,
                       edgecolors='white', linewidths=0.5)
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4),
                        fontsize=8, fontweight='bold', color='#333333',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7, edgecolor='none'))

    # Rounds 1 and 13 — triangles, grouped by coordinate
    fx = [0.0 for x, f in zip(pair_pcts,  is_first_rnd) if f]
    fy = [y   for y, f in zip(my_pcts,    is_first_rnd) if f]
    fk = [k for k, f in zip(round_keys, is_first_rnd) if f]
    fg = [g for g, f in zip(games,      is_first_rnd) if f]
    if fx:
        for x, y, label, color in _group_points(fx, fy, fk, fg):
            ax.scatter([x], [y], c=[color], s=90, alpha=0.85, zorder=4, marker='^',
                       edgecolors='white', linewidths=0.5)
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4),
                        fontsize=8, fontweight='bold', color='#333333',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7, edgecolor='none'))

    if is_specified and code:
        x_line = np.linspace(0, 100, 300)
        y_line = eval_strategy(code, x_line)
        if y_line is not None:
            ax.plot(x_line, y_line, color='#888888', linewidth=2,
                    linestyle='--', zorder=4, alpha=0.5)

    legend_els = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', markersize=9, label='Game 1 (r2–r12)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato',    markersize=9, label='Game 2 (r14–r24)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#9467bd',   markersize=9, label='Both games (overlap)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#2ca02c',   markersize=9, label='r1 (no prior info)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#d62728',   markersize=9, label='r13 (no prior info)'),
    ]
    if is_specified and code:
        legend_els.append(Line2D([0], [0], color='#888888', linewidth=2, linestyle='--',
                                 label='Coded strategy', alpha=0.5))
    ax.legend(handles=legend_els, fontsize=8, loc='upper left')

    ax.set_xlabel("Pair's donation % in their previous donor round  (% of 10 units)", fontsize=10)
    ax.set_ylabel("My donation % in round x  (% of 10 units)", fontsize=11)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.set_title(f'Agent {agent_name} — Generation {generation} — Donation Reactivity', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.plot([0, 100], [0, 100], color='grey', linewidth=0.8, linestyle=':', alpha=0.5)

    plt.tight_layout()

    spec_label = "FULLY SPECIFIED" if is_specified else "NOT FULLY SPECIFIED"
    strategy_wrapped = textwrap.fill(f"Strategy: {strategy}", width=110)
    explanation_line = f"[{spec_label}] {explanation}" if explanation else f"[{spec_label}]"
    fig.text(0.5, -0.02, f"{strategy_wrapped}\n{explanation_line}",
             ha='center', va='top', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.85, edgecolor='#cccccc'))

    os.makedirs(output_dir, exist_ok=True)
    out = os.path.join(output_dir, f"gen{generation}_{agent_name}_reactivity.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def process_file_for_plots(json_path, agent_filter, generation, output_dir,
                           strategy_cache, total_cost, cost_cap):
    """
    Generate reactivity plots for one generation of one JSON file.
    Returns (total_cost, underspecified_list).
    """
    data = load_data(json_path)
    agents_data = data.get('agents_data', [])
    underspecified = []

    if agent_filter:
        agent_names = [agent_filter]
    else:
        agent_names = sorted({r['agent_name'] for r in agents_data
                               if r.get('current_generation') == generation
                               and r.get('is_donor', False)})
        print(f"  Found {len(agent_names)} agents in generation {generation}: {agent_names}")

    for agent_name in agent_names:
        print(f"\n=== Agent: {agent_name}  |  Generation: {generation} ===")
        strategy = get_strategy(agents_data, agent_name, generation)
        points   = collect_scatter_data(agents_data, agent_name, generation)
        print(f"Strategy: {strategy}")
        print(f"Scatter points: {len(points)}")

        is_specified, code, explanation, total_cost = _classify_strategy(
            strategy, strategy_cache, total_cost, cost_cap)

        if not is_specified:
            underspecified.append({
                "agent": agent_name,
                "source_file": os.path.basename(json_path),
                "strategy": strategy,
                "explanation": explanation,
            })

        plot_reactivity(agent_name=agent_name, points=points, strategy=strategy,
                        generation=generation, output_dir=output_dir,
                        is_specified=is_specified, code=code, explanation=explanation)

    return total_cost, underspecified


def main():
    parser = argparse.ArgumentParser(
        description='Plot donation reactivity for agents in a given generation.')
    parser.add_argument('folder', help='Folder containing *Final_Accum_*.json files')
    parser.add_argument('--model', default=None, choices=['claude', 'gpt', 'gemini', 'qwen'],
                        help='Filter to a specific model family (default: all models)')
    parser.add_argument('--agent', default=None,
                        help='Agent name filter (e.g. 1_1). Omit to process all agents.')
    parser.add_argument('--generation', type=int, default=10,
                        help='Generation to generate plots for (default: 10)')
    parser.add_argument('--output_dir', default='figures/reactivity_plots/gen10',
                        help='Base output directory; one subdir per model is created')
    parser.add_argument('--cost_cap', type=float, default=10.0,
                        help='Max USD to spend on Claude API across all files (default: $10)')
    args = parser.parse_args()

    model_groups = discover_files(args.folder, model_filter=args.model)
    if not model_groups:
        print(f"No *Final_Accum_*.json files found in {args.folder}")
        return

    print(f"Discovered models: {list(model_groups.keys())}")

    total_cost = 0.0
    strategy_cache = {}

    for model_label, file_paths in sorted(model_groups.items()):
        print(f"\n{'='*60}")
        print(f"Model: {model_label}  ({len(file_paths)} file(s))")
        print(f"{'='*60}")

        model_out = os.path.join(args.output_dir, _short_name(model_label))

        # Plots: first run only, target generation only
        plot_file = file_paths[0]
        print(f"\n[PLOTS] {os.path.basename(plot_file)}")
        total_cost, run_underspec = process_file_for_plots(
            json_path=plot_file,
            agent_filter=args.agent,
            generation=args.generation,
            output_dir=model_out,
            strategy_cache=strategy_cache,
            total_cost=total_cost,
            cost_cap=args.cost_cap,
        )
        if run_underspec:
            ts = _extract_timestamp(plot_file) or "run1"
            upath = os.path.join(model_out, f"gen{args.generation}_underspecified_{ts}.json")
            os.makedirs(model_out, exist_ok=True)
            with open(upath, 'w') as f:
                json.dump(run_underspec, f, indent=2)
            print(f"  Underspecified saved: {upath}")

    print(f"\nDone. Estimated total API cost: ${total_cost:.2f} / ${args.cost_cap:.2f}")


if __name__ == '__main__':
    main()
