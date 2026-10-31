#!/usr/bin/env python3
"""
Compute strategy adherence metrics across all runs and generations.

For each model, processes every run file and every generation to produce:
  - % agents with underspecified strategies
  - For fully specified strategies: MAE, correlation, adherence gap
    (predicted avg - actual avg donation %)
  - Per-agent scatter points for downstream re-plotting

Output: one strategy_analysis.json per model in --output_dir/{model}/

Strategy cache: identical strategy strings are only sent to Claude once,
drastically reducing API cost across runs/generations.

Usage:
    python figures/strategy_analysis.py <folder> \
        [--model claude|gpt|gemini|qwen] [--agent NAME] \
        [--output_dir DIR] [--cost_cap X]

"""

import glob
import json
import argparse
import os
import re
import numpy as np


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
    """Map a full model label to a short directory name (claude/gpt/gemini/qwen).

    Only inspects the first segment of the label so downstream filename tags
    (e.g. _prompt-neutral_claude, _het-random) can't shadow the actual model
    name. The first segment is the substring up to the first underscore that
    is NOT part of a known model fragment.
    """
    # Strip everything after the first known boundary tag — these tags are
    # appended by the simulation script and may contain other model names
    # (e.g. _prompt-neutral_claude on a gemini run).
    ll = label.lower()
    for marker in ('_prompt-', '_het-', '_mode-', '_coopgain', '_coopgain_'):
        idx = ll.find(marker)
        if idx >= 0:
            ll = ll[:idx]
            break
    for short, kw in _MODEL_KEYWORDS.items():
        if kw in ll:
            return short
    return label


def discover_files(folder, model_filter=None):
    """
    Return {model_label: [path, ...]} sorted newest-first by timestamp.
    Optionally filter to one model family.
    """
    # Recursive: results are sometimes copied off the cluster as whole
    # array_<model>_<jobid>/ directories rather than as loose files, so the
    # runs for one model can sit at several depths under `folder`.
    paths = glob.glob(os.path.join(folder, "**", "*Final_Accum_*.json"),
                      recursive=True)
    groups = {}
    seen = set()
    for p in sorted(paths):
        label = _model_label(os.path.basename(p))
        # The same run can appear both loose and inside its array dir; the run
        # timestamp is the identity, so keep only the first copy of each.
        key = (label, _extract_timestamp(p))
        if key in seen:
            continue
        seen.add(key)
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
    which have no prior partner history.
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

        if round_num == 1:
            points.append((None, my_pct, global_round, game, True))
            continue

        pair_name = r.get('paired_with')
        if not pair_name:
            points.append((None, my_pct, global_round, game, True))
            continue

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
        "   The function MUST be a PURE function of pair_prev_pct. It is called once per\n"
        "   x value on a fresh interpreter and never in round order, so it CANNOT carry\n"
        "   state between calls. Specifically:\n"
        "   - Do NOT use function attributes (strategy_fn.state, hasattr(strategy_fn, ...)),\n"
        "     globals, closures, counters, caches, or round/call tracking of any kind.\n"
        "   - Do NOT reference anything the single argument cannot express: the round\n"
        "     number, your own savings or units, what you received on an earlier turn,\n"
        "     or the partner's identity.\n"
        "   - If the strategy DOES condition on such things, do not try to simulate them.\n"
        "     Substitute representative mid-game values (a typical round in the middle of\n"
        "     a game, with the agent holding its starting endowment), reduce the rule to a\n"
        "     function of pair_prev_pct alone, and say which values you assumed in\n"
        "     EXPLANATION. The output curve represents behaviour in a typical mid-game\n"
        "     round, which is what the analysis measures.\n"
        "3. If NO, explain why in one sentence.\n\n"
        "Respond in this EXACT format:\n"
        "FULLY_SPECIFIED: yes/no\n"
        "EXPLANATION: <one sentence>\n"
        "CODE:\n"
        "<python function or NONE>"
    )

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        # 600 truncated the longer emissions mid-statement: the code still
        # parsed (or defined strategy_fn and then raised on call), so the agent
        # was silently dropped by eval_strategy rather than flagged. Verbose
        # gen-10 strategies were the main casualty — 38 of 85 gpt-5 gen-10
        # strategies died this way.
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    response = msg.content[0].text
    # Echoing every response is ~1 KB x thousands of calls; redirected to a log
    # that filled the disk mid-run and killed the pass. Opt in with
    # STRATEGY_ANALYSIS_VERBOSE=1 when debugging a single strategy.
    if os.environ.get("STRATEGY_ANALYSIS_VERBOSE"):
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


def compute_adherence(points, code):
    """
    Compare actual donations to strategy predictions.
    Returns dict with mae, correlation, predicted_avg_pct, actual_avg_pct,
    adherence_gap (predicted - actual; positive = agent under-donates vs strategy).
    Rounds where the strategy raises an error are excluded from metrics.
    """
    pair_pcts = [p[0] for p in points]
    my_pcts   = [p[1] for p in points]

    predicted = eval_strategy(code, pair_pcts)
    if predicted is None or len(predicted) == 0:
        return None

    valid = [(a, p) for a, p in zip(my_pcts, predicted) if p is not None]
    if not valid:
        return None
    my_pcts_v   = [v[0] for v in valid]
    predicted_v = [v[1] for v in valid]

    errors = [abs(a - p) for a, p in zip(my_pcts_v, predicted_v)]
    mae = sum(errors) / len(errors)

    corr = None
    if len(my_pcts_v) > 1:
        c = np.corrcoef(my_pcts_v, predicted_v)[0, 1]
        if not np.isnan(c):
            corr = round(float(c), 3)

    pred_avg = sum(predicted_v) / len(predicted_v)
    act_avg  = sum(my_pcts_v) / len(my_pcts_v)

    n = len(my_pcts_v)
    n_nonzero_gap = sum(1 for a, p in zip(my_pcts_v, predicted_v) if abs(a - p) > 0.01)
    pct_nonzero_gap = round(n_nonzero_gap / n * 100, 1)

    return {
        "mae": round(mae, 2),
        "correlation": corr,
        "predicted_avg_pct": round(pred_avg, 2),
        "actual_avg_pct": round(act_avg, 2),
        "adherence_gap": round(pred_avg - act_avg, 2),
        "n_points": n,
        "pct_rounds_nonzero_gap": pct_nonzero_gap,
    }


def process_file_for_metrics(json_path, agent_filter, strategy_cache, total_cost, cost_cap,
                             generations=None):
    """
    Process all generations in one JSON file for metrics (no plots).
    Pass `generations` (an iterable of ints) to classify only those — each
    generation costs one classifier call per uncached strategy, so restricting
    to the generations a figure actually needs cuts the API bill accordingly.
    Returns (total_cost, list of per-agent record dicts).
    """
    data = load_data(json_path)
    agents_data = data.get('agents_data', [])
    run_ts = _extract_timestamp(json_path)

    all_generations = sorted({r['current_generation'] for r in agents_data})
    if generations is not None:
        wanted = set(generations)
        all_generations = [g for g in all_generations if g in wanted]
    records = []

    for gen in all_generations:
        if agent_filter:
            agent_names = [agent_filter]
        else:
            agent_names = sorted({r['agent_name'] for r in agents_data
                                   if r.get('current_generation') == gen
                                   and r.get('is_donor', False)})

        for agent_name in agent_names:
            strategy = get_strategy(agents_data, agent_name, gen)
            points   = collect_scatter_data(agents_data, agent_name, gen)

            is_specified, code, explanation, total_cost = _classify_strategy(
                strategy, strategy_cache, total_cost, cost_cap)

            adherence = None
            if is_specified and code and points:
                adherence = compute_adherence(points, code)

            adherence_by_game = {}
            if is_specified and code and points:
                for game_num in (1, 2):
                    game_points = [p for p in points if p[3] == game_num]
                    if game_points:
                        adh = compute_adherence(game_points, code)
                        if adh:
                            adherence_by_game[f"game_{game_num}"] = adh

            actual_avg = (sum(p[1] for p in points) / len(points)) if points else None

            scatter_points = [
                {
                    "pair_prev_pct": p[0],
                    "my_pct": round(p[1], 2),
                    "round_key": p[2],
                    "game": p[3],
                    "is_first_round": p[4],
                }
                for p in points
            ]

            records.append({
                "run": run_ts,
                "generation": gen,
                "agent": agent_name,
                "strategy": strategy,
                "is_specified": is_specified,
                "explanation": explanation,
                "strategy_code": code,
                "actual_avg_donation_pct": round(actual_avg, 2) if actual_avg is not None else None,
                "adherence": adherence,
                "adherence_by_game": adherence_by_game if adherence_by_game else None,
                "scatter_points": scatter_points,
            })

    return total_cost, records


def save_model_report(model_label, records, output_dir):
    """
    Compute and save strategy_analysis.json for one model.
    Returns summary dict.
    """
    total = len(records)
    if total == 0:
        return {}

    n_underspecified = sum(1 for r in records if not r['is_specified'])
    n_specified      = total - n_underspecified

    specified_with_adherence = [r for r in records if r['is_specified'] and r['adherence']]

    actual_avgs_all = [r['actual_avg_donation_pct'] for r in records
                       if r['actual_avg_donation_pct'] is not None]

    summary = {
        "model": model_label,
        "total_agent_observations": total,
        "n_runs": len({r['run'] for r in records}),
        "n_generations": len({r['generation'] for r in records}),
        "pct_underspecified": round(n_underspecified / total * 100, 1),
        "n_underspecified": n_underspecified,
        "n_fully_specified": n_specified,
        "mean_actual_cooperation_pct_all_agents": (
            round(sum(actual_avgs_all) / len(actual_avgs_all), 2) if actual_avgs_all else None
        ),
    }

    if specified_with_adherence:
        maes      = [r['adherence']['mae']               for r in specified_with_adherence]
        pred_avgs = [r['adherence']['predicted_avg_pct'] for r in specified_with_adherence]
        act_avgs  = [r['adherence']['actual_avg_pct']    for r in specified_with_adherence]
        gaps      = [r['adherence']['adherence_gap']     for r in specified_with_adherence]
        corrs     = [r['adherence']['correlation']       for r in specified_with_adherence
                     if r['adherence']['correlation'] is not None]
        nonzero_gap_pcts = [r['adherence']['pct_rounds_nonzero_gap'] for r in specified_with_adherence
                            if r['adherence'].get('pct_rounds_nonzero_gap') is not None]

        mean_pred  = round(sum(pred_avgs) / len(pred_avgs), 2)
        mean_act   = round(sum(act_avgs)  / len(act_avgs),  2)
        mean_gap   = round(sum(gaps)      / len(gaps),       2)
        mean_mae   = round(sum(maes)      / len(maes),       2)
        mean_corr  = round(sum(corrs)     / len(corrs),      3) if corrs else None
        mean_nonzero_gap_pct = round(sum(nonzero_gap_pcts) / len(nonzero_gap_pcts), 1) if nonzero_gap_pcts else None

        summary["fully_specified_adherence"] = {
            "n": len(specified_with_adherence),
            "mean_mae": mean_mae,
            "mean_predicted_cooperation_pct": mean_pred,
            "mean_actual_cooperation_pct": mean_act,
            "mean_adherence_gap": mean_gap,
            "mean_correlation": mean_corr,
            "mean_pct_rounds_nonzero_gap": mean_nonzero_gap_pct,
            "interpretation": (
                "adherence_gap > 0 means agents donate less than their strategy specifies; "
                "< 0 means they donate more; "
                "pct_rounds_nonzero_gap = % of rounds where actual donation differed from strategy prediction"
            ),
        }

        if mean_pred < 30:
            strat_quality = "low-cooperation strategies (predicted avg < 30%)"
        elif mean_pred < 60:
            strat_quality = "moderate-cooperation strategies (predicted avg 30–60%)"
        else:
            strat_quality = "high-cooperation strategies (predicted avg > 60%)"

        if abs(mean_gap) < 5:
            adherence_quality = "follow their strategies closely (gap < 5pp)"
        elif mean_gap > 0:
            adherence_quality = f"under-donate relative to strategy by {mean_gap:.1f}pp on average"
        else:
            adherence_quality = f"over-donate relative to strategy by {abs(mean_gap):.1f}pp on average"

        if mean_pred >= 40 and mean_gap > 10:
            driver = "primarily poor adherence (b): strategies specify reasonable cooperation but agents don't follow through"
        elif mean_pred < 30:
            driver = "primarily worse strategies (a): agents specify low-cooperation strategies"
        else:
            driver = "mixed: both strategy quality and adherence contribute"

        summary["cooperation_analysis"] = {
            "strategy_quality": strat_quality,
            "adherence_quality": adherence_quality,
            "main_driver_of_cooperation_gap": driver,
            "detail": (
                f"Specified strategies predict avg {mean_pred:.1f}% donation. "
                f"Actual avg: {mean_act:.1f}%. "
                f"Gap: {mean_gap:+.1f}pp. MAE: {mean_mae:.1f}pp."
            ),
        }

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"strategy_analysis_{_short_name(model_label)}.json")
    with open(report_path, 'w') as f:
        json.dump({"summary": summary, "agent_records": records}, f, indent=2)
    print(f"  Report saved: {report_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Compute strategy adherence metrics across all runs and generations.')
    parser.add_argument('folder', help='Folder containing *Final_Accum_*.json files')
    parser.add_argument('--model', default=None, choices=['claude', 'gpt', 'gemini', 'qwen'],
                        help='Filter to a specific model family (default: all models)')
    parser.add_argument('--agent', default=None,
                        help='Agent name filter (e.g. 1_1). Omit to process all agents.')
    parser.add_argument('--output_dir', default='figures/reactivity_plots/gen1',
                        help='Base output directory; one subdir per model is created')
    parser.add_argument('--cost_cap', type=float, default=10.0,
                        help='Max USD to spend on Claude API across all files (default: $10)')
    parser.add_argument('--generations', nargs='+', type=int, default=None,
                        help='Only classify these generations (e.g. 1 10). Each '
                             'generation costs classifier calls, so restricting to '
                             'the ones a figure needs cuts the API bill. '
                             'Default: all generations.')
    args = parser.parse_args()

    model_groups = discover_files(args.folder, model_filter=args.model)
    if not model_groups:
        print(f"No *Final_Accum_*.json files found in {args.folder}")
        return

    print(f"Discovered models: {list(model_groups.keys())}")

    total_cost = 0.0
    strategy_cache = {}
    all_summaries = {}

    for model_label, file_paths in sorted(model_groups.items()):
        print(f"\n{'='*60}")
        print(f"Model: {model_label}  ({len(file_paths)} file(s))")
        print(f"{'='*60}")

        model_out = os.path.join(args.output_dir, _short_name(model_label))

        print(f"\n[METRICS] Processing all {len(file_paths)} run(s), all generations...")
        all_records = []
        for i, fpath in enumerate(file_paths, 1):
            print(f"\n  File {i}/{len(file_paths)}: {os.path.basename(fpath)}")
            total_cost, records = process_file_for_metrics(
                json_path=fpath,
                agent_filter=args.agent,
                strategy_cache=strategy_cache,
                total_cost=total_cost,
                cost_cap=args.cost_cap,
                generations=args.generations,
            )
            all_records.extend(records)

        print(f"\n[REPORT] {model_label}: {len(all_records)} agent observations")
        summary = save_model_report(model_label, all_records, model_out)
        all_summaries[model_label] = summary

    print(f"\n{'='*60}")
    print(f"Done. Estimated total API cost: ${total_cost:.2f} / ${args.cost_cap:.2f}")
    print(f"Strategy cache size: {len(strategy_cache)} unique strategies\n")
    print("Cross-model summary:")
    for model_label, s in all_summaries.items():
        pct_u = s.get('pct_underspecified', '?')
        mean_coop = s.get('mean_actual_cooperation_pct_all_agents', '?')
        adh = s.get('fully_specified_adherence', {})
        gap = adh.get('mean_adherence_gap', '?')
        driver = s.get('cooperation_analysis', {}).get('main_driver_of_cooperation_gap', '')
        print(f"  {model_label}:")
        print(f"    Underspecified: {pct_u}%  |  Actual coop: {mean_coop}%  |  Adherence gap: {gap}pp")
        print(f"    {driver}")


if __name__ == '__main__':
    main()
