#!/usr/bin/env python3
"""
Convert strategies-only run outputs (text strategies, no games) into
strategy_analysis-style JSONs by classifying/compiling each strategy with the
same Claude classifier prompt as figures/strategy_analysis.py — but via the
Message Batches API (50% token pricing).

Input : simulations/gen0_strategies/{model}/all_strategies_*.json
Output: figures/reactivity_plots/gen0_strategies/{model}/strategy_analysis_{model}.json
        (agent_records compatible with all existing gen-0 plotting scripts)

State (in the output dir):
  _strategy_code_cache.json   sha1(strategy) -> {strategy, is_specified, explanation, strategy_code}
  _batch_state.json           list of submitted batch ids + pending custom_ids

Usage:
    python figures/code_gen0_strategies.py submit    # build + submit batches for uncached strategies
    python figures/code_gen0_strategies.py status    # check batch progress
    python figures/code_gen0_strategies.py collect   # fetch results, update cache, write analysis JSONs

The classifier model is claude-sonnet-4-6 — the same model used to code the
existing 7-variant analysis JSONs, kept for comparability.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys

SIM_DIR = "simulations/gen0_strategies"
OUT_DIR = "figures/reactivity_plots/gen0_strategies"
CACHE_PATH = os.path.join(OUT_DIR, "_strategy_code_cache.json")
STATE_PATH = os.path.join(OUT_DIR, "_batch_state.json")


def set_dirs(sim_dir, out_dir):
    """Point the pipeline at a different variant's folders (see --sim_dir)."""
    global SIM_DIR, OUT_DIR, CACHE_PATH, STATE_PATH
    SIM_DIR = sim_dir
    OUT_DIR = out_dir
    CACHE_PATH = os.path.join(OUT_DIR, "_strategy_code_cache.json")
    STATE_PATH = os.path.join(OUT_DIR, "_batch_state.json")

MODELS = ["claude", "gpt", "gemini"]
CLASSIFIER_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 600
BATCH_CHUNK = 2500          # requests per batch (well under API limits)
ENDOWMENT = 10.0

_TS_RE = re.compile(r'_(\d{8}_\d{6})\.json$')


def load_env_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY="):
                    os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break


def sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def classifier_prompt(strategy_text):
    """Identical to the prompt in figures/strategy_analysis.py."""
    return (
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


def parse_response(response_text):
    """Identical parsing to figures/strategy_analysis.py."""
    is_specified = bool(re.search(r'FULLY_SPECIFIED:\s*yes', response_text, re.IGNORECASE))
    explanation = ""
    m = re.search(r'EXPLANATION:\s*(.*?)(?:\nCODE:|$)', response_text, re.DOTALL)
    if m:
        explanation = m.group(1).strip()
    code = None
    m = re.search(r'CODE:\s*\n(.*)', response_text, re.DOTALL)
    if m:
        code_str = m.group(1).strip()
        code_str = re.sub(r'^```[a-z]*\n?', '', code_str)
        code_str = re.sub(r'\n?```\s*$', '', code_str).strip()
        if code_str and code_str.upper() != "NONE" and "def strategy_fn" in code_str:
            code = code_str
    return is_specified, code, explanation


def load_records():
    """All gen-1 records: {model: [(run_id, agent_name, strategy), ...]}."""
    out = {}
    for model in MODELS:
        recs = []
        for path in sorted(glob.glob(os.path.join(SIM_DIR, model, "all_strategies_*.json"))):
            m = _TS_RE.search(os.path.basename(path))
            run_id = m.group(1) if m else os.path.basename(path)
            for gen_list in json.load(open(path)):
                for r in gen_list:
                    if r.get("generation") == 1 and r.get("strategy"):
                        recs.append((run_id, r.get("agent_name", ""), r["strategy"]))
        out[model] = recs
        print(f"  [{model}] {len(recs)} gen-1 strategies from "
              f"{len(set(r[0] for r in recs))} runs")
    return out


def load_json(path, default):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return default


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def cmd_submit():
    load_env_key()
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()
    records = load_records()
    cache = load_json(CACHE_PATH, {})

    todo = {}   # sha1 -> strategy text
    for model_recs in records.values():
        for _, _, strategy in model_recs:
            h = sha1(strategy)
            if h not in cache:
                todo[h] = strategy
    print(f"\nUnique strategies: to classify {len(todo)}, already cached {len(cache)}")
    if not todo:
        print("Nothing to submit — run `collect` to write analysis JSONs.")
        return

    items = sorted(todo.items())
    batch_ids = []
    for i in range(0, len(items), BATCH_CHUNK):
        chunk = items[i:i + BATCH_CHUNK]
        requests = [
            Request(
                custom_id=h,
                params=MessageCreateParamsNonStreaming(
                    model=CLASSIFIER_MODEL,
                    max_tokens=MAX_TOKENS,
                    messages=[{"role": "user", "content": classifier_prompt(s)}],
                ),
            )
            for h, s in chunk
        ]
        batch = client.messages.batches.create(requests=requests)
        batch_ids.append(batch.id)
        print(f"  submitted batch {batch.id} ({len(chunk)} requests)")

    save_json(STATE_PATH, {
        "batch_ids": batch_ids,
        "pending": {h: s for h, s in items},
        "classifier_model": CLASSIFIER_MODEL,
    })
    print(f"\nState saved: {STATE_PATH}")


def cmd_status():
    load_env_key()
    import anthropic
    client = anthropic.Anthropic()
    state = load_json(STATE_PATH, None)
    if not state:
        print("No batch state — run submit first.")
        return 1
    all_ended = True
    for bid in state["batch_ids"]:
        b = client.messages.batches.retrieve(bid)
        c = b.request_counts
        print(f"  {bid}: {b.processing_status}  "
              f"(ok={c.succeeded} err={c.errored} processing={c.processing})")
        if b.processing_status != "ended":
            all_ended = False
    return 0 if all_ended else 2


def cmd_collect():
    load_env_key()
    import anthropic
    client = anthropic.Anthropic()
    state = load_json(STATE_PATH, None)
    cache = load_json(CACHE_PATH, {})

    if state:
        pending = state.get("pending", {})
        n_err = 0
        for bid in state["batch_ids"]:
            b = client.messages.batches.retrieve(bid)
            if b.processing_status != "ended":
                print(f"  {bid} not finished yet ({b.processing_status}) — try later.")
                return 2
            for result in client.messages.batches.results(bid):
                h = result.custom_id
                if result.result.type == "succeeded":
                    msg = result.result.message
                    text = next((blk.text for blk in msg.content if blk.type == "text"), "")
                    is_specified, code, explanation = parse_response(text)
                    cache[h] = {
                        "strategy": pending.get(h, ""),
                        "is_specified": is_specified,
                        "strategy_code": code,
                        "explanation": explanation,
                    }
                else:
                    n_err += 1
        save_json(CACHE_PATH, cache)
        print(f"Cache now has {len(cache)} strategies ({n_err} batch results errored)")

    # Write per-model strategy_analysis JSONs
    records = load_records()
    for model in MODELS:
        agent_records, n_missing = [], 0
        for run_id, agent_name, strategy in records[model]:
            entry = cache.get(sha1(strategy))
            if entry is None:
                n_missing += 1
                continue
            agent_records.append({
                "run": run_id,
                "generation": 1,
                "agent": agent_name,
                "strategy": strategy,
                "is_specified": entry["is_specified"],
                "explanation": entry["explanation"],
                "strategy_code": entry["strategy_code"],
            })
        out_path = os.path.join(OUT_DIR, model, f"strategy_analysis_{model}.json")
        save_json(out_path, {
            "summary": {
                "source": "simulations/gen0_strategies (strategies-only runs, neutral baseline prompt)",
                "classifier_model": CLASSIFIER_MODEL,
                "n_records": len(agent_records),
                "n_missing_from_cache": n_missing,
            },
            "agent_records": agent_records,
        })
        n_spec = sum(1 for r in agent_records if r["is_specified"] and r["strategy_code"])
        print(f"  [{model}] wrote {len(agent_records)} records "
              f"({n_spec} coded, {n_missing} missing) -> {out_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Batch text->code for gen0_strategies runs.")
    parser.add_argument("command", choices=["submit", "status", "collect"])
    parser.add_argument("--sim_dir", default=SIM_DIR,
                        help="Input dir with {model}/all_strategies_*.json "
                             "(e.g. simulations/gen0_strategies_neutral_deepseek)")
    parser.add_argument("--out_dir", default=None,
                        help="Output dir (default: figures/reactivity_plots/<basename of sim_dir>)")
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join("figures/reactivity_plots",
                                           os.path.basename(args.sim_dir.rstrip("/")))
    set_dirs(args.sim_dir, out_dir)
    rc = {"submit": cmd_submit, "status": cmd_status, "collect": cmd_collect}[args.command]()
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
