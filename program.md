# Research Program: LLM Cooperation in Evolutionary Donor Games

## Overview

This repo studies whether LLMs can develop cooperative strategies through cultural evolution in iterated Donor Games. Agents use LLMs to generate and refine strategies across generations. The best-performing agents survive to teach the next generation.

**Core question:** Under what game conditions do LLMs spontaneously evolve cooperation?

## Research Goal

Discover experiment configurations that maximize:
1. **High final-generation donation rates** — `avg_donation_pct` in the last generation (higher = more cooperation)
2. **Increasing trend** — cooperation growing across generations (evolution is working)
3. **Robustness** — low invalid response rate (model is functioning)

## Success Metric

```
score = avg_donation_percentage of the final generation  (0–100%, higher = better)
```

A result is **KEPT** if:
- `score ≥ baseline_score + 3.0` (at least 3 percentage points above baseline), AND
- `invalid_response_rate < 15%` of total pairings

Otherwise it is **DISCARDED** (but still documented).

## Parameter Space

| Env Var | Default | Options | Hypothesis |
|---------|---------|---------|------------|
| `HISTORY_MODE` | `3_traces` | `simple`, `3_traces`, `full` | More info → better reciprocity |
| `COOP_GAIN` | `2.0` | `1.5`, `2.0`, `2.5`, `3.0`, `4.0` | Higher gain → stronger incentive to cooperate |
| `PUNISHMENT_MECHANISM` | `none` | `none`, `costly_punishment` | Punishment → deters defectors |
| `SELECTION_METHOD` | `top` | `top`, `reputation`, `random` | Reputation selection → rewards cooperators directly |
| `INITIAL_ENDOWMENT` | `10` | `5`, `10`, `20` | Lower endowment → higher cooperation pressure |
| `PUNISHMENT_LOSS` | `2.0` | `1.5`, `2.0`, `3.0` | (Only relevant when punishment active) |

**Fixed for all experiments** (do not vary): `NUM_GENERATIONS=10`, `NUM_AGENTS=12`, `NUM_ROUNDS=12`, model.

## Experiment Workflow

```
1. python research_agent.py baseline           → check current best score
2. Read experiments/log.md                     → what has been tried? what worked?
3. Choose ONE unexplored parameter combination with a clear, falsifiable hypothesis
4. python research_agent.py run --hypothesis "..." [--history MODE] [--coop-gain N] ...
5. python research_agent.py compare <result_json>   → KEEP or DISCARD
6. Document:
   - KEPT:     python research_agent.py keep    <result_json> --hypothesis "..." --note "..."
   - DISCARDED: python research_agent.py discard <result_json> --hypothesis "..." --note "..."
7. Repeat
```

## Autonomous Loop

```bash
# Run N experiments without human involvement
python research_agent.py loop --n 10
```

The loop uses Claude to plan each experiment based on `program.md` and the current `experiments/log.md`, runs it, evaluates, and documents — all automatically.

## Manual Commands

```bash
python research_agent.py baseline                        # Show/compute baseline score
python research_agent.py score   <result.json>           # Score a result file
python research_agent.py compare <result.json>           # KEEP or DISCARD vs baseline
python research_agent.py keep    <result.json> -H "..." -n "..."
python research_agent.py discard <result.json> -H "..." -n "..."
python research_agent.py run     [--history MODE] [--coop-gain N] [--punishment TYPE]
                                 [--selection METHOD] [--endowment N] [--model STR]
                                 --hypothesis "..."
python research_agent.py loop    --n 10 [--model STR]
```

## What Makes a Scientifically Interesting Result

Beyond "score > baseline", prioritize:
- Cooperation **increases across generations** (evolutionary dynamics working)
- Strategies **converge** to a clear cooperative norm (emergent coordination)
- Large **effect size** (not just marginally above threshold)
- Result **confirms or refutes a clean mechanism hypothesis**

## Constraints

- **Do not modify** `simulations/donor_game_new.py` logic, Agent classes, or output format
- **Do not modify** `program.md` (that's the human researcher's domain)
- Document ALL attempts in `experiments/log.md`, not just successful ones
- Never delete result files, even from discarded experiments
- Default model: `gemini-2.5-flash-lite` (cheap and fast for iteration)
