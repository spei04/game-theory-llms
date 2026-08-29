# Codebase Summary: Game Theory with LLMs

## Overview

This repo simulates multi-generation iterated **Donor Games** where LLM agents play against each other, evolve strategies, and are selected based on performance. The architecture has two main simulation engines and a set of analysis/visualization scripts.

---

## Directory Structure

```
Game-Theory-with-LLMs/
├── .env                          # API keys (gitignored)
├── .venv/                        # Python virtual environment
├── requirements.txt              # Dependencies
│
├── simulations/
│   ├── donor_game_new.py         # PRIMARY simulation engine
│   ├── donor_game_old.py         # Legacy version (reference only)
│   ├── fixed_endowment.py        # Variant: fixed endowment per round
│   ├── strategy_response.py      # Test: single strategy vs. donation levels
│   └── prompt_strategy_combos.py # Test: 5 system prompt variants
│
├── modular system/
│   ├── gen_fol_ev.py             # Modular engine: separate models per role
│   ├── first_test.py             # Quickstart test for modular engine
│   └── run_model_combinations.py # Systematic model combination runner (⚠️ placeholder data)
│
└── figures/
    ├── figs_avg_grouped.py        # Average donation rate across generations
    ├── figs_compare_modes.py      # Compare simple/3_traces/full history modes
    ├── figs_fixed_endowment.py    # Fixed endowment specific plots
    ├── figs_individual_runs.py    # Per-run plots + average line (multi-model)
    ├── figs_overlay.py            # Overlay multiple runs
    ├── plot_strategy_combos.py    # Heatmaps for prompt strategy combos
    └── prompt_heatmaps.py         # Prompt variant heatmaps
```

---

## The Game Mechanics

All simulations run the same **Donor Game**:
- Agents start with `initial_endowment` (default: **10**) units
- Each round: randomly paired into donor/recipient
- Donor gives 0–100% of their resources; recipient gets `cooperationGain`× (default: **2×**) what was given
- Donor/recipient roles alternate each round
- After all rounds: **top 50%** survive and pass strategies to new agents

---

## Supported Models

All models require the corresponding key in `.env`.

| Model string | Routes to | Provider |
|---|---|---|
| `"gpt-4o"` | GPT-4o | OpenAI |
| `"claude-3-5-sonnet"` | claude-sonnet-4-5-20250929 | Anthropic |
| `"claude-3-sonnet"` | claude-sonnet-4-20250514 | Anthropic |
| `"claude-3-opus"` | claude-opus-4-20250514 | Anthropic |
| `"claude-3-haiku"` | claude-haiku-4-5-20251001 | Anthropic |
| `"claude-sonnet-4-6"` | claude-sonnet-4-6 | Anthropic |
| `"gemini-1.5-flash"` | gemini-1.5-flash | Google |
| `"llama-3.3-70b-versatile"` | Llama 3.3 70B | Groq |
| `"llama-3.1-8b-instant"` | Llama 3.1 8B | Groq |
| `"deepseek-r1-distill-llama-70b"` | DeepSeek R1 | Groq |
| `"mixtral-8x7b-32768"` | Mixtral MoE | Groq |
| `"gemma2-9b-it"` | Gemma 2 9B | Groq |
| `"qwen-qwq-32b"` | Qwen QwQ 32B | Groq |
| any other string | Ollama (local) | local |

---

## Experiment 1: Main Donor Game Simulation

**File**: `simulations/donor_game_new.py`

**What it does**: Full multi-generation donor game. All agents use the same model. Saves JSON to `simulation_results/`.

**Key parameters** (bottom of file, lines 1217–1234):

| Parameter | Default | Options |
|---|---|---|
| `model_str` | `"claude-3-5-sonnet"` | any model from table above |
| `numGenerations` | `10` | any int |
| `numAgents` | `12` | even int |
| `initial_endowment` | `10` | any int |
| `number_of_rounds` | `12` | any int |
| `history_mode` | `"3_traces"` | `"simple"`, `"3_traces"`, `"full"` |
| `selection_method` | `"top"` | `"top"`, `"random"`, `"imageScore"`, `"reputation"` |
| `punishment_mechanism` | `"none"` | `"none"`, `"costly_punishment"`, `"partner_choice"` |

**Run**:
```bash
.venv/bin/python simulations/donor_game_new.py
```

**To use a different model**: edit `model_str` at line 157 (or uncomment one of the commented options above it).

**Output**: `simulation_results/simulation_<model>_<timestamp>.json` and `simulation_results/all_strategies_<model>_<timestamp>.txt`

---

## Experiment 2: Modular Multi-Model Simulation

**File**: `modular system/gen_fol_ev.py`

**What it does**: Same donor game but with **three separate models** for three distinct roles:
- `generate_strategy` — model that writes the agent's strategy each generation
- `follow_strategy` — model that decides how much to donate each round
- `evolution` — model that evaluates and selects surviving strategies

**Key parameters** (top of file, lines 59–63):
```python
MODEL_CONFIG = {
    "generate_strategy": "claude-3-5-sonnet",
    "follow_strategy": "claude-3-5-sonnet",
    "evolution": "claude-3-5-sonnet"
}
```

**Run**:
```bash
.venv/bin/python "modular system/gen_fol_ev.py"
```

The `__main__` block runs the simulation with whatever `MODEL_CONFIG` is set. Uncomment one of the example `set_model_combination()` calls at lines 1121–1122 to test mixed models, e.g.:
```python
set_model_combination("gpt-4o", "claude-3-5-sonnet", "gemini-1.5-flash")
```

---

## Experiment 3: Quick Modular Test

**File**: `modular system/first_test.py`

**What it does**: Minimal sanity check — 3 generations, 6 agents, all `claude-3-5-sonnet`. Good for verifying the modular engine works before a big run.

**Run**:
```bash
cd "modular system" && ../.venv/bin/python first_test.py
```

**Output**: JSON in `simulation_results/`

---

## Experiment 4: Fixed Endowment Variant

**File**: `simulations/fixed_endowment.py`

**What it does**: Each round, agents receive a fresh fixed endowment (instead of accumulating resources). Default model: `gpt-4o`.

**Run**:
```bash
.venv/bin/python simulations/fixed_endowment.py
```

---

## Experiment 5: Strategy Response Probe

**File**: `simulations/strategy_response.py`

**What it does**: Tests a single strategy against recipients who donate different fixed amounts (e.g., 0%, 25%, 50%, 75%, 100%). Measures how the strategy responds to each. Default model: `gemini-1.5-flash`, temperature=0.

**Run**:
```bash
.venv/bin/python simulations/strategy_response.py
```

---

## Experiment 6: Prompt Strategy Combinations

**File**: `simulations/prompt_strategy_combos.py`

**What it does**: Runs the same game with 5 different system prompt variants to measure how prompt framing affects behavior. Default model: `claude-3-5-sonnet`.

**Run**:
```bash
.venv/bin/python simulations/prompt_strategy_combos.py
```

---

## Experiment 7: Model Combinations Sweep

**File**: `modular system/run_model_combinations.py`

**What it does**: Systematically runs many generate/follow/evolution model combinations and generates comparison plots.

> ⚠️ **Known issue**: The `run_model_combination()` function currently uses **placeholder random data** instead of calling `runGenerations_modular()`. The analysis/plots work, but the numbers are fake until this is fixed.

**Run**:
```bash
cd "modular system" && ../.venv/bin/python run_model_combinations.py
```

---

## Analysis Scripts (figures/)

All scripts in `figures/` read JSON files produced by the simulation engines. Run them from the repo root.

| Script | Input | What it plots |
|---|---|---|
| `figs_avg_grouped.py` | `simulation_results/*.json` | Average donation % per generation, grouped |
| `figs_individual_runs.py` | `simulation_results/*.json` | Per-run trajectories + average line, 3 models × 2 metrics |
| `figs_overlay.py` | `simulation_results/*.json` | Overlay multiple runs on same axes |
| `figs_compare_modes.py` | 3 JSON files (one per history mode) | Bar chart comparing `simple`/`3_traces`/`full` at gen 10 |
| `figs_fixed_endowment.py` | `simulation_results/*.json` | Fixed endowment specific plots |
| `plot_strategy_combos.py` | Prompt combo JSON | Heatmap of prompt strategy results |
| `prompt_heatmaps.py` | Prompt combo JSON | Alternative prompt heatmaps |

**Example run**:
```bash
.venv/bin/python figures/figs_avg_grouped.py
.venv/bin/python figures/figs_individual_runs.py
```

---

## Quick Reference: All Run Commands

```bash
# --- Core simulations ---
.venv/bin/python simulations/donor_game_new.py           # Main experiment
.venv/bin/python simulations/fixed_endowment.py          # Fixed endowment variant
.venv/bin/python simulations/strategy_response.py        # Strategy probe
.venv/bin/python simulations/prompt_strategy_combos.py   # Prompt variants

# --- Modular (multi-model) ---
.venv/bin/python "modular system/gen_fol_ev.py"                        # Modular engine
cd "modular system" && ../.venv/bin/python first_test.py               # Quick sanity check
cd "modular system" && ../.venv/bin/python run_model_combinations.py   # ⚠️ uses placeholder data

# --- Analysis (after simulations have run) ---
.venv/bin/python figures/figs_avg_grouped.py
.venv/bin/python figures/figs_individual_runs.py
.venv/bin/python figures/figs_overlay.py
.venv/bin/python figures/figs_compare_modes.py
```

---

## To Switch Models

- **donor_game_new.py**: change `model_str` at line 157
- **gen_fol_ev.py**: change `MODEL_CONFIG` at lines 59–63
- **fixed_endowment.py / strategy_response.py / prompt_strategy_combos.py**: change `llm` variable near top of file

---

## Simulation → Figure Script Mapping

### `donor_game_new.py` → 4 figure scripts

Output files: `Final_Accum_*.json`

| Figure script | What it shows | When useful |
|---|---|---|
| `figs_avg_grouped.py` | Mean resources + donation % per generation, with ±stderr band | Multiple runs per model (5+ recommended) |
| `figs_individual_runs.py` | Every run as a faint line + black dotted average | Multiple runs — good for seeing variance |
| `figs_overlay.py` | Models overlaid on the same axes | Comparing 2–3 models directly |
| `figs_compare_modes.py` | Bar chart at gen 10: `simple` vs `3_traces` vs `full` history | Need 3 separate sets of runs, one per `history_mode` |

### `fixed_endowment.py` → 1 figure script

Output files: `Final_Single_Trace_*.json`

| Figure script | What it shows |
|---|---|
| `figs_fixed_endowment.py` | Resources/donation plots for fixed-endowment runs |

### `prompt_strategy_combos.py` → 2 figure scripts

Output files: `prompt_strategy_*.json`

| Figure script | What it shows |
|---|---|
| `plot_strategy_combos.py` | How each strategy responds to different recipient donation levels |
| `prompt_heatmaps.py` | Heatmaps: strategy × prompt-style MSE (how well agents followed their stated strategy) |

### `gen_fol_ev.py` → same as `donor_game_new.py`

Produces the same JSON format, so `figs_avg_grouped`, `figs_individual_runs`, and `figs_overlay` all work with it.

### `strategy_response.py` → no dedicated figure script

No matching script in `figures/`. Requires custom analysis.

---

### Summary Table

| Simulation script | Output file prefix | Figure scripts |
|---|---|---|
| `donor_game_new.py` | `Final_Accum_*` | `figs_avg_grouped`, `figs_individual_runs`, `figs_overlay`, `figs_compare_modes` |
| `fixed_endowment.py` | `Final_Single_Trace_*` | `figs_fixed_endowment` |
| `prompt_strategy_combos.py` | `prompt_strategy_*` | `plot_strategy_combos`, `prompt_heatmaps` |
| `gen_fol_ev.py` | `Final_Accum_*` | `figs_avg_grouped`, `figs_individual_runs`, `figs_overlay` |
| `strategy_response.py` | — | *(no matching script)* |

### How to Plot

All figure scripts have **hardcoded file paths** in their `__main__` block — they won't auto-discover files. Edit the `model_files` dict at the bottom of each script to point to your result files, then run from the repo root:

```bash
.venv/bin/python figures/figs_avg_grouped.py
.venv/bin/python figures/figs_individual_runs.py
.venv/bin/python figures/figs_overlay.py
.venv/bin/python figures/figs_compare_modes.py
.venv/bin/python figures/figs_fixed_endowment.py
.venv/bin/python figures/plot_strategy_combos.py
.venv/bin/python figures/prompt_heatmaps.py
```

Example `model_files` entry:
```python
model_files = {
    'Claude 3.5 Sonnet': [
        'simulation_results/Final_Accum_claude-3-5-sonnet_..._gen2_agents6_top_20260223_162834.json',
        # add more runs of the same model here to average across runs
    ],
}
```

12 agents, 12 rounds