# Simulations Directory

This directory hosts the core simulation scripts and a record of baseline run outputs. The main execution path for the current repo is the SLURM baseline experiment in `slurm/experiments/donor_game_baseline.py`, with plotting handled by scripts in `figures/`.

## Game Summary (Iterated Donor Game)

Agents start each generation with an initial endowment. In each round, agents are paired; one is the donor and decides how many units to give, and the recipient receives a multiplied amount. Roles alternate across rounds. After all rounds in a generation, agents are evaluated by their final resources (or other selection method), the top half survive, and new agents are initialized for the next generation with inherited strategies and limited history based on the chosen history mode.

## Baseline Experiment (Current Focus)

**Primary script:** `slurm/experiments/donor_game_baseline.py`

**What it does**
- Runs the iterated donor game with cultural evolution across generations.
- Supports many model backends:
  - API: OpenAI, Anthropic, Google Gemini
  - Local / OSS: HuggingFace (transformers), vLLM, llama-cpp, AWQ, bitsandbytes
  - Groq and HuggingFace Inference API (OpenAI-compatible routers)
- Tracks token usage and estimated cost with a configurable cost cap.
- Saves raw per-agent data plus compact per-generation stats (used by plotting).

**Key environment variables**
- `MODEL_STR`: model name (e.g., `gpt-4o`, `claude-3-5-sonnet`, `gemini-1.5-flash`, `Qwen/Qwen2.5-7B-Instruct`)
- `RUN_ID`: integer run index (used to name checkpoints)
- `NUM_GENERATIONS`, `NUM_AGENTS`, `NUM_ROUNDS`, `INITIAL_ENDOWMENT`
- `COOP_GAIN`, `PUNISHMENT_LOSS`, `PUNISHMENT_MECHANISM`
- `SELECTION_METHOD`: `top`, `random`, `imageScore`, `reputation`
- `HISTORY_MODE`: `simple`, `3_traces`, `full`
- `COST_CAP`: USD cap per run

**Outputs** (relative to the working directory)
- `simulation_results/Final_Accum_*_YYYYMMDD_HHMMSS.json`: main results
- `simulation_results/all_strategies_*_YYYYMMDD_HHMMSS.txt`: full strategies by generation
- `simulation_results/checkpoint_baseline_*_runX.json`: resumable checkpoints

Notes on output content:
- `generation_summaries` is intentionally not saved in baseline runs.
- `generation_stats` is saved and is the preferred source for plotting.

**SLURM launcher**
- `slurm/experiments/run_baseline.sh` runs multiple models and multiple runs per model, setting `MODEL_STR`, `RUN_ID`, and other parameters.

**Baseline data location in this repo**
- Curated outputs are stored under `simulations/baseline/simulation_results/`.

## Other Simulation Scripts

These are still in the repo but are not the primary baseline path right now:
- `simulations/donor_game_old.py`: original baseline implementation.
- `simulations/donor_game_new.py`: earlier extended version with additional history modes and saving logic.
- `simulations/fixed_endowment.py`: variant with fixed per-round endowment.
- `simulations/strategy_response.py`: strategy adherence tests.
- `simulations/prompt_strategy_combos.py`: prompt/strategy combination testing.

## Figure Code (figures/)

The plotting scripts consume `simulation_results/*.json` files. Most expect `generation_stats` or raw `agents_data` and will compute averages if summaries are missing.

- `figures/figs_avg_grouped.py`: average resources and donation percentages over generations for multiple runs of one model.
- `figures/figs_overlay.py`: overlay multiple models with mean and standard error bands. You explicitly list the files to plot; it reads `generation_stats` first and falls back to raw `agents_data` if needed.
- `figures/figs_fixed_endowment.py`: plotting tailored to fixed-endowment runs.
- `figures/figs_compare_modes.py`: compares end-of-run donation percentages across game modes.
- `figures/figs_individual_runs.py`: plots individual runs plus averages.
- `figures/plot_strategy_combos.py`: strategy/prompt combo analysis plots.
- `figures/prompt_heatmaps.py`: heatmaps for strategy adherence and prompt effects.

Outputs are written to `figures/overlay_figures/` or other subfolders depending on the script.
