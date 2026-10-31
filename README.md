# Prompts, Models, or Luck? Decomposing Outcome Variability in LLM Cultural-Evolution Simulations

Code for the paper. It runs an evolutionary Donor Game with LLM agents, adapted from Vallinder and Hughes ([arXiv:2412.10270](https://arxiv.org/abs/2412.10270)), and contains the analysis script behind every figure and table. Simulation outputs are not included; the scripts expect them under `simulations/` (see [simulations/README.md](simulations/README.md)).

## Layout

| Path | Contents |
|---|---|
| `slurm/experiments/donor_game_baseline.py` | The simulation. All game-rule prompts are defined here and selected with `PROMPT_VARIANT`. |
| `slurm/run_baseline_array.sh`, `submit_baseline_array.sh`, `run_aggregate.sh` | SLURM launchers: a 20-run array job per configuration, plus aggregation. |
| `slurm/setup_vertex_project.sh` | Optional: bill Gemini calls through Vertex AI instead of an API key. |
| `figures/` | Analysis and figure scripts (mapped to the paper below). |

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repository root:

```bash
OPENAI_API_KEY=...      # GPT-4o agents
ANTHROPIC_API_KEY=...   # Claude agents, and the text-to-code step
GOOGLE_API_KEY=...      # Gemini agents (or use Vertex AI, see slurm/setup_vertex_project.sh)
```

## Models

The launcher's short names map to these API models (decoding settings are in the paper's appendix):

| Short name | Label stored in run files | API model |
|---|---|---|
| `claude` | `claude-3-5-sonnet` | `claude-sonnet-4-5-20250929` |
| `gpt` | `gpt-4o` | `gpt-4o` |
| `gemini` | `gemini-1.5-flash` | `gemini-2.5-flash-lite` |

The stored labels are kept from an earlier version of the code; `donor_game_baseline.py` maps each label to the API model in the last column.

## Running the simulations

Every dataset has 20 runs per model of 12 agents, 12 rounds per game and 10 generations. On a SLURM cluster, edit the `#SBATCH` lines and paths at the top of `slurm/run_baseline_array.sh`, then submit one array job per model:

```bash
sbatch slurm/run_baseline_array.sh --model claude --variant neutral_baseline
```

| Dataset (`simulations/<name>/`) | Flags |
|---|---|
| `neutral_baseline` | `--variant neutral_baseline` |
| `neutral_claude`, `neutral_gpt`, `neutral_gemini`, `neutral_deepseek` | `--variant neutral_claude` (and so on) |
| `direct_evo_50_neutral` | `--variant neutral_baseline --direct_evolution --evo_beta 50` |
| `direct_evo_{001,1,5,50,100,500}_old` | `--variant baseline --direct_evolution --evo_beta {0.01,1,5,50,100,500}` |

Each array job writes to `simulations/array_<jobid>/`; move its `Final_Accum_*.json` and `all_strategies_*.json` files into `simulations/<name>/<model>/`.

Without SLURM, set the same environment variables and run one simulation at a time:

```bash
MODEL_STR=claude-3-5-sonnet PROMPT_VARIANT=neutral_baseline COST_CAP=50 \
OUTPUT_DIR=simulations/neutral_baseline/claude \
python slurm/experiments/donor_game_baseline.py
```

Add `DIRECT_EVOLUTION=1 DIRECT_EVOLUTION_BETA=50` for direct evolution.

## Coding strategies as response curves

Strategies are free text. Claude classifies each one and, if it is fully specified, converts it to a Python function of the partner's last donation (paper appendix, "Text-to-code conversion"):

```bash
# each cultural-evolution dataset
python figures/strategy_analysis.py simulations/neutral_baseline \
    --output_dir figures/reactivity_plots/neutral_baseline

# direct-evolution runs (batch API; run `collect` once the batch finishes)
python figures/code_gen0_strategies.py submit --sim_dir simulations/direct_evo_50_neutral \
    --out_dir figures/reactivity_plots/direct_evo_50_neutral
python figures/code_gen0_strategies.py collect --sim_dir simulations/direct_evo_50_neutral \
    --out_dir figures/reactivity_plots/direct_evo_50_neutral
```

`strategy_analysis.py` also records the check that coded strategies reproduce the agents' actual donations (`summary.fully_specified_adherence` in each `strategy_analysis_<model>.json`).

## Reproducing the paper

Run from the repository root, after the simulations and the coding step above.

| Paper item | Command |
|---|---|
| Figure 1, response curves at Generations 1 and 10 | `python figures/plot_gen0_reactivity_all_variants.py --variants neutral_baseline neutral_claude neutral_gpt neutral_gemini neutral_deepseek --paper --font_scale 1.6` (add `--generation 10`) |
| Figure 2, game setup | `python figures/fig_setup_schematic.py` |
| Outcome progression, baseline prompt | `python figures/plot_avg_resources_by_gen.py --experiments neutral_baseline --runs` |
| Appendix trajectories, LLM-written prompts | `python figures/plot_avg_resources_by_gen.py --experiments neutral_claude neutral_deepseek neutral_gemini neutral_gpt --runs --grid llm_neutral_prompt_trajectories` |
| Variance figure, cultural vs direct evolution | `python figures/plot_variance_diversity_summary.py` |
| Run feature table (input to the next three rows) | `python figures/build_run_feature_table.py` |
| Variance shares table and balanced-subsample check | `python figures/appendix_variance_stats.py` |
| Prediction tables | `python figures/appendix_prediction_spec.py` |
| What the run component holds | `python figures/appendix_residual_sources.py` |
| L1 norm distance table | `python figures/strategy_curve_spread.py` |
| Newcomer distance to the closest survivor | `python figures/strategy_closest_survivor.py` |
| Distinct strategies, cooperation by rule, model-share interval, coded coverage, direct-evolution variance table | `python figures/rule_comparison_stats.py` |
| Text-to-code retention table | `python figures/report_retention.py` |

Each script's docstring lists its inputs and outputs. The helpers `compute_auc.py`, `compare_gen0_variants.py` and `variance_decomposition.py` are imported by the scripts above.
