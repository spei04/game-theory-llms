#!/usr/bin/env bash
# 20-run array job for donor_game_baseline.py.
# All experiment options are exposed via named flags so any combination —
# baseline, prompt-variant, heterogeneous, strategy_mode, frankenstein — can
# be launched from a single script.
#
# Usage:
#   sbatch slurm/run_baseline_array.sh [flags]
#
# Flags (all optional, all default to "off / baseline / claude"):
#   --model <name>        single-model run. Short: claude | gpt | gemini, or full string
#   --variant <name>      prompt variant: baseline | neutral_baseline | neutral_deepseek | neutral_claude |
#                         neutral_gpt | neutral_gemini | aggressive | collaborative
#                         (legacy alias: neutral2 = neutral_deepseek)
#   --het <mode>          heterogeneous population: random | replacement | claude | gpt | gemini
#   --strategy <mode>     strategy execution mode: llm | coded
#   --ask_coded           agents return their strategy entirely as Python code
#                         (def strategy_fn(pair_prev_pct)) instead of text;
#                         the code is executed locally during gameplay
#   --direct_evolution    generations 2+ skip the LLM evolve prompt: each new
#                         agent adopts a surviving strategy verbatim, sampled
#                         from a softmax over survivor final scores:
#                         p_i = exp(v_i/beta) / sum_j exp(v_j/beta)
#   --evo_beta <x>        softmax noise parameter beta for --direct_evolution
#                         (default 1.0; small = copy the best, large = uniform)
#   --frankenstein g,e,x  Frankenstein roles, comma-separated: generator,evolver,executor
#                         e.g. --frankenstein claude,gemini,gpt
#                         Overrides --model for the three pipeline stages.
#   --generations <n>     number of generations (default: 10)
#   --resume_dir <path>   reuse an existing simulations/array_* directory instead
#                         of creating a new one. Tasks find the checkpoint left
#                         by a previous run with the same RUN_ID and continue
#                         from the last completed generation. Use to finish runs
#                         that died on the cost cap or the wall clock.
#   --workers <n>         concurrent LLM calls per task (default 4). A round has
#                         numAgents/2 donor pairs (6 at 12 agents), so 6 lets a
#                         whole round run at once — the cheapest way to cut wall
#                         time for slow models like gpt-5.
#   --cost_cap <usd>      max estimated USD per task before the run aborts
#                         (default: 50). Raise for gpt_new / claude_new — a
#                         10-gen gpt-5 run measured $49.88, i.e. right at the
#                         default cap; a run that trips it dies with no results.
#   --strategies_only     generate + record the 12 gen-1 strategies, then exit —
#                         no gameplay, no Final_Accum; writes all_strategies_*.json
#   --repeats <n>         run the Python script n times sequentially within each
#                         array task (default: 1). Combine with --strategies_only
#                         for many cheap runs without hitting the QOS job-submit
#                         limit, e.g. 200 sequential strategy-only runs in ONE job:
#                         sbatch --array=1-1 slurm/run_baseline_array.sh --model claude --strategies_only --repeats 200
#
# To override the SLURM array spec (default: 1-20%5), pass --array to sbatch
# itself, BEFORE the script name — it is not a script flag:
#   sbatch --array=1-1 slurm/run_baseline_array.sh --model gemini --ask_coded
#
# Examples:
#   # Default Claude baseline
#   sbatch slurm/run_baseline_array.sh
#
#   # GPT-4o with aggressive prompt
#   sbatch slurm/run_baseline_array.sh --model gpt --variant aggressive
#
#   # Claude baseline, heterogeneous random population
#   sbatch slurm/run_baseline_array.sh --het random
#
#   # Coded strategy mode (LLM bypassed during gameplay)
#   sbatch slurm/run_baseline_array.sh --strategy coded
#
#   # Agents write their strategy directly as code (no classifier pass)
#   sbatch slurm/run_baseline_array.sh --model gpt --ask_coded
#
#   # Softmax strategy adoption instead of LLM evolution
#   sbatch slurm/run_baseline_array.sh --model gpt --direct_evolution --evo_beta 50
#
#   # Frankenstein: claude writes strategy, gemini evolves, gpt plays
#   sbatch slurm/run_baseline_array.sh --frankenstein claude,gemini,gpt
#
#   # Combined: Frankenstein + coded mode + aggressive prompt
#   sbatch slurm/run_baseline_array.sh --frankenstein claude,claude,gpt \
#          --strategy coded --variant aggressive

#SBATCH -o /home/USERNAME/slurm_job/logs/%A_%a_baseline_array.log
#SBATCH --mem=32GB
#SBATCH --time=6:00:00
#SBATCH --partition=gpu_partition_name
#SBATCH --qos=normal
#SBATCH --account=slurm_account_name
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=1-20%5

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
MODEL="claude"
VARIANT="baseline"
HET=""
SMODE="llm"
ASK_CODED="" # truthy = agents return strategies as Python code instead of text
DIRECT_EVO="" # truthy = new agents adopt surviving strategies via softmax over scores (no LLM evolve)
EVO_BETA="1.0" # softmax noise parameter beta (only used with --direct_evolution)
FRANK=""    # "gen,evo,exe" or empty
HMODE=""    # "simple" | "3_traces" | "full" — empty = use Python default
FIXED_ROUNDS=""  # truthy = strip the "80% chance" probabilistic-rounds text
FIXED_INITIAL="" # truthy = share gen-1 strategies across all array tasks
GENERATIONS="10"
COSTCAP="50"
WORKERS=""
RESUME_DIR=""
STRATEGIES_ONLY="" # truthy = record gen-1 strategies and exit (no gameplay)
REPEATS="1"        # sequential python invocations per array task

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)         MODEL="$2";   shift 2 ;;
        --variant)       VARIANT="$2"; shift 2 ;;
        --het)           HET="$2";     shift 2 ;;
        --strategy)      SMODE="$2";   shift 2 ;;
        --ask_coded)     ASK_CODED="1"; shift 1 ;;
        --direct_evolution) DIRECT_EVO="1"; shift 1 ;;
        --evo_beta)      EVO_BETA="$2"; shift 2 ;;
        --frankenstein)  FRANK="$2";   shift 2 ;;
        --history)       HMODE="$2";   shift 2 ;;
        --fixed_rounds)  FIXED_ROUNDS="1"; shift 1 ;;
        --fixed_initial) FIXED_INITIAL="1"; shift 1 ;;
        --generations)   GENERATIONS="$2"; shift 2 ;;
        --cost_cap)      COSTCAP="$2"; shift 2 ;;
        --workers)       WORKERS="$2"; shift 2 ;;
        --resume_dir)    RESUME_DIR="$2"; shift 2 ;;
        --strategies_only) STRATEGIES_ONLY="1"; shift 1 ;;
        --repeats)       REPEATS="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
REPO=/home/USERNAME/Game-Theory-with-LLMs
SCRIPT=$REPO/slurm/experiments/donor_game_baseline.py

source /etc/profile.d/modules.sh
module load miniforge/24.3.0-0
source /path/to/cluster/miniforge/etc/profile.d/conda.sh
conda activate myconda

PYTHON=/home/USERNAME/.conda/envs/myconda/bin/python
export LD_LIBRARY_PATH=/home/USERNAME/.conda/envs/myconda/lib:$LD_LIBRARY_PATH
set -a; source $REPO/.env; set +a
cd $REPO

if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGINGFACE_TOKEN:-}" ]; then
    export HF_TOKEN="$HUGGINGFACE_TOKEN"
fi

# the cluster outbound HTTPS proxy — required for compute nodes to reach the
# Anthropic / OpenAI / Google APIs. Compute nodes do not inherit the proxy
# from the submitter's environment unless we export it here explicitly.
export HTTP_PROXY=http://proxy.cluster.example.edu:3128
export HTTPS_PROXY=http://proxy.cluster.example.edu:3128
export http_proxy=http://proxy.cluster.example.edu:3128
export https_proxy=http://proxy.cluster.example.edu:3128
export NO_PROXY=localhost,127.0.0.1,.cluster.example.edu
export no_proxy=localhost,127.0.0.1,.cluster.example.edu

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
export NUM_GENERATIONS="$GENERATIONS"
export NUM_AGENTS=12
export NUM_ROUNDS=12
export COST_CAP="$COSTCAP"   # max USD per task (override with --cost_cap)
[ -n "$WORKERS" ] && export MAX_WORKERS="$WORKERS"

# ---------------------------------------------------------------------------
# Resolve short model names → full strings
# ---------------------------------------------------------------------------
resolve_model() {
    case "$1" in
        claude)      echo "claude-3-5-sonnet" ;;
        gpt)         echo "gpt-4o" ;;
        gemini)      echo "gemini-1.5-flash" ;;
        # *_new short names → latest available model in each family.
        # Update the right-hand side here when newer model IDs become available.
        claude_new)  echo "claude-new" ;;
        gpt_new)     echo "gpt-new" ;;
        gemini_new)  echo "gemini-new" ;;
        *)           echo "$1" ;;
    esac
}

export MODEL_STR=$(resolve_model "$MODEL")
export PROMPT_VARIANT="$VARIANT"
export HETEROGENEOUS="$HET"
export STRATEGY_MODE="$SMODE"
[ -n "$ASK_CODED" ]    && export ASK_CODED=1
[ -n "$DIRECT_EVO" ]   && export DIRECT_EVOLUTION=1
[ -n "$DIRECT_EVO" ]   && export DIRECT_EVOLUTION_BETA="$EVO_BETA"
[ -n "$HMODE" ]        && export HISTORY_MODE="$HMODE"
[ -n "$FIXED_ROUNDS" ] && export FIXED_ROUNDS="$FIXED_ROUNDS"
[ -n "$STRATEGIES_ONLY" ] && export STRATEGIES_ONLY=1

# When --fixed_initial is set, point the python script at a shared JSON file
# scoped to this array job. Task 1 generates and saves; tasks 2..N load from it
# (fcntl lock makes this race-safe even with parallel array tasks).
if [ -n "$FIXED_INITIAL" ]; then
    export FIXED_INITIAL_PATH="$REPO/simulations/fixed_initial_${SLURM_ARRAY_JOB_ID}.json"
fi

# Frankenstein: parse "gen,evo,exe" into three env vars
if [ -n "$FRANK" ]; then
    IFS=',' read -r FRANK_GEN FRANK_EVO FRANK_EXE <<< "$FRANK"
    export GENERATOR_MODEL=$(resolve_model "$FRANK_GEN")
    export EVOLVER_MODEL=$(resolve_model "$FRANK_EVO")
    export EXECUTOR_MODEL=$(resolve_model "$FRANK_EXE")
fi

# ---------------------------------------------------------------------------
# Output directory tag — include enough flags that parallel submissions don't
# collide. Always anchored to the array job ID so re-runs get fresh dirs.
# ---------------------------------------------------------------------------
TAG=""
if [ -n "$FRANK" ]; then
    TAG="${TAG}_frank-${FRANK//,/_}"
else
    TAG="${TAG}_${MODEL}"
fi
[ "$VARIANT" != "baseline" ] && TAG="${TAG}_prompt-${VARIANT}"
[ -n "$HET" ]               && TAG="${TAG}_het-${HET}"
[ "$SMODE" != "llm" ]       && TAG="${TAG}_mode-${SMODE}"
[ -n "$ASK_CODED" ]         && TAG="${TAG}_askcoded"
[ -n "$DIRECT_EVO" ]        && TAG="${TAG}_directevo-b${EVO_BETA}"
[ -n "$HMODE" ]             && TAG="${TAG}_hist-${HMODE}"
[ -n "$FIXED_ROUNDS" ]      && TAG="${TAG}_fixed"
[ -n "$FIXED_INITIAL" ]     && TAG="${TAG}_fixed-init"
[ -n "$STRATEGIES_ONLY" ]   && TAG="${TAG}_stratonly"

if [ -n "$RESUME_DIR" ]; then
    export OUTPUT_DIR="$RESUME_DIR"
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "ERROR: --resume_dir does not exist: $OUTPUT_DIR" >&2; exit 1
    fi
    echo "Resuming into existing directory: $OUTPUT_DIR"
else
    export OUTPUT_DIR=$REPO/simulations/array${TAG}_${SLURM_ARRAY_JOB_ID}
fi
mkdir -p "$OUTPUT_DIR"

export RUN_ID=$SLURM_ARRAY_TASK_ID

echo "===================================================================="
echo "Array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID} on $(hostname) at $(date)"
echo "  model            : $MODEL_STR"
echo "  prompt_variant   : $PROMPT_VARIANT"
echo "  heterogeneous    : ${HETEROGENEOUS:-off}"
echo "  strategy_mode    : $STRATEGY_MODE"
echo "  ask_coded        : ${ASK_CODED:-off}"
echo "  direct_evolution : $([ -n "$DIRECT_EVO" ] && echo "on (beta=$EVO_BETA)" || echo off)"
if [ -n "$FRANK" ]; then
echo "  frankenstein     : gen=$GENERATOR_MODEL  evo=$EVOLVER_MODEL  exe=$EXECUTOR_MODEL"
fi
echo "  output_dir       : $OUTPUT_DIR"
echo "===================================================================="

# Defensive unset right before Python — some clusters re-set proxy env vars
# via /etc/profile.d/ scripts that get sourced by `module load` / `conda
# activate`. This ensures the proxy is gone at the exact moment Python starts.
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy

# ── DEBUG block — verify env right before Python runs.
# Remove this once the job is reliably succeeding.
echo "=== JOB DEBUG ==="
echo "  hostname      : $(hostname)"
echo "  HTTP_PROXY    =$HTTP_PROXY"
echo "  HTTPS_PROXY   =$HTTPS_PROXY"
echo "  http_proxy    =$http_proxy"
echo "  https_proxy   =$https_proxy"
echo "  PYTHON        : $PYTHON"
$PYTHON --version
echo "  --- in-process anthropic API test ---"
$PYTHON -c "
import os
for k in ['HTTP_PROXY','HTTPS_PROXY','http_proxy','https_proxy']:
    print(f'  py-env {k}={os.environ.get(k)!r}')
from dotenv import load_dotenv
load_dotenv('/home/USERNAME/Game-Theory-with-LLMs/.env')
import anthropic
try:
    c = anthropic.Anthropic()
    m = c.messages.create(model='claude-opus-4-8', max_tokens=10,
                          messages=[{'role':'user','content':'hi'}])
    print('  api test: OK ->', m.content[0].text[:60])
except Exception as e:
    print('  api test: FAIL ->', type(e).__name__, str(e)[:200])
"
echo "=== END DEBUG ==="

if [ "$REPEATS" -gt 1 ]; then
    # Sequential repeats within one task — each gets a unique RUN_ID so
    # checkpoints and all_strategies filenames never collide.
    for rep in $(seq 1 "$REPEATS"); do
        export RUN_ID="${SLURM_ARRAY_TASK_ID}_${rep}"
        echo "--- repeat ${rep}/${REPEATS} (RUN_ID=${RUN_ID}) at $(date) ---"
        $PYTHON -u "$SCRIPT"
    done
else
    $PYTHON -u "$SCRIPT"
fi

echo "Task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
