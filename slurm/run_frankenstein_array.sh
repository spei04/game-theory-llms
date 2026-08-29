#!/usr/bin/env bash
# Frankenstein experiment: one model generates strategies, another evolves them,
# another executes them in-game. Submitted as a SLURM array job (20 runs per
# role-combination).
#
# Usage:
#   sbatch slurm/run_frankenstein_array.sh <generator> <evolver> <executor>
#
# Examples:
#   sbatch slurm/run_frankenstein_array.sh claude gemini gpt
#   sbatch slurm/run_frankenstein_array.sh gpt claude claude
#
# Each role argument accepts: claude | gpt | gemini  (or a full model string).

#SBATCH -o /home/USERNAME/slurm_job/logs/%A_%a_frankenstein_array.log
#SBATCH --mem=32GB
#SBATCH --time=6:00:00
#SBATCH --partition=gpu_partition_name
#SBATCH --qos=normal
#SBATCH --account=slurm_account_name
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=1-20%5

REPO=/home/USERNAME/Game-Theory-with-LLMs
SCRIPT=$REPO/slurm/experiments/donor_game_frankenstein.py

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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Simulation parameters ───────────────────────────────────────────────
export NUM_GENERATIONS=10
export NUM_AGENTS=12
export NUM_ROUNDS=12
export COST_CAP=15   # higher than baseline since multiple models may be billed

# ── Role → model mapping ────────────────────────────────────────────────
short_to_full() {
    case "$1" in
        claude) echo "claude-3-5-sonnet" ;;
        gpt)    echo "gpt-4o" ;;
        gemini) echo "gemini-1.5-flash" ;;
        *)      echo "$1" ;;
    esac
}

GEN_ARG="${1:?Usage: sbatch run_frankenstein_array.sh <generator> <evolver> <executor>}"
EVO_ARG="${2:?Usage: sbatch run_frankenstein_array.sh <generator> <evolver> <executor>}"
EXE_ARG="${3:?Usage: sbatch run_frankenstein_array.sh <generator> <evolver> <executor>}"

export GENERATOR_MODEL=$(short_to_full "$GEN_ARG")
export EVOLVER_MODEL=$(short_to_full   "$EVO_ARG")
export EXECUTOR_MODEL=$(short_to_full  "$EXE_ARG")
# MODEL_STR is still used as a fallback for any helper that reads it directly.
export MODEL_STR="$EXECUTOR_MODEL"

# ── Output directory: tag the combination so multiple submissions don't collide
TAG="gen-${GEN_ARG}_evo-${EVO_ARG}_exe-${EXE_ARG}"
export OUTPUT_DIR=$REPO/simulations/frankenstein_${TAG}_${SLURM_ARRAY_JOB_ID}
mkdir -p "$OUTPUT_DIR"

export RUN_ID=$SLURM_ARRAY_TASK_ID

echo "Array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID} starting on $(hostname) at $(date)"
echo "Generator: $GENERATOR_MODEL"
echo "Evolver  : $EVOLVER_MODEL"
echo "Executor : $EXECUTOR_MODEL"
echo "Output   : $OUTPUT_DIR"

$PYTHON -u "$SCRIPT"

echo "Task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
