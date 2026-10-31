#!/usr/bin/env bash
# Aggregate results after all array tasks complete.
#
# Submit this *after* the array job, specifying the array job ID as a
# dependency so it only runs once every task has finished successfully:
#
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> \
#          slurm/run_aggregate.sh simulations/array_<ARRAY_JOB_ID>
#
# Or use the wrapper submit_baseline_array.sh which handles both steps.
#
# Argument:
#   $1 — output directory that was used for the array job (required)

#SBATCH -o /home/USERNAME/slurm_job/logs/%j_aggregate.log
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --partition=gpu_partition_name
#SBATCH --qos=normal
#SBATCH --account=slurm_account_name
#SBATCH --cpus-per-task=4

OUTPUT_DIR="${1:?Usage: sbatch run_aggregate.sh <output_dir>}"

REPO=/home/USERNAME/Game-Theory-with-LLMs

source /etc/profile.d/modules.sh
module load miniforge/24.3.0-0
source /path/to/cluster/miniforge/etc/profile.d/conda.sh
conda activate myconda

PYTHON=/home/USERNAME/.conda/envs/myconda/bin/python
export LD_LIBRARY_PATH=/home/USERNAME/.conda/envs/myconda/lib:$LD_LIBRARY_PATH
set -a; source $REPO/.env; set +a
cd $REPO

echo "Aggregation job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "Output directory: $OUTPUT_DIR"
echo ""

# ---------------------------------------------------------------------------
# 1. Count result files so we know what landed
# ---------------------------------------------------------------------------
N_ACCUM=$(find "$OUTPUT_DIR" -maxdepth 1 -name "Final_Accum_*.json" | wc -l)
N_STRAT=$(find "$OUTPUT_DIR" -maxdepth 1 -name "all_strategies_*.json" | wc -l)
echo "Files found: $N_ACCUM Final_Accum files, $N_STRAT all_strategies files"

# ---------------------------------------------------------------------------
# 2. Run strategy analysis (is_specified + coded strategies)
#    One call per model found in the output directory.
# ---------------------------------------------------------------------------
echo ""
echo "=== Running strategy analysis ==="
for MODEL in claude gpt gemini; do
    N=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*Final_Accum*${MODEL}*.json" | wc -l)
    if [ "$N" -gt 0 ]; then
        echo "  Analysing $MODEL ($N runs)..."
        $PYTHON -u figures/strategy_analysis.py "$OUTPUT_DIR" \
            --model "$MODEL" \
            --output_dir "$OUTPUT_DIR/strategy_analysis/${MODEL}"
    fi
done

echo ""
echo "Aggregation complete at $(date)"
echo "Results: $OUTPUT_DIR"
