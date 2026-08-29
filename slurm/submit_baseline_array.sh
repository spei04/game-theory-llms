#!/usr/bin/env bash
# Submit the 20-run array job, then queue an aggregation job dependent on it.
# Forwards all flags through to run_baseline_array.sh so any combination of
# model / prompt-variant / heterogeneous / strategy-mode / frankenstein works.
#
# Usage:
#   bash slurm/submit_baseline_array.sh [flags]
#
# Flags (same as run_baseline_array.sh):
#   --model <name>        claude | gpt | gemini  (default: claude)
#   --variant <name>      baseline | neutral2 | aggressive | collaborative  (default: baseline)
#   --het <mode>          random | replacement | claude | gpt | gemini      (default: off)
#   --strategy <mode>     llm | coded                                        (default: llm)
#   --frankenstein g,e,x  per-role models, comma-separated (overrides --model)
#
# Examples:
#   bash slurm/submit_baseline_array.sh
#   bash slurm/submit_baseline_array.sh --model gpt --variant aggressive
#   bash slurm/submit_baseline_array.sh --het random
#   bash slurm/submit_baseline_array.sh --strategy coded
#   bash slurm/submit_baseline_array.sh --frankenstein claude,gemini,gpt
#   bash slurm/submit_baseline_array.sh --frankenstein claude,gpt,gpt \
#        --variant aggressive --strategy coded

# Forward every argument to the array script; sbatch ignores them but the
# script body parses them out via its own --flag handling.
echo "Submitting baseline array job: $@"

# --fixed_initial → 10 runs sharing a single set of gen-1 strategies.
# Override the array spec at submission time (overrides the #SBATCH default).
SBATCH_ARRAY_OPT=""
for a in "$@"; do
    if [ "$a" = "--fixed_initial" ]; then
        SBATCH_ARRAY_OPT="--array=1-10"
        echo "  (fixed_initial: overriding array spec → 1-10)"
        break
    fi
done

ARRAY_JOB_ID=$(sbatch --parsable $SBATCH_ARRAY_OPT slurm/run_baseline_array.sh "$@")
echo "  Array job ID: $ARRAY_JOB_ID"

# Reconstruct the same output-dir tag the array script computes so the
# aggregation step knows where to look. (Mirrors logic in run_baseline_array.sh.)
MODEL="claude"; VARIANT="baseline"; HET=""; SMODE="llm"; FRANK=""; HMODE=""; FIXED=""; FIXED_INIT=""
ARGS=("$@")
i=0
while [ $i -lt ${#ARGS[@]} ]; do
    case "${ARGS[$i]}" in
        --model)         MODEL="${ARGS[$((i+1))]}";   i=$((i+2)) ;;
        --variant)       VARIANT="${ARGS[$((i+1))]}"; i=$((i+2)) ;;
        --het)           HET="${ARGS[$((i+1))]}";     i=$((i+2)) ;;
        --strategy)      SMODE="${ARGS[$((i+1))]}";   i=$((i+2)) ;;
        --frankenstein)  FRANK="${ARGS[$((i+1))]}";   i=$((i+2)) ;;
        --history)       HMODE="${ARGS[$((i+1))]}";   i=$((i+2)) ;;
        --fixed_rounds)  FIXED="1";                   i=$((i+1)) ;;
        --fixed_initial) FIXED_INIT="1";              i=$((i+1)) ;;
        *) i=$((i+1)) ;;
    esac
done
TAG=""
if [ -n "$FRANK" ]; then
    TAG="${TAG}_frank-${FRANK//,/_}"
else
    TAG="${TAG}_${MODEL}"
fi
[ "$VARIANT" != "baseline" ] && TAG="${TAG}_prompt-${VARIANT}"
[ -n "$HET" ]               && TAG="${TAG}_het-${HET}"
[ "$SMODE" != "llm" ]       && TAG="${TAG}_mode-${SMODE}"
[ -n "$HMODE" ]             && TAG="${TAG}_hist-${HMODE}"
[ -n "$FIXED" ]             && TAG="${TAG}_fixed"
[ -n "$FIXED_INIT" ]        && TAG="${TAG}_fixed-init"
OUTPUT_DIR="simulations/array${TAG}_${ARRAY_JOB_ID}"

echo ""
echo "Submitting aggregation job (depends on $ARRAY_JOB_ID)..."
AGG_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${ARRAY_JOB_ID} \
    slurm/run_aggregate.sh "$OUTPUT_DIR")
echo "  Aggregation job ID: $AGG_JOB_ID"

echo ""
echo "Done. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f /home/USERNAME/slurm_job/logs/${ARRAY_JOB_ID}_<task>_baseline_array.log"
echo "  tail -f /home/USERNAME/slurm_job/logs/${AGG_JOB_ID}_aggregate.log"
echo ""
echo "Output will be written to: /home/USERNAME/Game-Theory-with-LLMs/$OUTPUT_DIR"
