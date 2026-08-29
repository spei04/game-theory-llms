# Implementation Plan: Multi-Game LLM Game Theory Research Framework

## Research Objectives

**Core Question**: How do different LLMs play games?

**Specific Goals**:
1. Test LLMs in Donor Game
2. Manipulate contextual framing (heterogeneous population, frankenstein model)
3. Manipulate instruction framing (prompt variations)
4. Analyze three aspects: strategy generation, following, and evolution
5. Compare across different LLMs

ssh <user>@<cluster-host>
ssh <user>@<cluster-host>
source ~/.bashrc
git clone https://github.com/<user>/Game-Theory-with-LLMs /home/<user>/Game-Theory-with-LLMs
cd /home/<user>/Game-Theory-with-LLMs
git checkout serena

# Local: after making changes
git add -p   # stage what you want
git commit -m "your message"
git push origin serena

# Cluster: pull the latest
cd /home/<user>/Game-Theory-with-LLMs
git checkout serena
git pull origin serena
Don't forget .env on the cluster — it's gitignored, so you still need to create it manually once:

# SUBMITTING SCRIPTS
# 1. Default baseline — Claude, baseline prompt, no extras
sbatch slurm/run_baseline_array.sh

# fixed rounds
# trace: simple, 3_traces, 3_chain
3 chain Claude: 15919151
                GPT: 15937384
                gemini: 15937385

scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_gemini_fixed_15904092/ \
       <local-path>/Downloads/Game-Theory-with-LLMs/simulations/fixed_rounds/gemini

scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_gemini_hist-3_traces_15904098/ \
       <local-path>/Downloads/Game-Theory-with-LLMs/simulations/history_3traces/gemini

scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_gemini_hist-3_chain_15937385 \
       <local-path>/Downloads/Game-Theory-with-LLMs/simulations/history_3chain/gemini



# 8/18

# CODED STRATEGY FOR FULL GAME
for m in claude gpt gemini; do
  sbatch slurm/run_baseline_array.sh --model $m --ask_coded --generations 10
done


# direct evolution (full runs 20)
sbatch --array=1-20%5 slurm/run_baseline_array.sh --model gpt --direct_evolution --evo_beta 500

    ls -dt ~/Game-Theory-with-LLMs/simulations/array_* | head -1

    scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_gpt_new_21034350/ \
    <local-path>/Downloads/Game-Theory-with-LLMs/simulations/baseline_new/gpt/


coded strategies (run full)
JOB: sbatch slurm/run_baseline_array.sh --model gpt --ask_coded


JOB: sbatch slurm/run_baseline_array.sh --model gpt_new --ask_coded



<!-- # create text -> code
python figures/stratesgy_analysis.py simulations/baseline_new --output_dir figures/reactivity_plots/neutral_baseline -->

# plot
python3.10 figures/compare_askcoded_strategies.py --generation 1/10


    CLAUDE: 17996208
    RUNNING GEMINI: 18013974
    GPT: 17996197
    copy results: scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_gemini_askcoded_18013974 \
    <local-path>/Downloads/Game-Theory-with-LLMs/simulations/ask_coded/gemini

    view logs: tail -f /home/<user>/slurm_job/logs/18013974_1_baseline_array.log


# direct evolution (gens 2+ adopt surviving strategies verbatim, sampled from
# a softmax over survivor final scores: p_i = exp(v_i/beta)/sum_j exp(v_j/beta)
# — no LLM evolve prompt; --evo_beta sets the noise parameter beta)
sbatch slurm/run_baseline_array.sh --model claude --direct_evolution --evo_beta 100
    RUNNING
    ls -dt ~/Game-Theory-with-LLMs/simulations/array_* | head -3

    
    for m in claude gemini gpt; do echo "$m: $(ls simulations/direct_evo_001_old/$m/*Final_Accum*.json | wc -l)"; done

.venv/bin/python figures/plot_direct_evolution.py \
  --directevo_dir simulations/direct_evo_001_old --baseline_dir simulations/fixed_rounds \
  --model_suffix "" --subtitle "beta=0.01 (argmax), original models" \
  --output figures/direct_evolution_b001_vs_rulematched.png

    

# gen0 (run for all prompt variants)
sbatch --array=1-1 slurm/run_baseline_array.sh --variant aggressive --model gpt --strategies_only --repeats 200
    Claude: 18328181
    Gemini: 18328158
    GPT: 18328157

    scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_gemini_directevo-b100_20622669 \
    <local-path>/Downloads/Game-Theory-with-LLMs/simulations/direct_evo_100_old/claude



# 2. Prompt variant (different framings)
[baseline(default)/neutral_deepseek/neutral_claude/neutral_gpt/neutral_gemini/aggressive/collaborative]
sbatch slurm/run_baseline_array.sh --model claude --variant neutral_deepseek
    scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_gpt_prompt-neutral2_14411037/ \
       <local-path>/Downloads/Game-Theory-with-LLMs/simulations/neutral2

sbatch slurm/run_baseline_array.sh --model gemini --variant neutral_claude
    scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_gpt_prompt-neutral_claude_15301011/ \
       <local-path>/Downloads/Game-Theory-with-LLMs/simulations/neutral_claude

sbatch slurm/run_baseline_array.sh --model gpt --variant neutral_gpt
    claude: 15358625
    RUNNING gpt: 15442203
    gemini: 15358629
    scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_gpt_prompt-neutral_gpt_15442203/ \
       <local-path>/Downloads/Game-Theory-with-LLMs/simulations/neutral_gpt

sbatch slurm/run_baseline_array.sh --model gpt --variant neutral_gemini
    claude: 15334838
    RUNNING gpt: 15442179
    gemini: 15334878
    scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_gpt_prompt-neutral_gemini_15442179/ \
       <local-path>/Downloads/Game-Theory-with-LLMs/simulations/neutral_gemini

sbatch slurm/run_baseline_array.sh --model gpt --variant aggressive
    claude: 14298710
    gemini: 14411045
    gpt: 14524823
    scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_gpt_prompt-aggressive_14524823/ \
       <local-path>/Downloads/Game-Theory-with-LLMs/simulations/aggressive

sbatch slurm/run_baseline_array.sh --model gpt --variant collaborative
    claude: 14463927
    gemini: 14463930
    gpt: 14506164
    scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_gpt_prompt-collaborative_14506164/ \
       <local-path>/Downloads/Game-Theory-with-LLMs/simulations/collaborative

# Fixed initial strategies
sbatch slurm/run_baseline_array.sh --model gemini --fixed_initial
    GPT: 16801324
    Gemini: 16801339
    scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_gemini_fixed-init_16801339/ \
       <local-path>/Downloads/Game-Theory-with-LLMs/simulations/fixed_initial/gemini

scp slurm/run_baseline_array.sh slurm/submit_baseline_array.sh \
    <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/slurm/

# New models
Claude_new: 16309939
GPT_new: 16437752
Gemini_new: 16251348
sbatch slurm/run_baseline_array.sh --model gpt_new
scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_gpt_new_16437752/ \
       <local-path>/Downloads/Game-Theory-with-LLMs/simulations/baseline_new/gpt/

claude_new: 
gpt_new: 
gemini_new: 
sbatch slurm/run_baseline_array.sh --model claude_new --variant collaborative
sbatch slurm/run_baseline_array.sh --model gemini_new --history 3_chain

# 3. Heterogeneous population (start with 4/4/4) and use claude for translating strategies to code
sbatch slurm/run_baseline_array.sh --het random
15377179
scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_claude_het-random_15377179/ \
       <local-path>/Downloads/Game-Theory-with-LLMs/simulations/
    
    Get the model distribution at the end of the game (claude, gpt, gemini)
    """
    python3 << 'EOF'
    import json, glob, os
    from collections import Counter

    DIR = "/home/<user>/Game-Theory-with-LLMs/simulations/array_claude_het-random_15244328"
    files = sorted(glob.glob(f"{DIR}/all_strategies_*.json"))
    print(f"Analyzing {len(files)} runs from {DIR}\n")

    def short(m):
        m = (m or "").lower()
        if 'claude' in m: return 'claude'
        if 'gpt'    in m: return 'gpt'
        if 'gemini' in m: return 'gemini'
        return m or 'unknown'

    per_run_final = []
    agg_final = Counter()
    for f in files:
        data = json.load(open(f))
        # Find the last non-empty generation
        last = next((g for g in reversed(data) if g), None)
        if not last:
            continue
        c = Counter(short(a.get('model','')) for a in last)
        per_run_final.append((os.path.basename(f), c, last[0]['generation']))
        agg_final += c

    print(f"=== Final-generation totals across {len(per_run_final)} runs ===")
    total = sum(agg_final.values())
    for m in ('claude', 'gpt', 'gemini'):
        if agg_final[m]:
            print(f"  {m:<8} {agg_final[m]:>4d} / {total}  ({100*agg_final[m]/total:5.1f}%)")

    print(f"\n=== Per-run final composition (claude / gpt / gemini) at gen N ===")
    for fname, c, gen in per_run_final:
        tag = fname.replace('all_strategies_', '').replace('.json', '')[-15:]
        print(f"  {tag}  gen{gen:>2}:  {c['claude']:>2d} / {c['gpt']:>2d} / {c['gemini']:>2d}")
    EOF
    """

# 4. Frankenstein (different models per pipeline stage)
sbatch slurm/run_baseline_array.sh --frankenstein claude,gemini,gpt
sbatch slurm/run_baseline_array.sh --frankenstein gpt,claude,claude
sbatch slurm/run_baseline_array.sh --frankenstein claude,claude,gpt

# 5. Strategy mode (coded execution skips LLM during gameplay)
sbatch slurm/run_baseline_array.sh --model gpt --strategy coded

# 6. Ask-coded (agents write their strategy directly as a Python
#    `def strategy_fn(pair_prev_pct)` instead of text; no classifier pass,
#    code is executed locally during gameplay)
sbatch slurm/run_baseline_array.sh --model gpt --ask_coded

claude: 15377294
gpt: 15442282
scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_gpt_mode-coded_15442282/ \
       <local-path>/Downloads/Game-Theory-with-LLMs/simulations/coded_new_gpt


# Find recent array jobs
squeue -u <user>                       # currently queued/running
sacct -u <user> --format=JobID,JobName,Start,State -X | tail -20   # recent history

# Tail one task live
tail -f /home/<user>/slurm_job/logs/14222090_1_baseline_array.log

# All 20 tasks at once
tail -f /home/<user>/slurm_job/logs/<ARRAY_JOB_ID>_*_baseline_array.log


How to check variant: 
head -20 /home/<user>/slurm_job/logs/<ARRAY_JOB_ID>_1_baseline_array.log


# COPY RESULTS
scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/<DIR_NAME>/ \
       <local-path>/Downloads/Game-Theory-with-LLMs/simulations/

Run type	Source directory
Default baseline	array_claude_<jobid>
Prompt variant	array_claude_prompt-neutral2_<jobid>
Heterogeneous	array_claude_het-random_<jobid>
Frankenstein	array_frank-claude_gemini_gpt_<jobid>
Strategy-mode coded	array_claude_mode-coded_<jobid>

# get strategy comparison
cat simulations/array_*_<jobid>/gen1_vs_final_*.txt

## GRAPHS

## FIGS OVERLAY
# --base_dir can be either a flat folder of Final_Accum files, or a folder
# containing claude/, gpt/, gemini/ subfolders. Plots default to saving next to the data; pass --output_dir to put them elsewhere.

# Default neutral baseline
python figures/figs_overlay.py --base_dir simulations/baseline_new --max_runs 0

# Prompt variant (e.g. all 3 models with aggressive prompt merged into one dir)
python figures/figs_overlay.py --base_dir simulations/baseline_new --max_runs 0

# Compare two prompt variants
python figures/figs_overlay.py \
    --base_dir simulations/neutral_baseline \
    --extra_dirs simulations/fixed_initial \
    --max_runs 0

# Overlay a [frankenstein/heterogeneous] run on top of the 3-model baseline
python figures/figs_overlay.py \
    --base_dir simulations/neutral_baseline --max_runs 0 \
    --extra_dirs '<local-path>/Downloads/Game-Theory-with-LLMs/simulations/heterogeneous_random2'

# Multiple frankenstein combos at once
python figures/figs_overlay.py \
    --base_dir simulations/neutral_baseline --max_runs 0 \
    --extra_dirs <local-path>/Downloads/Game-Theory-with-LLMs/simulations/fixed_initial

# Explicit output dir (override the default)
python figures/figs_overlay.py \
    --base_dir simulations/neutral_baseline --max_runs 0 \
    --output_dir figures/overlay_figures/neutral_baseline

## STRATEGY ANALYSIS (mark fully specified and emit code, strategy stats)
# Folder is a positional arg (not --sim_dir)
# Run on each (variant, model) combo
python figures/strategy_analysis.py simulations/neutral_claude/gpt \
    --model gpt \
    --output_dir figures/reactivity_plots/neutral_claude


for m in gemini; do
    if [ -f figures/reactivity_plots/neutral_gemini/$m/strategy_analysis_$m.json ]; then
        echo "skip $m (already exists)"
        continue
    fi
    python figures/strategy_analysis.py simulations/neutral_gemini/$m \
        --model $m \
        --output_dir figures/reactivity_plots/neutral_gemini
done


## AREA UNDER STRATEGY CURVE
# Default — reads figures/reactivity_plots/gen1, writes figures/auc_stats.json + .png
python figures/compute_auc.py

# Different analysis dir
python figures/compute_auc.py \
    --analysis_dir figures/reactivity_plots/0424 \
    --output figures/auc_stats_0424.json


## STRATEGY ADHERENCE METRICS
python figures/plot_strategy_adherence.py \
    --analysis_dirs figures/reactivity_plots/neutral_baseline \
                    figures/reactivity_plots/neutral_deepseek \
                    figures/reactivity_plots/neutral_claude \
                    figures/reactivity_plots/neutral_gpt \
                    figures/reactivity_plots/neutral_gemini \
    --output_dir figures/strategy_groups/all_variants


## PRINT STRATEGIES
 python figures/compare_strategy_complexity.py --model claude --run_idx 0



## STRATEGY GROUPS
# Two batches side-by-side
python figures/plot_strategy_groups.py \
    --dir1 simulations/0417 \
    --dir2 simulations/0424 \
    --analysis_dir1 figures/reactivity_plots/0417 \
    --analysis_dir2 figures/reactivity_plots/0424 \
    --output_dir figures/strategy_groups

# Same directory for both if you only care about one batch
python figures/plot_strategy_groups.py \
    --dir1 simulations/neutral_baseline \
    --dir2 simulations/neutral_baseline \
    --output_dir figures/strategy_groups


## TRACE STRATEGY LINEAGE (new strategy vs. rank 1 strategy)
# Claude in batch 0424, run 0, gen 10 (defaults)
python figures/trace_strategy_lineage.py

# Specific config
python figures/trace_strategy_lineage.py \
    --model claude \
    --batch 0424 \
    --run_idx 0 \
    --target_gen 10 \
    --output_dir figures/strategy_inheritance


## COMPARE INHERITED STRATEGY
python figures/comparing_inherited_strategy.py \
    --generation 2 \
    --model claude \
    --sim_dir simulations/new_baseline \
    --analysis_dir figures/reactivity_plots/gen1 \
    --output_dir figures/strategy_inherit_comparison


## EXAMPLE PIPELINE
# 1. Copy from cluster
scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_*_<jobid>/ \
       simulations/my_run/

# 2. Run strategy_analysis on each model (needed for adherence + groups plots)
for m in claude gpt gemini; do
    python figures/strategy_analysis.py simulations/my_run --model $m \
        --output_dir figures/reactivity_plots/my_run
done

# 3. Plot overlays (resources + donations curves)
python figures/figs_overlay.py --base_dir simulations/my_run \
    --output_dir figures/overlay_figures/my_run --max_runs 0

# 4. Optionally also plot AUC, adherence, groups, lineage as needed
python figures/compute_auc.py --analysis_dir figures/reactivity_plots/my_run \
    --output figures/auc_stats_my_run.json
