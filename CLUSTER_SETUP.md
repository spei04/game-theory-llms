1. Push your current branch from Mac
git push origin serena

2. On the cluster — clone the repo and check out the branch

ssh <user>@<cluster-host>
git clone https://github.com/<user>/Game-Theory-with-LLMs /home/<user>/Game-Theory-with-LLMs
cd /home/<user>/Game-Theory-with-LLMs
git checkout serena
3. For future updates — push from Mac, pull on cluster


# Mac: after making changes
git add -p   # stage what you want
git commit -m "your message"
git push origin serena

# Cluster: pull the latest
cd /home/<user>/Game-Theory-with-LLMs
git checkout serena
git pull origin serena
4. Don't forget .env on the cluster — it's gitignored, so you still need to create it manually once:

python 3.9.4 <-- most recent
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Create .env with your API keys
nano .env

# paste
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
GROQ_API_KEY=gsk_...
HUGGINGFACE_TOKEN=hf_...

# Run on the login node
mkdir -p /home/<user>/lib
cp /usr/lib64/libssl.so.10 /home/<user>/lib/
cp /usr/lib64/libcrypto.so.10 /home/<user>/lib/   # also required by ssl


# submit job
sbatch slurm/donor_game.sh

# monitor
squeue -u <user>
tail -f /home/<user>/slurm_job/logs/9816102.log

## Predownloading models
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
  --force-reinstall

python3 -c "
from llama_cpp import Llama
Llama.from_pretrained(
    repo_id='unsloth/Qwen3.5-27B-GGUF',
    filename='Q4_K_M/Qwen3.5-27B-Q4_K_M.gguf',
    n_ctx=4096,
    verbose=True,
)
print('Download complete')
"

## CUDA setup
module load cuda/11.3
export LD_LIBRARY_PATH=/home/software/cuda/11.3/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/home/software/cuda/11.3

## miniforge conda setup
module load miniforge
conda create -n myconda python=3.10 -y
conda activate myconda
conda install -y numpy pandas scipy matplotlib seaborn
pip install openai anthropic google-genai groq python-dotenv "urllib3<2"
module load gcc/11.2.0
module load cmake/3.17.3
module load cuda/12.0
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir

## the cluster CLUSTER
sed -i '/^export PYTORCH_CUDA_ALLOC_CONF/i export PYTHONNOUSERSITE=1' \
    /home/<user>/Game-Theory-with-LLMs/slurm/run_baseline_array.sh

grep -E "PYTHONNOUSERSITE|^PYTHON=" /home/<user>/Game-Theory-with-LLMs/slurm/run_baseline_array.sh


# Make sure the SLURM script still points at the conda env's Python
grep "^PYTHON=" /home/<user>/Game-Theory-with-LLMs/slurm/run_baseline_array.sh

# And the unset is still there
grep "unset HTTP_PROXY\|PYTHONNOUSERSITE" /home/<user>/Game-Theory-with-LLMs/slurm/run_baseline_array.sh



## STEPS TO RUN SIMULATION (baseline)
ssh <user>@<cluster-host>, ssh <user>@<cluster-host>
source ~/.bashrc
sbatch slurm/submit_baseline_array.sh gpt super_neutral
tail -f /home/<user>/slurm_job/logs/13529832_1_baseline_array.log
tail -f /home/<user>/slurm_job/logs/13529921_1_baseline_array.log

13710809 claude
13710876 gpt
13710845 gemini
scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_13529921/ <local-path>/Downloads/Game-Theory-with-LLMs/simulations/new_baseline/

# clean up folder
rm -f /home/<user>/Game-Theory-with-LLMs/simulations/array_13529832/*.{json,txt,log}

# frankenstein
sbatch /home/<user>/Game-Theory-with-LLMs/slurm/run_frankenstein_array.sh claude gemini gpt

scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/frankenstein_gen-claude_evo-gemini_exe-gpt_14078342 \
       <local-path>/Downloads/Game-Theory-with-LLMs/simulations/

sbatch /home/<user>/Game-Theory-with-LLMs/slurm/run_frankenstein_array.sh gemini gpt claude

scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/frankenstein_gen-gemini_evo-gpt_exe-claude_14078715 \
       <local-path>/Downloads/Game-Theory-with-LLMs/simulations/

sbatch /home/<user>/Game-Theory-with-LLMs/slurm/run_frankenstein_array.sh gpt claude gemini 

scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/frankenstein_gen-gpt_evo-claude_exe-gemini_14078767 \
       <local-path>/Downloads/Game-Theory-with-LLMs/simulations/


# heterogeneous population
sbatch slurm/run_baseline_array.sh claude baseline random
sbatch slurm/run_baseline_array.sh claude baseline replacement
sbatch slurm/run_baseline_array.sh claude baseline claude

tail -f /home/<user>/slurm_job/logs/<ARRAY_JOB_ID>_*_baseline_array.log

# copy over results
# mode: random, replacement, claude
scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_14109276 \
       <local-path>/Downloads/Game-Theory-with-LLMs/simulations/heterogeneous_random/


## Graph plot once simulation done

# all models donations over generations
python3 figures/figs_overlay.py

# frankenstein added
python figures/figs_overlay.py \
    --base_dir simulations/neutral_baseline --max_runs 0 \
    --frankenstein_dirs simulations/frankenstein_gen-gemini_evo-gpt_exe-claude_14078715 \
    simulations/frankenstein_gen-claude_evo-gemini_exe-gpt_<otherjobid>

# per model agent vs. pair donation plot
# run for gen 1 and gen 10
python3 figures/plot_agent_donations.py \
  simulations/new_baseline \
  --model claude \
  --generation 10 \
  --output_dir figures/reactivity_plots/gen10 \
  --cost_cap 10

# strategy analysis
python figures/strategy_analysis.py simulations/new_baseline \
    --output_dir figures/reactivity_plots/gen1

# all models aggregated code strategy plot
python figures/plot_coded_strategies.py \
  --base_dir figures/reactivity_plots/gen10 \
  --generation 10 \
  --output_dir figures/coded_strats

# strategy inheritance stats
python figures/analyze_strategy_inheritance.py --model gpt

python -c "import json; print(len(json.load(open('<local-path>/Downloads/Game-Theory-with-LLMs/figures/reactivity_plots/claude-3-5-sonnet_coopGain_2.0punLoss_2.0_simple_gen10_agents12_top/gen1_underspecified_20260409_183517.json'))))"

# strategy inheritance comparison with rank 1
python figures/comparing_inherited_strategy.py \
        --generation 2 \
        --model gpt \
        [--sim_dir simulations/new_baseline] \
        [--analysis_dir figures/reactivity_plots/gen1] \
        [--output_dir figures/strategy_inherit_comparison]

# strategy group analysis
python figures/plot_strategy_groups.py \
        [--dir_0417 simulations/0417] \
        [--dir_0424 simulations/0424] \
        [--output_dir figures/strategy_groups]

# copy results
scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulation_results/ \
    <local-path>/Downloads/Game-Theory-with-LLMs/simulations/new_baseline

# clean up folder
rm -f /home/<user>/Game-Theory-with-LLMs/simulation_results/*.{json,txt,log}


# setup apptainer
ssh <user>@<cluster-host>
module load apptainer/1.4.2

srun --jobid=11445571 --overlap python3 /home/<user>/Game-Theory-with-LLMs/qwen35_apptainer_vllm018_minimal/probe_in_job.py \
  --host 127.0.0.1 \
  --port 8000 \
  --model Qwen/Qwen3.5-9B \
  --prompt "Reply with exactly: hello"

  tail -f /home/<user>/Game-Theory-with-LLMs/qwen35_apptainer_vllm018_minimal/logs/qwen35-serve-appt-11445571.out





  # BASELINE
ssh <user>@<cluster-host>
source ~/.bashrc
sbatch slurm/submit_baseline_array.sh claude/gpt/gemini
tail -f /home/<user>/slurm_job/logs/13529832_1_baseline_array.log

scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_13529921/ <local-path>/Downloads/Game-Theory-with-LLMs/simulations/new_baseline/

# CLEAN UP FOLDER
rm -f /home/<user>/Game-Theory-with-LLMs/simulations/array_13529832/*.{json,txt,log}


# PROMPT VARIANT
sbatch slurm/submit_baseline_array.sh gpt neutral2

[baseline/neutral2/aggressive/collaborative]
claude neutral2: 14167696
gemini neutral2: 14167729
gpt neutral2: 14167738

# View logs
tail -f /home/<user>/slurm_job/logs/14167729_*_baseline_array.log


# copy results
mkdir -p <local-path>/Downloads/Game-Theory-with-LLMs/simulations/neutral2

# Copy each model's array dir; the trailing slash + dot copies contents only
scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_14167696/. \
        <local-path>/Downloads/Game-Theory-with-LLMs/simulations/neutral2/
scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_14167738/. \
        <local-path>/Downloads/Game-Theory-with-LLMs/simulations/neutral2/
scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_14167729/. \
        <local-path>/Downloads/Game-Theory-with-LLMs/simulations/neutral2/


# FRANKENSTEIN
sbatch /home/<user>/Game-Theory-with-LLMs/slurm/run_frankenstein_array.sh claude gemini gpt

tail -f /home/<user>/slurm_job/logs/14109058_14_frankenstein_array.log
tail -f /home/<user>/slurm_job/logs/14166629_1_frankenstein_array.log


scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/frankenstein_gen-claude_evo-gemini_exe-gpt_14126877 \
       <local-path>/Downloads/Game-Theory-with-LLMs/simulations/

sbatch /home/<user>/Game-Theory-with-LLMs/slurm/run_frankenstein_array.sh gemini gpt claude

scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/frankenstein_gen-gemini_evo-gpt_exe-claude_14109090 \
       <local-path>/Downloads/Game-Theory-with-LLMs/simulations/

sbatch /home/<user>/Game-Theory-with-LLMs/slurm/run_frankenstein_array.sh gpt claude gemini 

scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/frankenstein_gen-gpt_evo-claude_exe-gemini_14166629 \
       <local-path>/Downloads/Game-Theory-with-LLMs/simulations/


# HETEROGENEOUS
sbatch slurm/run_baseline_array.sh claude baseline random
sbatch slurm/run_baseline_array.sh claude baseline replacement
sbatch slurm/run_baseline_array.sh claude baseline claude/gpt/gemini

tail -f /home/<user>/slurm_job/logs/<logid>_*_baseline_array.log

# COPY: mode: random, replacement, claude
scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulations/array_14109276 \
       <local-path>/Downloads/Game-Theory-with-LLMs/simulations/heterogeneous_random/






# all model donations over generations
python3 figures/figs_overlay.py

# with frankenstein
python figures/figs_overlay.py \
    --base_dir simulations/neutral_baseline --max_runs 0 \
    --frankenstein_dirs simulations/frankenstein_gen-gemini_evo-gpt_exe-claude_14078715 \
    simulations/frankenstein_gen-claude_evo-gemini_exe-gpt_<otherjobid>

# per model agent vs. pair donation plot
# run for gen 1 and gen 10
python3 figures/plot_agent_donations.py \
  simulations/new_baseline \
  --model claude \
  --generation 10 \
  --output_dir figures/reactivity_plots/gen10 \
  --cost_cap 10

# strategy analysis
python figures/strategy_analysis.py simulations/new_baseline \
    --output_dir figures/reactivity_plots/gen1

# all models aggregated code strategy plot
python figures/plot_coded_strategies.py \
  --base_dir figures/reactivity_plots/gen10 \
  --generation 10 \
  --output_dir figures/coded_strats

# strategy inheritance stats
python figures/analyze_strategy_inheritance.py --model gpt

python -c "import json; print(len(json.load(open('<local-path>/Downloads/Game-Theory-with-LLMs/figures/reactivity_plots/claude-3-5-sonnet_coopGain_2.0punLoss_2.0_simple_gen10_agents12_top/gen1_underspecified_20260409_183517.json'))))"

# strategy inheritance comparison with rank 1
python figures/comparing_inherited_strategy.py \
        --generation 2 \
        --model gpt \
        [--sim_dir simulations/new_baseline] \
        [--analysis_dir figures/reactivity_plots/gen1] \
        [--output_dir figures/strategy_inherit_comparison]

# strategy group analysis
python figures/plot_strategy_groups.py \
        [--dir_0417 simulations/0417] \
        [--dir_0424 simulations/0424] \
        [--output_dir figures/strategy_groups]

# copy results
scp -r <user>@<cluster-host>:/home/<user>/Game-Theory-with-LLMs/simulation_results/ \
    <local-path>/Downloads/Game-Theory-with-LLMs/simulations/new_baseline

# clean up folder
rm -f /home/<user>/Game-Theory-with-LLMs/simulation_results/*.{json,txt,log}


