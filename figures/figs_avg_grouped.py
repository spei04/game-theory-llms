import json
import re
import matplotlib.pyplot as plt
import numpy as np
import os


def extract_data_from_og_code_format(data):
    """
    Helper function to extract generation data from og_code format
    and convert it to the format expected by figs.py
    """
    agents_data = data.get('agents_data', [])
    print(f"Processing {len(agents_data)} data points...")
    
    # Group data by generation
    generations = {}
    for agent_round in agents_data:
        gen = agent_round.get('current_generation', 1)
        if gen not in generations:
            generations[gen] = []
        generations[gen].append(agent_round)
    
    print(f"Found generations: {sorted(generations.keys())}")
    
    # Calculate averages for each generation
    generation_summaries = []
    for gen_num in sorted(generations.keys()):
        gen_data = generations[gen_num]
        
        # Calculate average resources for this generation
        # Get the final resources for each agent in this generation
        agent_final_resources = {}
        for round_data in gen_data:
            agent_name = round_data['agent_name']
            resources = round_data['resources']
            # Keep the latest resource value for each agent in this generation
            agent_final_resources[agent_name] = resources
        
        if agent_final_resources:
            avg_resources = sum(agent_final_resources.values()) / len(agent_final_resources)
        else:
            avg_resources = 0
        
        # Calculate average donation percentage for this generation
        # Get all donation events for this generation
        donations = []
        for round_data in gen_data:
            if round_data.get('is_donor', False) and round_data.get('donated', 0) > 0:
                # Calculate donation percentage (donated / (donated + remaining resources))
                donated = round_data['donated']
                remaining = round_data['resources']
                if donated + remaining > 0:
                    donation_pct = (donated / (donated + remaining)) * 100
                    donations.append(donation_pct)
        
        if donations:
            avg_donations = sum(donations) / len(donations)
        else:
            avg_donations = 0
        
        # Create summary string in the format figs.py expects
        summary = f"Average resources for Generation {gen_num}: {avg_resources:.1f} units\nAverage donation percentage for Generation {gen_num}: {avg_donations:.1f}%"
        generation_summaries.append(summary)
    
    return generation_summaries

filenames = [
            "simulations/gpt-4o-gen10-rounds12-agents12/Final_Accum_gpt-4o_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20260228_001723.json",
            "simulations/gpt-4o-gen10-rounds12-agents12/Final_Accum_gpt-4o_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20260228_011041.json",
            "simulations/gpt-4o-gen10-rounds12-agents12/Final_Accum_gpt-4o_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20260228_020622.json",
            "simulations/gpt-4o-gen10-rounds12-agents12/Final_Accum_gpt-4o_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20260228_030150.json",
            "simulations/gpt-4o-gen10-rounds12-agents12/Final_Accum_gpt-4o_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20260228_040018.json",
            "simulations/gpt-4o-gen10-rounds12-agents12/Final_Accum_gpt-4o_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20260228_130007.json",
            "simulations/gpt-4o-gen10-rounds12-agents12/Final_Accum_gpt-4o_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20260228_140134.json",
            "simulations/gpt-4o-gen10-rounds12-agents12/Final_Accum_gpt-4o_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20260228_150142.json",
            "simulations/gpt-4o-gen10-rounds12-agents12/Final_Accum_gpt-4o_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20260228_160059.json",
            "simulations/gpt-4o-gen10-rounds12-agents12/Final_Accum_gpt-4o_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20260228_165512.json",
            "simulations/gpt-4o-gen10-rounds12-agents12/Final_Accum_gpt-4o_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20260228_175052.json",
            "simulations/gpt-4o-gen10-rounds12-agents12/Final_Accum_gpt-4o_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20260228_184724.json",
            "simulations/gpt-4o-gen10-rounds12-agents12/Final_Accum_gpt-4o_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20260228_195332.json",
    ]

# Check if files exist
existing_files = [f for f in filenames if os.path.exists(f)]
if not existing_files:
    print("No files found. Please check your file paths.")
    exit(1)

print(f"Processing {len(existing_files)} files:")
for f in existing_files:
    print(f"  - {f}")

all_runs_resources = []
all_runs_donations = []

for results_file in existing_files:
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Check if data has generation_summaries, if not, extract from og_code format
    summaries = data.get('generation_summaries', [])
    if not summaries:
        print(f"Converting og_code format data for {results_file}")
        summaries = extract_data_from_og_code_format(data)
    
    run_resources = []
    run_donations = []
    for summary in summaries:
        # Extract average resources
        resources_match = re.search(r'Average resources for Generation (\d+): ([\d.]+) units', summary)
        if resources_match:
            avg_resources = float(resources_match.group(2))
            run_resources.append(avg_resources)
        
        # Extract average donation percentage
        donations_match = re.search(r'Average donation percentage for Generation (\d+): ([\d.]+)%', summary)
        if donations_match:
            avg_donations = float(donations_match.group(2))
            run_donations.append(avg_donations)
    
    if run_resources:
        all_runs_resources.append(run_resources)
    if run_donations:
        all_runs_donations.append(run_donations)

if not all_runs_resources and not all_runs_donations:
    print("No data found in any files.")
    exit(1)

# Pad runs to the same length (in case some runs have fewer generations)
max_gens_resources = max(len(run) for run in all_runs_resources) if all_runs_resources else 0
max_gens_donations = max(len(run) for run in all_runs_donations) if all_runs_donations else 0
max_gens = max(max_gens_resources, max_gens_donations)

for run in all_runs_resources:
    while len(run) < max_gens:
        run.append(np.nan)

for run in all_runs_donations:
    while len(run) < max_gens:
        run.append(np.nan)

# Convert to numpy arrays
all_runs_resources = np.array(all_runs_resources) if all_runs_resources else np.array([])
all_runs_donations = np.array(all_runs_donations) if all_runs_donations else np.array([])

# Calculate means and standard errors
if len(all_runs_resources) > 0:
    mean_resources = np.nanmean(all_runs_resources, axis=0)
    stderr_resources = np.nanstd(all_runs_resources, axis=0) / np.sqrt(np.sum(~np.isnan(all_runs_resources), axis=0))
else:
    mean_resources = np.array([])
    stderr_resources = np.array([])

if len(all_runs_donations) > 0:
    mean_donations = np.nanmean(all_runs_donations, axis=0)
    stderr_donations = np.nanstd(all_runs_donations, axis=0) / np.sqrt(np.sum(~np.isnan(all_runs_donations), axis=0))
else:
    mean_donations = np.array([])
    stderr_donations = np.array([])

generations = np.arange(1, max_gens + 1)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot each individual run (resources)
for run in all_runs_resources:
    ax1.plot(generations[:len(run)], run, color='gray', alpha=0.3, linewidth=1)

# Plot each individual run (donations)
for run in all_runs_donations:
    ax2.plot(generations[:len(run)], run, color='gray', alpha=0.3, linewidth=1)

# Plot 1: Average Final Resources
if len(mean_resources) > 0:
    ax1.plot(generations[:len(mean_resources)], mean_resources, marker='o', label='GPT-4o (mean of runs)')
    ax1.fill_between(generations[:len(mean_resources)], 
                     mean_resources - stderr_resources, 
                     mean_resources + stderr_resources, alpha=0.2)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Average Final Resources')
    ax1.set_title('Cultural Evolution of Cooperation - Resources')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
else:
    ax1.text(0.5, 0.5, 'No resource data available', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title('Cultural Evolution of Cooperation - Resources')

# Plot 2: Average Donation Percentage
if len(mean_donations) > 0:
    ax2.plot(generations[:len(mean_donations)], mean_donations, marker='s', color='orange', label='GPT-4o (mean of runs)')
    ax2.fill_between(generations[:len(mean_donations)], 
                     mean_donations - stderr_donations, 
                     mean_donations + stderr_donations, alpha=0.2, color='orange')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Average Donation Percentage (%)')
    ax2.set_title('Cultural Evolution of Cooperation - Donations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
else:
    ax2.text(0.5, 0.5, 'No donation data available', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Cultural Evolution of Cooperation - Donations')

plt.tight_layout()
plt.show()