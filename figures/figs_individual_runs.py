#!/usr/bin/env python3
"""
Plot individual runs for multiple models with average lines
Shows 6 subplots: 3 models x 2 metrics (average units and donation percentage)
Each subplot shows individual runs in different colors with black dotted average line
"""

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
                donated = round_data['donated']
                remaining = round_data['resources']
                
                # Use appropriate calculation based on file type
                # For accumulated endowment: (donated / (donated + remaining)) * 100
                # For fixed endowment: (donated / 10) * 100
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

def process_model_data(file_path, model_name):
    """Process data for a single model and return resources and donations arrays"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Always extract from agents_data to get correct donation percentages
    # Don't trust generation_stats as they may have incorrect values
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
    
    return run_resources, run_donations

def process_multiple_runs(model_files, max_gens=10):
    """
    Process multiple runs for each model type using exact file names
    
    Args:
        model_files: dict with model_name -> list_of_file_paths mapping
        max_gens: maximum number of generations to process
    
    Returns:
        dict with model_name -> aggregated_data mapping
    """
    model_data = {}
    
    for model_name, file_list in model_files.items():
        print(f"\nProcessing {model_name} files...")
        print(f"Found {len(file_list)} files for {model_name}")
        
        if not file_list:
            print(f"No files provided for {model_name}")
            continue
        
        # Process each file
        all_resources = []
        all_donations = []
        
        for file_path in file_list:
            if os.path.exists(file_path):
                try:
                    resources, donations = process_model_data(file_path, f"{model_name}_run")
                    
                    # Pad to max_gens if needed
                    while len(resources) < max_gens:
                        resources.append(np.nan)
                    while len(donations) < max_gens:
                        donations.append(np.nan)
                    
                    all_resources.append(resources[:max_gens])
                    all_donations.append(donations[:max_gens])
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
            else:
                print(f"File not found: {file_path}")
        
        if all_resources and all_donations:
            # Convert to numpy arrays
            resources_array = np.array(all_resources)
            donations_array = np.array(all_donations)
            
            # Calculate statistics across runs
            mean_resources = np.nanmean(resources_array, axis=0)
            mean_donations = np.nanmean(donations_array, axis=0)
            
            model_data[model_name] = {
                'resources_array': resources_array,
                'donations_array': donations_array,
                'mean_resources': mean_resources,
                'mean_donations': mean_donations,
                'num_runs': len(all_resources)
            }
            
            print(f"Successfully processed {len(all_resources)} runs for {model_name}")
            print(f"Mean resources across runs: {np.nanmean(mean_resources):.1f}")
            print(f"Mean donations across runs: {np.nanmean(mean_donations):.1f}%")
        else:
            print(f"No valid data processed for {model_name}")
    
    return model_data

def plot_individual_runs(model_files, output_dir='individual_runs_figures', max_gens=10):
    """
    Plot individual runs for multiple models with average lines
    
    Args:
        model_files: dict with model_name -> list_of_file_paths mapping
        output_dir: directory to save plots
        max_gens: maximum number of generations to process
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all model data
    model_data = process_multiple_runs(model_files, max_gens)
    
    if not model_data:
        print("No valid data found!")
        return
    
    # Define colors for individual runs
    run_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Create subplots: 2 rows (resources, donations) x 3 columns (models)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    generations = np.arange(1, max_gens + 1)
    
    # Get model names in order
    model_names = list(model_data.keys())
    
    # Plot resources (top row)
    for i, model_name in enumerate(model_names):
        if i >= 3:  # Only plot first 3 models
            break
            
        ax = axes[0, i]
        data = model_data[model_name]
        resources_array = data['resources_array']
        mean_resources = data['mean_resources']
        num_runs = data['num_runs']
        
        # Plot individual runs
        for run_idx in range(num_runs):
            run_data = resources_array[run_idx]
            valid_mask = ~np.isnan(run_data)
            if np.any(valid_mask):
                valid_gens = generations[valid_mask]
                valid_data = run_data[valid_mask]
                color = run_colors[run_idx % len(run_colors)]
                ax.plot(valid_gens, valid_data, color=color, linewidth=1, alpha=0.7)
        
        # Plot average line
        valid_mask = ~np.isnan(mean_resources)
        if np.any(valid_mask):
            valid_gens = generations[valid_mask]
            valid_means = mean_resources[valid_mask]
            ax.plot(valid_gens, valid_means, color='black', linewidth=2, 
                   linestyle='--', label='Average', alpha=0.9)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Average Final Resources')
        ax.set_title(f'{model_name} - Resources ({num_runs} runs)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot donations (bottom row)
    for i, model_name in enumerate(model_names):
        if i >= 3:  # Only plot first 3 models
            break
            
        ax = axes[1, i]
        data = model_data[model_name]
        donations_array = data['donations_array']
        mean_donations = data['mean_donations']
        num_runs = data['num_runs']
        
        # Plot individual runs
        for run_idx in range(num_runs):
            run_data = donations_array[run_idx]
            valid_mask = ~np.isnan(run_data)
            if np.any(valid_mask):
                valid_gens = generations[valid_mask]
                valid_data = run_data[valid_mask]
                color = run_colors[run_idx % len(run_colors)]
                ax.plot(valid_gens, valid_data, color=color, linewidth=1, alpha=0.7)
        
        # Plot average line
        valid_mask = ~np.isnan(mean_donations)
        if np.any(valid_mask):
            valid_gens = generations[valid_mask]
            valid_means = mean_donations[valid_mask]
            ax.plot(valid_gens, valid_means, color='black', linewidth=2, 
                   linestyle='--', label='Average', alpha=0.9)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Average Donation Percentage (%)')
        ax.set_title(f'{model_name} - Donations ({num_runs} runs)')
        ax.set_ylim(0, 100)  # Set y-axis range from 0 to 100%
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = "individual_runs_comparison"
    filename = f"individual_runs_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {filepath}")
    
    plt.show()

# Example usage
if __name__ == "__main__":
    # Define exact file names for each model type
    model_files = {
        'Claude 3.5 Sonnet': [
                'simulations/claude-3.5-sonnet-gen10-rounds12-agents12/Final_Accum_claude-3-5-sonnet_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20260227_120326.json',
                'simulations/claude-3.5-sonnet-gen10-rounds12-agents12/Final_Accum_claude-3-5-sonnet_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20260227_123757.json',
                'simulations/claude-3.5-sonnet-gen10-rounds12-agents12/Final_Accum_claude-3-5-sonnet_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20260227_131037.json',
                'simulations/claude-3.5-sonnet-gen10-rounds12-agents12/Final_Accum_claude-3-5-sonnet_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20260227_134345.json',
                'simulations/claude-3.5-sonnet-gen10-rounds12-agents12/Final_Accum_claude-3-5-sonnet_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20260227_141453.json',
            ],
            # Add Gemini and GPT-4o runs here when available:
            # 'Gemini 2.5 Flash': [...],
            # 'GPT-4o': [...],
    }
    
    # Check which files exist
    existing_files = {}
    for model_name, file_list in model_files.items():
        existing_files[model_name] = []
        for file_path in file_list:
            if os.path.exists(file_path):
                existing_files[model_name].append(file_path)
            else:
                print(f"File not found: {file_path}")
        
        if existing_files[model_name]:
            print(f"Found {len(existing_files[model_name])} files for {model_name}")
    
    if any(existing_files.values()):
        print(f"\nProcessing {len([k for k, v in existing_files.items() if v])} model types...")
        plot_individual_runs(existing_files, max_gens=10)
    else:
        print("No model files found. Please update the model_files dictionary with your exact file paths.")
        