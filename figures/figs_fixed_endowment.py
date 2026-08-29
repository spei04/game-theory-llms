#!/usr/bin/env python3

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import re

def process_fixed_endowment_file(file_path, model_name):
    """Process a fixed_endowment2.py file by recalculating from agents_data"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Group agents_data by generation
    agents_by_generation = {}
    for agent_data in data['agents_data']:
        gen = agent_data['current_generation']
        if gen not in agents_by_generation:
            agents_by_generation[gen] = []
        agents_by_generation[gen].append(agent_data)
    
    # Calculate resources and donations for each generation
    resources_list = []
    donations_list = []
    
    for gen_num in sorted(agents_by_generation.keys()):
        agents = agents_by_generation[gen_num]
        
        # Calculate average resources (matching figs.py logic - final resources for each agent)
        agent_final_resources = {}
        for agent_data in agents:
            agent_name = agent_data['agent_name']
            resources = agent_data['resources']
            # Keep the latest resource value for each agent in this generation
            agent_final_resources[agent_name] = resources
        
        if agent_final_resources:
            avg_resources = sum(agent_final_resources.values()) / len(agent_final_resources)
        else:
            avg_resources = 0
        resources_list.append(avg_resources)
        
        # Calculate donation percentages (fixed endowment: donated / 10, capped at 10 units)
        donation_percentages = []
        for agent_data in agents:
            if agent_data['is_donor'] and agent_data['donated'] > 0:
                # Cap donations at 10 units to enforce fixed endowment rule
                capped_donation = min(agent_data['donated'], 10)
                donation_pct = (capped_donation / 10) * 100  # Fixed endowment calculation
                donation_percentages.append(donation_pct)
        
        avg_donation_pct = np.mean(donation_percentages) if donation_percentages else 0
        donations_list.append(avg_donation_pct)
    return resources_list, donations_list

def process_multiple_fixed_endowment_runs(model_files, max_gens=10):
    """Process multiple runs for each model and return aggregated data"""
    model_data = {}
    
    for model_name, file_paths in model_files.items():
        print(f"\nProcessing {model_name} with {len(file_paths)} files...")
        
        all_resources = []
        all_donations = []
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                resources, donations = process_fixed_endowment_file(file_path, model_name)
                if resources:
                    all_resources.append(resources)
                if donations:
                    all_donations.append(donations)
            else:
                print(f"Warning: File not found: {file_path}")
        
        if all_resources and all_donations:
            # Pad arrays to same length
            max_len = max(len(run) for run in all_resources)
            for run in all_resources:
                while len(run) < max_len:
                    run.append(np.nan)
            
            max_len = max(len(run) for run in all_donations)
            for run in all_donations:
                while len(run) < max_len:
                    run.append(np.nan)
            
            # Convert to numpy arrays
            resources_array = np.array(all_resources)
            donations_array = np.array(all_donations)
            
            # Calculate statistics
            mean_resources = np.nanmean(resources_array, axis=0)
            std_resources = np.nanstd(resources_array, axis=0)
            sem_resources = std_resources / np.sqrt(np.sum(~np.isnan(resources_array), axis=0))  # Standard error
            
            mean_donations = np.nanmean(donations_array, axis=0)
            std_donations = np.nanstd(donations_array, axis=0)
            sem_donations = std_donations / np.sqrt(np.sum(~np.isnan(donations_array), axis=0))  # Standard error
            
            model_data[model_name] = {
                'mean_resources': mean_resources,
                'sem_resources': sem_resources,
                'mean_donations': mean_donations,
                'sem_donations': sem_donations,
                'resources_array': resources_array,
                'donations_array': donations_array,
                'num_runs': len(all_resources)
            }
        else:
            print(f"Warning: No valid data found for {model_name}")
    
    return model_data

def plot_fixed_endowment_results(model_files, output_dir='fixed_endowment_figures', max_gens=10):
    """Create overlay plots for fixed_endowment2.py results"""
    model_data = process_multiple_fixed_endowment_runs(model_files, max_gens)
    
    if not model_data:
        print("No valid data found for any model")
        return
    
    # Define colors for each model
    colors = {
        'gemini 2.5 flash': '#4285F4',  # Google Blue
        'claude 3.5 sonnet': '#FF6B35',  # Anthropic Orange
        'gpt-4o': '#10A37F',     # OpenAI Green
    }
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Average Resources
    for model_name, data in model_data.items():
        mean_resources = data['mean_resources']
        sem_resources = data['sem_resources']
        num_runs = data['num_runs']
        
        generations = range(1, len(mean_resources) + 1)
        
        # Remove NaN values for plotting
        valid_mask = ~np.isnan(mean_resources)
        valid_gens = np.array(generations)[valid_mask]
        valid_means = mean_resources[valid_mask]
        valid_sems = sem_resources[valid_mask]
        
        if len(valid_gens) > 0:
            # Plot mean line
            ax1.plot(valid_gens, valid_means, 
                    color=colors.get(model_name.lower(), '#000000'),
                    marker='o', linewidth=2, markersize=6,
                    label=f'{model_name} ({num_runs} runs)')
            
            # Plot error band
            ax1.fill_between(valid_gens,
                           np.maximum(valid_means - valid_sems, 0),  # Clamp at 0
                           valid_means + valid_sems,
                           alpha=0.2,
                           color=colors.get(model_name.lower(), '#000000'))
    
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Average Resources')
    ax1.set_title('Average Resources per Agent')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Average Donation Percentage
    for model_name, data in model_data.items():
        mean_donations = data['mean_donations']
        sem_donations = data['sem_donations']
        num_runs = data['num_runs']
        
        generations = range(1, len(mean_donations) + 1)
        
        # Remove NaN values for plotting
        valid_mask = ~np.isnan(mean_donations)
        valid_gens = np.array(generations)[valid_mask]
        valid_means = mean_donations[valid_mask]
        valid_sems = sem_donations[valid_mask]
        
        if len(valid_gens) > 0:
            # Plot mean line
            ax2.plot(valid_gens, valid_means, 
                    color=colors.get(model_name.lower(), '#000000'),
                    marker='s', linewidth=2, markersize=6,
                    label=f'{model_name} ({num_runs} runs)')
            
            # Plot error band
            ax2.fill_between(valid_gens,
                           np.maximum(valid_means - valid_sems, 0),  # Clamp at 0
                           valid_means + valid_sems,
                           alpha=0.2,
                           color=colors.get(model_name.lower(), '#000000'))
    
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Average Donation Percentage (%)')
    ax2.set_title('Average Donation Percentage')
    ax2.set_ylim(0, 100)  # Set y-axis range from 0 to 100
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, 'fixed_endowment_overlay.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Also save to overlay_figures folder
    overlay_dir = 'overlay_figures'
    os.makedirs(overlay_dir, exist_ok=True)
    overlay_path = os.path.join(overlay_dir, 'fixed_endowment_overlay.png')
    plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
    print(f"Plot also saved to: {overlay_path}")
    
    plt.show()

if __name__ == "__main__":
    # Define your fixed_endowment2.py files here (using FIXED versions)
    model_files = {
        'Gemini 2.5 Flash': [
           'simulation_results/Final_Single_Trace_gemini-1.5-flash_coopGain_2punLoss_2_none_gen10_agents12_top_20250814_152356.json',
           'simulation_results/Final_Single_Trace_gemini-1.5-flash_coopGain_2punLoss_2_none_gen10_agents12_top_20250815_110757.json',
           'simulation_results/Final_Single_Trace_gemini-1.5-flash_coopGain_2punLoss_2_none_gen10_agents12_top_20250815_112232.json'
       ],
       'Claude 3.5 Sonnet': [
           'simulation_results/Final_Single_Trace_claude-3-5-sonnet_coopGain_2punLoss_2_none_gen10_agents12_top_20250813_134022.json',
           'simulation_results/Final_Single_Trace_claude-3-5-sonnet_coopGain_2punLoss_2_none_gen10_agents12_top_20250815_114030.json',
           'simulation_results/Final_Single_Trace_claude-3-5-sonnet_coopGain_2punLoss_2_none_gen10_agents12_top_20250815_120635.json'
       ],
       'GPT-4o': [
           'simulation_results/Final_Single_Trace_gpt-4o_coopGain_2punLoss_2_none_gen10_agents12_top_20250813_143650.json',
           'simulation_results/Final_Single_Trace_gpt-4o_coopGain_2punLoss_2_none_gen10_agents12_top_20250815_124517.json',
           'simulation_results/Final_Single_Trace_gpt-4o_coopGain_2punLoss_2_none_gen10_agents12_top_20250815_141832.json'

       ]
    }
    
    # Check if any files are specified
    total_files = sum(len(files) for files in model_files.values())
    if total_files == 0:
        print("Please add your file paths to the model_files dictionary above")
    else:
        print(f"Processing {total_files} files...")
        plot_fixed_endowment_results(model_files) 