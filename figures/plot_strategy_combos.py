import os
import sys
import json
import datetime
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
import collections
import matplotlib
import numpy as np


def get_model_family(model_str):
    model_str = model_str.lower()
    if 'claude' in model_str:
        return 'claude'
    elif 'gpt' in model_str:
        return 'gpt'
    elif 'gemini' in model_str:
        return 'gemini'
    else:
        # Safer fallback - return the model string itself if we can't parse it
        return model_str

def compare_strategies_across_prompts(json_files, initial_endowment=10, output_dir_root='strategy_figures'):
    # Creates strategy-centric plots showing how each strategy performs across different prompts
    # Load all dataframes and extract model names
    dfs = []
    model_names = []
    model_families = []
    for json_file in json_files:
        df = pd.read_json(json_file)
        # Try to extract model name from file name
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        parts = base_name.split('_')
        
        # Handle different filename formats
        if len(parts) >= 3 and parts[0] == 'prompt' and parts[1] == 'strategy':
            # Format: prompt_strategy_gpt-4o_20241201_143022.json
            model_name = parts[2]
        elif len(parts) >= 4 and parts[0] == 'single' and parts[1] == 'donor':
            # Format: single_donor_responses_claude-3-5-sonnet_20250724_101625.json
            model_name = parts[3]
        else:
            # Fallback: try to find any part that looks like a model name
            model_name = "unknown_model"
            for part in parts:
                if any(keyword in part.lower() for keyword in ['gpt-4o', 'claude-3-5-sonnet', 'gemini-1.5-flash']):
                    model_name = part
                    break
        
        model_family = get_model_family(model_name)
        df['model_str'] = model_name
        df['model_family'] = model_family
        dfs.append(df)
        model_names.append(model_name)
        model_families.append(model_family)

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Get unique strategies and prompts
    strategies = combined_df['strategy'].unique()
    n_strategies = len(strategies)
    
    # Create prompt labels for legend
    prompt_labels = [
        "Original",
        "Shortened", 
        "Strategic",
        "Social Cooperation",
        "Competitive"
    ]

    # Assign a unique color to each prompt
    colors = matplotlib.colormaps['tab10']
    prompt_to_color = {i: colors(i) for i in range(len(prompt_labels))}

    # Output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir_root, f"strategy_grouped_{model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    for i, strategy in enumerate(strategies, 1):
        plt.figure(figsize=(12, 8))
        wrapped_strategy = textwrap.fill(strategy, width=80)
        plt.figtext(0.5, 0.95, f"Strategy {i}:\n{wrapped_strategy}",
                    ha='center', va='top', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.6))
        
        # Plot each prompt for this strategy
        for prompt_idx in range(len(prompt_labels)):
            # Get the prompt data by matching the system prompt content
            # We'll use a more reliable method to identify prompts
            if prompt_idx == 0:  # Original
                prompt_data = combined_df[
                    (combined_df['strategy'] == strategy) & 
                    (combined_df['system_prompt'].str.contains('game-playing agent', case=False, na=False))
                ]
            elif prompt_idx == 1:  # Shortened
                prompt_data = combined_df[
                    (combined_df['strategy'] == strategy) & 
                    (combined_df['system_prompt'].str.contains('donor game paired', case=False, na=False))
                ]
            elif prompt_idx == 2:  # Strategic
                prompt_data = combined_df[
                    (combined_df['strategy'] == strategy) & 
                    (combined_df['system_prompt'].str.contains('multi-agent economics experiment', case=False, na=False))
                ]
            elif prompt_idx == 3:  # Social Cooperation
                prompt_data = combined_df[
                    (combined_df['strategy'] == strategy) & 
                    (combined_df['system_prompt'].str.contains('social cooperation', case=False, na=False))
                ]
            elif prompt_idx == 4:  # Competitive
                prompt_data = combined_df[
                    (combined_df['strategy'] == strategy) & 
                    (combined_df['system_prompt'].str.contains('competitive resource game', case=False, na=False))
                ]
            
            if not prompt_data.empty:
                # Sort by recipient percentage
                prompt_data = prompt_data.sort_values('recipient_percentage')
                
                # Add small jitter to make overlapping points visible
                #jitter_x = np.random.normal(0, 0.5, len(prompt_data))
                #jitter_y = np.random.normal(0, 0.5, len(prompt_data))
                
                # Plot all data points with alpha transparency and jitter
                plt.scatter(
                    #prompt_data['recipient_percentage'] + jitter_x,
                    #prompt_data['donor_donation_pct'] + jitter_y,
                    prompt_data['recipient_percentage'],
                    prompt_data['donor_donation_pct'],
                    alpha=0.6,
                    s=30,
                    color=prompt_to_color[prompt_idx],
                    label=prompt_labels[prompt_idx]
                )
        plt.xlabel('Recipient Previous Donation (%)', fontsize=12)
        plt.ylabel('Donor Donation (%)', fontsize=12)
        plt.title(f'{model_name} Strategy {i}: Performance Across Different Prompts', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.legend(title='Prompt Style', fontsize='medium')
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"strategy_{i:02d}_prompt_comparison.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nCreated {n_strategies} strategy-centric plots in {output_dir}/")

def plot_strategy_averages_with_error(json_files, initial_endowment=10, output_dir_root='strategy_figures'):
    """
    For each strategy, shows the mean donor response for each prompt style,
    with error shading (standard deviation).
    """
    dfs = []
    model_names = []
    model_families = []
    for json_file in json_files:
        df = pd.read_json(json_file)
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        parts = base_name.split('_')
        
        # Handle different filename formats
        if len(parts) >= 3 and parts[0] == 'prompt' and parts[1] == 'strategy':
            # Format: prompt_strategy_gpt-4o_20241201_143022.json
            model_name = parts[2]
        elif len(parts) >= 4 and parts[0] == 'single' and parts[1] == 'donor':
            # Format: single_donor_responses_claude-3-5-sonnet_20250724_101625.json
            model_name = parts[3]
        else:
            # Fallback: try to find any part that looks like a model name
            model_name = "unknown_model"
            for part in parts:
                if any(keyword in part.lower() for keyword in ['gpt-4o', 'claude-3-5-sonnet', 'gemini-1.5-flash']):
                    model_name = part
                    break
        
        model_family = get_model_family(model_name)
        df['model_str'] = model_name
        df['model_family'] = model_family
        dfs.append(df)
        model_names.append(model_name)
        model_families.append(model_family)

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    strategies = combined_df['strategy'].unique()
    n_strategies = len(strategies)
    
    prompt_labels = [
        "Original",
        "Shortened", 
        "Strategic",
        "Social Cooperation",
        "Competitive"
    ]

    colors = matplotlib.colormaps['tab10']
    prompt_to_color = {i: colors(i) for i in range(len(prompt_labels))}

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir_root, f"strategy_avg_{model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    for i, strategy in enumerate(strategies, 1):
        plt.figure(figsize=(12, 8))
        wrapped_strategy = textwrap.fill(strategy, width=80)
        plt.figtext(0.5, 0.95, f"Strategy {i}:\n{wrapped_strategy}",
                    ha='center', va='top', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.6))
        
        for prompt_idx in range(len(prompt_labels)):
            # Get the prompt data by matching the system prompt content
            # We'll use a more reliable method to identify prompts
            if prompt_idx == 0:  # Original
                prompt_data = combined_df[
                    (combined_df['strategy'] == strategy) & 
                    (combined_df['system_prompt'].str.contains('game-playing agent', case=False, na=False))
                ]
            elif prompt_idx == 1:  # Shortened
                prompt_data = combined_df[
                    (combined_df['strategy'] == strategy) & 
                    (combined_df['system_prompt'].str.contains('donor game paired', case=False, na=False))
                ]
            elif prompt_idx == 2:  # Strategic
                prompt_data = combined_df[
                    (combined_df['strategy'] == strategy) & 
                    (combined_df['system_prompt'].str.contains('multi-agent economics experiment', case=False, na=False))
                ]
            elif prompt_idx == 3:  # Social Cooperation
                prompt_data = combined_df[
                    (combined_df['strategy'] == strategy) & 
                    (combined_df['system_prompt'].str.contains('social cooperation', case=False, na=False))
                ]
            elif prompt_idx == 4:  # Competitive
                prompt_data = combined_df[
                    (combined_df['strategy'] == strategy) & 
                    (combined_df['system_prompt'].str.contains('competitive resource game', case=False, na=False))
                ]
            
            if not prompt_data.empty:
                # Group by recipient_percentage, aggregate mean and std
                grouped = prompt_data.groupby('recipient_percentage')['donor_donation_pct'].agg(['mean', 'std']).reset_index()
                
                plt.plot(
                    grouped['recipient_percentage'],
                    grouped['mean'],
                    marker='o',
                    linewidth=2,
                    markersize=6,
                    color=prompt_to_color[prompt_idx],
                    label=prompt_labels[prompt_idx]
                )
                plt.fill_between(
                    grouped['recipient_percentage'],
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    color=prompt_to_color[prompt_idx],
                    alpha=0.2
                )
        plt.xlabel('Recipient Previous Donation (%)', fontsize=12)
        plt.ylabel('Average Donor Donation (%)', fontsize=12)
        plt.title(f'{model_name} Strategy {i}: Average Performance Across Prompts (Mean ± Std)', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.legend(title='Prompt Style', fontsize='medium')
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"strategy_{i:02d}_prompt_avg.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nCreated {n_strategies} strategy average plots in {output_dir}/")

def plot_single_prompt_strategies(json_files, initial_endowment=10, output_dir_root='strategy_figures'):
    """
    Creates strategy-centric plots for a single prompt type (Original prompt only).
    Simplified version for when you're only testing one prompt framing.
    """
    # Load all dataframes and extract model names
    dfs = []
    model_names = []
    model_families = []
    for json_file in json_files:
        df = pd.read_json(json_file)
        # Try to extract model name from file name
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        parts = base_name.split('_')
        
        # Handle different filename formats
        if len(parts) >= 3 and parts[0] == 'prompt' and parts[1] == 'strategy':
            # Format: prompt_strategy_gpt-4o_20241201_143022.json
            model_name = parts[2]
        elif len(parts) >= 4 and parts[0] == 'single' and parts[1] == 'donor':
            # Format: single_donor_responses_claude-3-5-sonnet_20250724_101625.json
            model_name = parts[3]
        else:
            # Fallback: try to find any part that looks like a model name
            model_name = "unknown_model"
            for part in parts:
                if any(keyword in part.lower() for keyword in ['gpt-4o', 'claude-3-5-sonnet', 'gemini-1.5-flash']):
                    model_name = part
                    break
        
        model_family = get_model_family(model_name)
        df['model_str'] = model_name
        df['model_family'] = model_family
        dfs.append(df)
        model_names.append(model_name)
        model_families.append(model_family)

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Get unique strategies
    strategies = combined_df['strategy'].unique()
    n_strategies = len(strategies)
    
    # Output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir_root, f"strategies_{model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    for i, strategy in enumerate(strategies, 1):
        plt.figure(figsize=(12, 8))
        wrapped_strategy = textwrap.fill(strategy, width=80)
        plt.figtext(0.5, 0.95, f"Strategy {i}:\n{wrapped_strategy}",
                    ha='center', va='top', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.6))
        
        # Simple plotting for single prompt type (Original)
        prompt_data = combined_df[
            (combined_df['strategy'] == strategy) & 
            (combined_df['system_prompt'].str.contains('game-playing agent', case=False, na=False))
        ]
        
        if not prompt_data.empty:
            # Sort by recipient percentage
            prompt_data = prompt_data.sort_values('recipient_percentage')
            
            # Plot all data points
            plt.scatter(
                prompt_data['recipient_percentage'],
                prompt_data['donor_donation_pct'],
                alpha=0.6,
                s=30,
                color='blue',
                label='Original Prompt'
            )
        
        plt.xlabel('Recipient Previous Donation (%)', fontsize=12)
        plt.ylabel('Donor Donation (%)', fontsize=12)
        plt.title(f'{model_name} Strategy {i}: Performance (Original Prompt)', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.legend(fontsize='medium')
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"strategy_{i:02d}_single_prompt.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nCreated {n_strategies} single-prompt strategy plots in {output_dir}/")

def plot_single_prompt_averages(json_files, initial_endowment=10, output_dir_root='strategy_figures'):
    """
    Creates averaged plots with error shading for a single prompt type.
    Simplified version for when you're only testing one prompt framing.
    """
    # Load all dataframes and extract model names
    dfs = []
    model_names = []
    model_families = []
    for json_file in json_files:
        df = pd.read_json(json_file)
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        parts = base_name.split('_')
        
        # Handle different filename formats
        if len(parts) >= 3 and parts[0] == 'prompt' and parts[1] == 'strategy':
            model_name = parts[2]
        elif len(parts) >= 4 and parts[0] == 'single' and parts[1] == 'donor':
            model_name = parts[3]
        else:
            model_name = "unknown_model"
            for part in parts:
                if any(keyword in part.lower() for keyword in ['gpt-4o', 'claude-3-5-sonnet', 'gemini-1.5-flash']):
                    model_name = part
                    break
        
        model_family = get_model_family(model_name)
        df['model_str'] = model_name
        df['model_family'] = model_family
        dfs.append(df)
        model_names.append(model_name)
        model_families.append(model_family)

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    strategies = combined_df['strategy'].unique()
    n_strategies = len(strategies)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir_root, f"single_prompt_avg_{model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    for i, strategy in enumerate(strategies, 1):
        plt.figure(figsize=(12, 8))
        wrapped_strategy = textwrap.fill(strategy, width=80)
        plt.figtext(0.5, 0.95, f"Strategy {i}:\n{wrapped_strategy}",
                    ha='center', va='top', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.6))
        
        # Simple plotting for single prompt type (Original)
        prompt_data = combined_df[
            (combined_df['strategy'] == strategy) & 
            (combined_df['system_prompt'].str.contains('game-playing agent', case=False, na=False))
        ]
        
        if not prompt_data.empty:
            # Group by recipient_percentage, aggregate mean and std
            grouped = prompt_data.groupby('recipient_percentage')['donor_donation_pct'].agg(['mean', 'std']).reset_index()
            
            plt.plot(
                grouped['recipient_percentage'],
                grouped['mean'],
                marker='o',
                linewidth=2,
                markersize=6,
                color='blue',
                label='Original Prompt (Mean ± Std)'
            )
            plt.fill_between(
                grouped['recipient_percentage'],
                grouped['mean'] - grouped['std'],
                grouped['mean'] + grouped['std'],
                color='blue',
                alpha=0.2
            )
        
        plt.xlabel('Recipient Previous Donation (%)', fontsize=12)
        plt.ylabel('Average Donor Donation (%)', fontsize=12)
        plt.title(f'{model_name} Strategy {i}: Average Performance (Original Prompt)', fontsize=14, pad=20)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.legend(fontsize='medium')
        plt.tight_layout()
        fig_path = os.path.join(output_dir, f"strategy_{i:02d}_single_prompt_avg.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nCreated {n_strategies} single-prompt average plots in {output_dir}/")

if __name__ == "__main__":
    base_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'simulation_results')
    json_files = [
       'prompt_strategy_claude-3-5-sonnet_20250815_163556.json',
       'prompt_strategy_claude-3-5-sonnet_20250816_001608.json'
   ]
    files = [os.path.join(base_folder, filename) for filename in json_files]
    
    # Create both types of plots (multi-prompt)
    #compare_strategies_across_prompts(files, initial_endowment=10)
    #plot_strategy_averages_with_error(files, initial_endowment=10)
    
    # Create single-prompt plots (simplified)
    plot_single_prompt_strategies(files, initial_endowment=10)
    plot_single_prompt_averages(files, initial_endowment=10) 