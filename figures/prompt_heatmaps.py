import os
import sys
import json
import datetime
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib


def get_model_family(model_str):
    model_str = model_str.lower()
    if 'claude' in model_str:
        return 'claude'
    elif 'gpt' in model_str:
        return 'gpt'
    elif 'gemini' in model_str:
        return 'gemini'
    else:
        return model_str

def create_prompt_heatmap(json_files, initial_endowment=10, output_dir_root='strategy_figures'):
    """
    Creates a heatmap matrix showing:
    - Rows: Different prompts
    - Columns: Different recipient donation levels
    - Values: Average donor donation percentage
    - Color intensity: Higher donations = darker
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
    
    # Create prompt labels
    prompt_labels = [
        "Original",
        "Shortened", 
        "Strategic",
        "Social Cooperation",
        "Competitive"
    ]
    
    # Get unique recipient donation levels
    recipient_levels = sorted(combined_df['recipient_percentage'].unique())
    
    # Create the heatmap data
    heatmap_data = []
    
    # Define reliable prompt identifiers 
    prompt_identifiers = [
        'game-playing agent',      # Original
        'donor game paired',       # Shortened
        'multi-agent economics experiment',  # Strategic
        'social cooperation',      # Social Cooperation
        'competitive resource game'  # Competitive
    ]
    
    for prompt_idx in range(len(prompt_labels)):
        row_data = []
        for recipient_level in recipient_levels:
            # Get data for this prompt and recipient level (average across all strategies)
            prompt_data = combined_df[
                (combined_df['system_prompt'].str.contains(prompt_identifiers[prompt_idx], case=False, na=False)) &
                (combined_df['recipient_percentage'] == recipient_level)
            ]
            
            if not prompt_data.empty:
                avg_donation = prompt_data['donor_donation_pct'].mean()
                row_data.append(avg_donation)
            else:
                row_data.append(np.nan)
        
        heatmap_data.append(row_data)
    
    # Convert to numpy array
    heatmap_array = np.array(heatmap_data)
    
    # Create the heatmap
    plt.figure(figsize=(14, 8))
    
    # Create custom colormap (green for low donations, red for high donations)
    colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
    n_bins = 100
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_diverging', colors, N=n_bins)
    
    # Create the heatmap
    sns.heatmap(heatmap_array, 
                annot=True, 
                fmt='.1f',
                cmap=cmap,
                cbar_kws={'label': 'Average Donor Donation (%)'},
                xticklabels=[f'{level}%' for level in recipient_levels],
                yticklabels=prompt_labels,
                linewidths=0.5,
                linecolor='white')
    
    plt.xlabel('Recipient Previous Donation (%)', fontsize=12)
    plt.ylabel('Prompt Style', fontsize=12)
    plt.title(f'Prompt Performance Heatmap: Average Donor Donation by Prompt and Recipient Level', fontsize=14, pad=20)
    plt.tight_layout()
    
    # Save the plot
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir_root, f"heatmap_{model_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    fig_path = os.path.join(output_dir, f"prompt_heatmap.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nCreated prompt heatmap in {output_dir}/")
    
    return heatmap_array, output_dir

def create_strategy_heatmap(json_files, initial_endowment=10, output_dir_root='strategy_figures', output_dir=None):
    """
    Creates a heatmap matrix showing:
    - Rows: Different strategies
    - Columns: Different recipient donation levels
    - Values: Average donor donation percentage
    - Color intensity: Higher donations = darker
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
    
    # Get unique strategies and recipient donation levels
    strategies = combined_df['strategy'].unique()
    recipient_levels = sorted(combined_df['recipient_percentage'].unique())
    
    # Create the heatmap data
    heatmap_data = []
    strategy_labels = []
    
    for strategy in strategies:
        # Use full strategy text with text wrapping for better readability
        wrapped_strategy = textwrap.fill(strategy, width=60, break_long_words=False, break_on_hyphens=False)
        strategy_labels.append(wrapped_strategy)
        row_data = []
        for recipient_level in recipient_levels:
            # Get data for this strategy and recipient level (average across all prompts)
            strategy_data = combined_df[
                (combined_df['strategy'] == strategy) &
                (combined_df['recipient_percentage'] == recipient_level)
            ]
            
            if not strategy_data.empty:
                avg_donation = strategy_data['donor_donation_pct'].mean()
                row_data.append(avg_donation)
            else:
                row_data.append(np.nan)
        
        heatmap_data.append(row_data)
    
    # Convert to numpy array
    heatmap_array = np.array(heatmap_data)
    
    # Create the heatmap
    plt.figure(figsize=(16, 10))
    
    # Create custom colormap (green for low donations, red for high donations)
    colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
    n_bins = 100
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_diverging', colors, N=n_bins)
    
    # Create the heatmap
    sns.heatmap(heatmap_array, 
                annot=True, 
                fmt='.1f',
                cmap=cmap,
                cbar_kws={'label': 'Average Donor Donation (%)'},
                xticklabels=[f'{level}%' for level in recipient_levels],
                yticklabels=strategy_labels,
                linewidths=0.5,
                linecolor='white')
    
    plt.xlabel('Recipient Previous Donation (%)', fontsize=12)
    plt.ylabel('Strategy', fontsize=12)
    plt.title(f'Strategy Performance Heatmap: Average Donor Donation by Strategy and Recipient Level', fontsize=14, pad=20)
    plt.tight_layout()
    
    # Use provided output directory or create new one
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir_root, f"heatmap_{model_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
    
    fig_path = os.path.join(output_dir, f"strategy_heatmap.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nCreated strategy heatmap in {output_dir}/")
    
    return heatmap_array

def create_strategy_adherence_heatmap(json_files, initial_endowment=10, output_dir_root='strategy_figures', output_dir=None):
    """
    Creates a heatmap matrix showing:
    - Rows: Different strategies
    - Columns: Different prompt styles
    - Values: Mean squared error between actual donations and strategy-specified amounts
    - Color intensity: Lower MSE = better adherence (darker green)
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
    
    # Get unique strategies and create prompt labels
    strategies = combined_df['strategy'].unique()
    prompt_labels = [
        "Original",
        "Shortened", 
        "Strategic",
        "Social Cooperation",
        "Competitive"
    ]
    
    # Define reliable prompt identifiers
    prompt_identifiers = [
        'game-playing agent',      # Original
        'donor game paired',       # Shortened
        'multi-agent economics experiment',  # Strategic
        'social cooperation',      # Social Cooperation
        'competitive resource game'  # Competitive
    ]
    
    def calculate_strategy_target(strategy_text, recipient_percentage, initial_endowment):
        """Calculate the target donation amount based on the strategy"""
        strategy_lower = strategy_text.lower()
        
        # Strategy 1: "donate the same amount as the recipient's previous donation"
        if "same amount as" in strategy_lower or "same as" in strategy_lower:
            return (recipient_percentage / 100) * initial_endowment
        
        # Strategy 2: "donate 50% of my units every time"
        if "50%" in strategy_lower and "every time" in strategy_lower:
            return 0.5 * initial_endowment
        
        # Strategy 4: "always donate 10% more than the recipient previously donated, capped at 100%"
        if "10% more" in strategy_lower and "capped" in strategy_lower:
            target_percentage = min(recipient_percentage + 10, 100)
            return (target_percentage / 100) * initial_endowment
        
        # Strategy 5: Empty strategy
        if not strategy_text.strip():
            return None  # No clear target
        
        # Strategy 3: Complex conditional strategy - skip for now
        if "more than 50%" in strategy_lower and "less than 50%" in strategy_lower:
            return None  # Too complex to parse
        
        return None  # Unknown strategy
    
    # Create the heatmap data
    heatmap_data = []
    strategy_labels = []
    
    for strategy in strategies:
        wrapped_strategy = textwrap.fill(strategy, width=60, break_long_words=False, break_on_hyphens=False)
        strategy_labels.append(wrapped_strategy)
        row_data = []
        
        for prompt_idx in range(len(prompt_labels)):
            # Get data for this strategy and prompt
            strategy_prompt_data = combined_df[
                (combined_df['strategy'] == strategy) & 
                (combined_df['system_prompt'].str.contains(prompt_identifiers[prompt_idx], case=False, na=False))
            ]
            
            if not strategy_prompt_data.empty:
                # Calculate MSE for each data point
                mse_values = []
                for _, row in strategy_prompt_data.iterrows():
                    target_amount = calculate_strategy_target(
                        row['strategy'], 
                        row['recipient_percentage'], 
                        initial_endowment
                    )
                    
                    if target_amount is not None:
                        actual_amount = row['donor_donation']
                        mse = (actual_amount - target_amount) ** 2
                        mse_values.append(mse)
                
                # Calculate mean MSE
                if mse_values:
                    avg_mse = np.mean(mse_values)
                    row_data.append(avg_mse)
                else:
                    row_data.append("None")  # Strategy too complex to parse
            else:
                row_data.append(np.nan)  # No data available
        
        heatmap_data.append(row_data)
    
    # Convert to numpy array and handle "None" strings
    heatmap_array = []
    for row in heatmap_data:
        converted_row = []
        for val in row:
            if val == "None":
                converted_row.append(np.nan)
            else:
                converted_row.append(val)
        heatmap_array.append(converted_row)
    
    heatmap_array = np.array(heatmap_array)
    
    # Create custom annotations for the heatmap
    annot_array = []
    for row in heatmap_data:
        annot_row = []
        for val in row:
            if val == "None":
                annot_row.append("None")
            elif np.isnan(val):
                annot_row.append("")
            else:
                annot_row.append(f"{val:.2f}")
        annot_array.append(annot_row)
    
    annot_array = np.array(annot_array)
    
    # Create the heatmap with larger height to accommodate full strategy text
    plt.figure(figsize=(16, 12))
    
    # Create custom colormap (green for low MSE = good adherence, red for high MSE = poor adherence)
    colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
    n_bins = 100
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_diverging', colors, N=n_bins)
    
    # Create the heatmap
    sns.heatmap(heatmap_array, 
                annot=annot_array,  # Use custom annotations
                fmt='',  # Don't format, we handle it ourselves
                cmap=cmap,
                cbar_kws={'label': 'Mean Squared Error (Lower = Better Adherence)'},
                xticklabels=prompt_labels,
                yticklabels=strategy_labels,
                linewidths=0.5,
                linecolor='white')
    
    plt.xlabel('Prompt Style', fontsize=12)
    plt.ylabel('Strategy', fontsize=12)
    plt.title(f'Strategy Adherence Heatmap: MSE Between Actual and Strategy-Intended Donations', fontsize=14, pad=20)
    plt.tight_layout()
    
    # Use provided output directory or create new one
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir_root, f"heatmap_{model_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
    
    fig_path = os.path.join(output_dir, f"strategy_adherence_heatmap.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nCreated strategy adherence heatmap in {output_dir}/")
    
    return heatmap_array

if __name__ == "__main__":
    base_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'simulation_results')
    json_files = [
        'prompt_strategy_claude-3-5-sonnet_20250728_152636.json',
        'prompt_strategy_claude-3-5-sonnet_20250728_155612.json',
        'prompt_strategy_claude-3-5-sonnet_20250728_161350.json',
        'prompt_strategy_claude-3-5-sonnet_20250728_163410.json',
        'prompt_strategy_claude-3-5-sonnet_20250728_164547.json'
    ]
    files = [os.path.join(base_folder, filename) for filename in json_files]
    
    # Create both types of heatmaps
    _, output_dir = create_prompt_heatmap(files, initial_endowment=10)
    create_strategy_heatmap(files, initial_endowment=10, output_dir=output_dir) 
    create_strategy_adherence_heatmap(files, initial_endowment=10, output_dir=output_dir)
    print(f"\nAll three heatmaps saved in: {output_dir}/")