#!/usr/bin/env python3
"""
Compare three modes: Original/Accumulated, Fixed Endowment, and Single Trace History
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

def extract_generation_10_data(file_path):
    """Extract generation 10 data from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Try to get data from generation_stats first
        generation_stats = data.get('generation_stats', [])
        if generation_stats and len(generation_stats) >= 10:
            gen_10_stats = generation_stats[9]  # 0-indexed, so gen 10 is at index 9
            avg_donation_pct = gen_10_stats.get('average_donation_percentage', 0)
            
            # If donation percentage is 0.0, recalculate from agents_data (like figs.py does)
            if avg_donation_pct == 0.0:
                print(f"Recalculating donation percentage for {file_path} (was 0.0)")
                agents_data = data.get('agents_data', [])
                if agents_data:
                    # Find generation 10 data
                    gen_10_data = [d for d in agents_data if d['current_generation'] == 10]
                    if gen_10_data:
                        # Calculate donation percentage using figs.py method
                        donations = []
                        for round_data in gen_10_data:
                            if round_data.get('is_donor', False) and round_data.get('donated', 0) > 0:
                                # Calculate donation percentage (donated / (donated + remaining resources))
                                donated = round_data['donated']
                                remaining = round_data['resources']
                                if donated + remaining > 0:
                                    donation_pct = (donated / (donated + remaining)) * 100
                                    donations.append(donation_pct)
                        
                        if donations:
                            avg_donation_pct = sum(donations) / len(donations)
                            print(f"  Recalculated: {avg_donation_pct:.1f}%")
                        else:
                            print(f"  No donation events found")
            
            return {
                'avg_donation_pct': avg_donation_pct
            }
        
        # Fallback: calculate from agents_data
        agents_data = data.get('agents_data', [])
        if not agents_data:
            return None
        
        # Find generation 10 data
        gen_10_data = [d for d in agents_data if d['current_generation'] == 10]
        if not gen_10_data:
            return None
        
        # Calculate average donation percentage (only donor rounds)
        donation_percentages = []
        for agent_data in gen_10_data:
            if agent_data.get('is_donor', False) and agent_data['donated'] > 0:
                # For fixed endowment: cap at 10 units
                donated = min(agent_data['donated'], 10)
                donation_pct = (donated / 10) * 100
                donation_percentages.append(donation_pct)
        
        avg_donation_pct = np.mean(donation_percentages) if donation_percentages else 0
        
        return {
            'avg_donation_pct': avg_donation_pct
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def collect_mode_data():
    """Collect data for all three modes using manually specified files"""
    
    # Manually specify files for each mode and model
    mode_files = {
        'Accumulated Endowment': {
            'Gemini 2.5 Flash': [
                'simulation_results/Final_Accum_gemini-1.5-flash_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20250805_155410.json',
                'simulation_results/Final_Accum_gemini-1.5-flash_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20250805_172651.json',
                'simulation_results/Final_Accum_gemini-1.5-flash_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20250805_175910.json',
                'simulation_results/Final_Accum_gemini-1.5-flash_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20250805_190826.json',
                'simulation_results/Final_Accum_gemini-1.5-flash_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20250805_220801.json'
            ],
            'Claude 3.5 Sonnet': [
                'simulation_results/Original_claude-3-5-sonnet_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20250731_144553.json',
                'simulation_results/Original_claude-3-5-sonnet_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20250804_111059.json',
                'simulation_results/Original_claude-3-5-sonnet_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20250804_113628.json',
                'simulation_results/Original_claude-3-5-sonnet_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20250804_123524.json',
                'simulation_results/Final_Accum_claude-3-5-sonnet_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20250810_150846.json'
            ],
            'GPT-4o': [
                'simulation_results/Final_Accum_gpt-4o_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20250806_130254.json',
                'simulation_results/Final_Accum_gpt-4o_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20250806_171645.json',
                'simulation_results/Final_Accum_gpt-4o_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20250806_205939.json',
                'simulation_results/Final_Accum_gpt-4o_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20250806_232828.json',
                'simulation_results/Final_Accum_gpt-4o_coopGain_2punLoss_2_3_traces_gen10_agents12_top_20250807_052929.json'
            ]
        },
        'Fixed Endowment': {
            'Gemini 2.5 Flash': [
                'simulation_results/Final_Fixed_Endowment_gemini-1.5-flash_coopGain_2punLoss_2_none_gen10_agents12_top_20250814_154916.json',
                'simulation_results/Final_Fixed_Endowment_gemini-1.5-flash_coopGain_2punLoss_2_none_gen10_agents12_top_20250814_161350.json',
                'simulation_results/Final_Fixed_Endowment_gemini-1.5-flash_coopGain_2punLoss_2_none_gen10_agents12_top_20250814_164838.json',
                'simulation_results/Final_Fixed_Endowment_gemini-1.5-flash_coopGain_2punLoss_2_none_gen10_agents12_top_20250814_165734.json',
                'simulation_results/Final_Fixed_Endowment_gemini-1.5-flash_coopGain_2punLoss_2_none_gen10_agents12_top_20250814_171001.json',
            ],
            'Claude 3.5 Sonnet': [
                'simulation_results/Final_Fixed_Endowment_claude-3-5-sonnet_coopGain_2punLoss_2_none_gen10_agents12_top_20250810_171845.json',
                'simulation_results/Final_Fixed_Endowment_claude-3-5-sonnet_coopGain_2punLoss_2_none_gen10_agents12_top_20250810_202137.json',
                'simulation_results/Final_Fixed_Endowment_claude-3-5-sonnet_coopGain_2punLoss_2_none_gen10_agents12_top_20250810_205345.json',
                'simulation_results/Final_Fixed_Endowment_claude-3-5-sonnet_coopGain_2punLoss_2_none_gen10_agents12_top_20250810_212158.json',
                'simulation_results/Final_Fixed_Endowment_claude-3-5-sonnet_coopGain_2punLoss_2_none_gen10_agents12_top_20250810_214314.json'
            ],
            'GPT-4o': [
                'simulation_results/Final_Fixed_Endowment_gpt-4o_coopGain_2punLoss_2_none_gen10_agents12_top_20250810_231826.json',
                'simulation_results/Final_Fixed_Endowment_gpt-4o_coopGain_2punLoss_2_none_gen10_agents12_top_20250811_015331.json',
                'simulation_results/Final_Fixed_Endowment_gpt-4o_coopGain_2punLoss_2_none_gen10_agents12_top_20250811_092837.json',
                'simulation_results/Final_Fixed_Endowment_gpt-4o_coopGain_2punLoss_2_none_gen10_agents12_top_20250811_200244.json',
                'simulation_results/Final_Fixed_Endowment_gpt-4o_coopGain_2punLoss_2_none_gen10_agents12_top_20250811_211005.json'
            ]
        },
        'Single Trace History': {
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
    }
    
    mode_data = {}
    
    for mode_name, models in mode_files.items():
        mode_data[mode_name] = {}
        
        for model_name, file_list in models.items():
            mode_data[mode_name][model_name] = []
            
            for file_path in file_list:
                if os.path.exists(file_path):
                    data = extract_generation_10_data(file_path)
                    if data:
                        mode_data[mode_name][model_name].append(data)
                else:
                    print(f"Warning: File not found: {file_path}")
    
    return mode_data

def create_donation_comparison_plot(mode_data):
    """Create bar plot comparing donation percentages across modes with error bars"""
    
    # Define colors for each model
    model_colors = {
        'Gemini 2.5 Flash': '#4285F4',  # Google Blue
        'Claude 3.5 Sonnet': '#FF6B35',  # Anthropic Orange
        'GPT-4o': '#10A37F'     # OpenAI Green
    }
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Mode names for x-axis
    modes = list(mode_data.keys())
    model_names = ['Gemini 2.5 Flash', 'Claude 3.5 Sonnet', 'GPT-4o']
    
    # Set up bar positions
    x = np.arange(len(modes))
    width = 0.05  # Width of bars
    
    # Plot donation percentages with error bars
    for i, model_name in enumerate(model_names):
        donation_means = []
        donation_errors = []
        
        for mode in modes:
            if model_name in mode_data[mode] and mode_data[mode][model_name]:
                # Calculate mean and standard error
                donations = [d['avg_donation_pct'] for d in mode_data[mode][model_name]]
                mean_donation = np.mean(donations)
                # Standard error = std / sqrt(n)
                std_error = np.std(donations) / np.sqrt(len(donations)) if len(donations) > 1 else 0
                
                donation_means.append(mean_donation)
                donation_errors.append(std_error)
            else:
                donation_means.append(0)
                donation_errors.append(0)
        
        # Plot bars with error bars
        bars = ax.bar(x + i*width, donation_means, width, 
                     label=model_name, color=model_colors[model_name], alpha=0.8)
        
        # Add error bars
        for j, (mean, error) in enumerate(zip(donation_means, donation_errors)):
            if error > 0:  # Only add error bars if there's variation
                ax.errorbar(x[j] + i*width, mean, yerr=error, 
                           color='black', capsize=3, capthick=1, linewidth=1)
    
    ax.set_xlabel('Game Setting')
    ax.set_ylabel('Average Donation Percentage (%)')
    ax.set_title('Average Donation Percentage Comparison at Generation 10')
    ax.set_xticks(x + width)
    ax.set_xticklabels(modes, rotation=45, ha='right')
    ax.set_ylim(0, 100)  # Set y-axis from 0 to 100%
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot to overlay_figures folder
    output_dir = 'overlay_figures'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'modes_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    mode_data = collect_mode_data()
    create_donation_comparison_plot(mode_data)
    
