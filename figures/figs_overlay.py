#!/usr/bin/env python3
"""
Modified figs.py to overlay multiple model results with different colors and average error shading
Enhanced to handle multiple runs per model type with exact file names
"""

import json
import re
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

_TS_RE = re.compile(r'_(\d{8}_\d{6})\.json$')

def _extract_timestamp(path: str) -> str:
    m = _TS_RE.search(os.path.basename(path))
    return m.group(1) if m else ""

_SUBDIR_TO_MODEL_LABEL = {
    "claude": "Claude 3.5 Sonnet",
    "gpt":    "GPT-4o",
    "gemini": "Gemini 2.5 Flash",
}

def discover_model_files(base_dir: str, required_substring: str | None = None, max_runs: int = 5):
    """Group Final_Accum JSONs by model.

    Two layouts supported:
      1. Subfolder layout — base_dir contains subdirectories named 'claude',
         'gpt', 'gemini'; files inside each subfolder are attributed to that
         model regardless of filename. Used for runs where filenames don't
         carry the model string (e.g. Frankenstein output).
      2. Flat layout — base_dir contains all Final_Accum files at the top
         level; model is inferred from the filename substring
         (claude-3-5-sonnet / gpt-4o / gemini-1.5-flash).

    Each group is sorted newest-first by filename timestamp.
    """
    groups = {
        "Claude 3.5 Sonnet": [],
        "GPT-4o": [],
        "Gemini 2.5 Flash": [],
    }

    # ── Layout 1: subfolders named claude / gpt / gemini ──────────────────
    subdir_layout = any(
        os.path.isdir(os.path.join(base_dir, sub))
        for sub in _SUBDIR_TO_MODEL_LABEL
    )
    if subdir_layout:
        for sub, label in _SUBDIR_TO_MODEL_LABEL.items():
            sub_path = os.path.join(base_dir, sub)
            if not os.path.isdir(sub_path):
                continue
            paths = glob.glob(os.path.join(sub_path, "*Final_Accum_*.json"))
            if required_substring:
                paths = [p for p in paths if required_substring in os.path.basename(p)]
            groups[label].extend(paths)
    else:
        # ── Layout 2: flat folder, infer model from filename ──────────────
        paths = glob.glob(os.path.join(base_dir, "*Final_Accum_*.json"))
        basenames = {p: os.path.basename(p) for p in paths}
        if required_substring:
            paths = [p for p in paths if required_substring in basenames[p]]
        for p in paths:
            bn = basenames[p]
            if "Final_Accum_claude-3-5-sonnet_" in bn:
                groups["Claude 3.5 Sonnet"].append(p)
            elif "Final_Accum_gpt-4o_" in bn:
                groups["GPT-4o"].append(p)
            elif "Final_Accum_gemini-1.5-flash_" in bn:
                groups["Gemini 2.5 Flash"].append(p)

    # Sort by timestamp embedded in filename so ordering matches dates.
    timestamps = {}
    for k in list(groups.keys()):
        groups[k].sort(key=_extract_timestamp, reverse=True)
        if max_runs > 0:
            groups[k] = groups[k][:max_runs]
        if not groups[k]:
            del groups[k]
        else:
            timestamps[k] = [_extract_timestamp(p) for p in groups[k]]

    return groups, timestamps

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
    
    # Calculate averages for each generation
    generation_summaries = []
    for gen_num in sorted(generations.keys()):
        gen_data = generations[gen_num]
        
        # Calculate total accumulated resources per agent for this generation.
        # Total wealth = sum of all `resources` values (endowment kept each round)
        #                + sum of all `received` values (donations received go to savings)
        agent_wealth = {}  # agent_name -> total accumulated wealth
        for round_data in gen_data:
            agent_name = round_data['agent_name']
            resources = round_data['resources'] or 0
            received = round_data.get('received', 0) or 0
            agent_wealth[agent_name] = agent_wealth.get(agent_name, 0) + resources + received

        avg_resources = sum(agent_wealth.values()) / len(agent_wealth) if agent_wealth else 0
        
        # Calculate average donation percentage for this generation (includes 0-donation rounds).
        donations = []
        for round_data in gen_data:
            if round_data.get('is_donor', False):
                donated = round_data.get('donated', 0) or 0
                remaining = round_data.get('resources', 0) or 0
                denom = donated + remaining
                donation_pct = (donated / denom) * 100 if denom > 0 else 0
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
    """Process a single model's data file and return resources and donations arrays"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if data has generation_summaries, if not, extract from og_code format
        summaries = data.get('generation_summaries', [])
        if not summaries:
            print(f"Converting og_code format data for {file_path}")
            summaries = extract_data_from_og_code_format(data)
        
        resources = []
        donations = []
        
        for summary in summaries:
            # Extract average resources
            resources_match = re.search(r'Average resources for Generation (\d+): ([\d.]+) units', summary)
            if resources_match:
                avg_resources = float(resources_match.group(2))
                resources.append(avg_resources)
            
            # Extract average donations
            donations_match = re.search(r'Average donation percentage for Generation (\d+): ([\d.]+)%', summary)
            if donations_match:
                avg_donations = float(donations_match.group(2))
                donations.append(avg_donations)
        
        return resources, donations
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [], []

def process_multiple_runs(model_files, max_gens=10):
    """Process multiple runs for each model and return aggregated data"""
    model_data = {}
    
    for model_name, file_paths in model_files.items():
        print(f"\nProcessing {model_name} with {len(file_paths)} files...")
        
        all_resources = []
        all_donations = []
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                resources, donations = process_model_data(file_path, model_name)
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
                'std_resources': std_resources,
                'sem_resources': sem_resources,
                'mean_donations': mean_donations,
                'std_donations': std_donations,
                'sem_donations': sem_donations,
                'resources_array': resources_array,
                'donations_array': donations_array,
                'num_runs': len(all_resources)
            }
        else:
            print(f"Warning: No valid data found for {model_name}")
    
    return model_data

def overlay_model_results(model_files, output_dir='overlay_figures', max_gens=10):
    """Create overlay plots for multiple model results"""
    model_data = process_multiple_runs(model_files, max_gens)
    
    if not model_data:
        print("No valid data found for any model")
        return
    
    # Define colors for each model
    colors = {
        'gemini 1.5 flash': '#4285F4',        # Google Blue
        'gemini 2.5 flash': '#4285F4',          # Google Blue
        'claude 3.5 sonnet': '#FF6B35',        # Anthropic Orange
        'gpt-4o': '#10A37F',                   # OpenAI Green
        'qwen 3.5': '#9B59B6',                 # Purple
        'qwen 2.5 7b': '#9B59B6',              # Purple
    }

    # Map short-name → canonical baseline color so extra-dir lines can inherit
    # the matching baseline model's color.
    _short_color = {
        'claude': colors['claude 3.5 sonnet'],
        'gpt':    colors['gpt-4o'],
        'gemini': colors['gemini 1.5 flash'],
    }

    # Per-line style: extra-dir entries labelled "<batch> (<model>)" inherit
    # the model's baseline color and render as a dashed line so they're visibly
    # distinct from the solid baseline line.
    import re as _re_style
    _extra_re = _re_style.compile(r'\(\s*(claude|gpt|gemini)\s*\)\s*$')
    line_styles = {}  # name -> '-' or '--'
    for name in model_data.keys():
        m = _extra_re.search(name)
        if m and m.group(1) in _short_color:
            colors[name.lower()] = _short_color[m.group(1)]
            line_styles[name] = '--'
        else:
            line_styles[name] = '-'

    # Auto-assign distinct colors to any leftover label not in the palette
    # (e.g. frankenstein tags like 'gen-gemini_evo-gpt_exe-claude').
    _frankenstein_palette = ['#7B2CBF', '#E63946', '#06A77D', '#F4A261',
                             '#264653', '#9D4EDD', '#F72585']
    _i = 0
    for name in model_data.keys():
        if name.lower() not in colors:
            colors[name.lower()] = _frankenstein_palette[_i % len(_frankenstein_palette)]
            _i += 1
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    os.makedirs(output_dir, exist_ok=True)

    # y-axis limits: pad at least 10 units below the min and above the max of
    # the actual data being plotted (across all runs of all models). Donations
    # are additionally clamped to [0, 100] since that's the natural range.
    def _data_ylim(arr_key, clamp=None, pad=10.0):
        all_vals = []
        for data in model_data.values():
            arr = data[arr_key]
            mask = ~np.isnan(arr)
            if mask.any():
                all_vals.append(arr[mask])
        if not all_vals:
            return (0.0, 1.0)
        flat = np.concatenate(all_vals)
        lo, hi = float(np.min(flat)) - pad, float(np.max(flat)) + pad
        if clamp is not None:
            lo = max(lo, clamp[0])
            hi = min(hi, clamp[1])
        return (lo, hi)

    RESOURCES_YLIM = _data_ylim('resources_array')
    DONATIONS_YLIM = _data_ylim('donations_array', clamp=(0.0, 100.0))

    def _draw_individual_lines(ax, arr_key, marker, ylabel, title, ylim):
        """Thin lines for every run, no shading."""
        legend_handles = {}
        for model_name, data in model_data.items():
            arr   = data[arr_key]
            color = colors.get(model_name.lower(), '#000000')
            ls    = line_styles.get(model_name, '-')
            for run_vals in arr:
                valid_mask = ~np.isnan(run_vals)
                valid_gens = np.where(valid_mask)[0] + 1
                valid_vals = run_vals[valid_mask]
                if len(valid_gens) == 0:
                    continue
                line, = ax.plot(valid_gens, valid_vals,
                                color=color, linewidth=1.2, alpha=0.5,
                                linestyle=ls,
                                marker=marker, markersize=3)
                if model_name not in legend_handles:
                    legend_handles[model_name] = line
        ax.set_xlabel('Generation')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(ylim)
        ax.legend(legend_handles.values(), legend_handles.keys(), fontsize=9)
        ax.grid(True, alpha=0.3)

    def _draw_mean_ci(ax, mean_key, sem_key, marker, ylabel, title, ylim):
        """Mean line + 95% CI shaded band (mean ± 1.96 * SEM)."""
        for model_name, data in model_data.items():
            mean = data[mean_key]
            sem  = data[sem_key]
            color = colors.get(model_name.lower(), '#000000')
            ls    = line_styles.get(model_name, '-')
            generations = np.arange(1, len(mean) + 1)
            valid_mask  = ~np.isnan(mean)
            gens_v = generations[valid_mask]
            mean_v = mean[valid_mask]
            ci_v   = 1.96 * sem[valid_mask]
            ax.fill_between(gens_v, mean_v - ci_v, mean_v + ci_v,
                            color=color, alpha=0.2)
            ax.plot(gens_v, mean_v,
                    color=color, linewidth=2.5, linestyle=ls,
                    marker=marker, markersize=5,
                    label=model_name)
        ax.set_xlabel('Generation')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(ylim)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # ── Figure 1: individual runs ──────────────────────────────────────────
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    _draw_individual_lines(ax1, 'resources_array', 'o',
                           'Average Final Resources',
                           'Cultural Evolution of Cooperation - Resources',
                           RESOURCES_YLIM)
    _draw_individual_lines(ax2, 'donations_array', 's',
                           'Average Donation Percentage (%)',
                           'Cultural Evolution of Cooperation - Donations',
                           DONATIONS_YLIM)
    plt.tight_layout()
    path_lines = os.path.join(output_dir, 'model_comparison_individual_runs.png')
    fig1.savefig(path_lines, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"Plot saved to: {path_lines}")

    # ── Figure 2: mean ± 95% CI ───────────────────────────────────────────
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))
    _draw_mean_ci(ax3, 'mean_resources', 'sem_resources', 'o',
                  'Average Final Resources',
                  'Cultural Evolution of Cooperation - Resources (mean ± 95% CI)',
                  RESOURCES_YLIM)
    _draw_mean_ci(ax4, 'mean_donations', 'sem_donations', 's',
                  'Average Donation Percentage (%)',
                  'Cultural Evolution of Cooperation - Donations (mean ± 95% CI)',
                  DONATIONS_YLIM)
    plt.tight_layout()
    path_mean = os.path.join(output_dir, 'model_comparison_mean_ci.png')
    fig2.savefig(path_mean, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"Plot saved to: {path_mean}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Overlay multiple model results.")
    parser.add_argument("--base_dir",   default="simulations/new_baseline",
                        help="Directory containing Final_Accum_*.json files")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save the figure (default: first --frankenstein_dirs if given, else --base_dir)")
    parser.add_argument("--required",   default=None,
                        help="Optional substring filter on filenames")
    parser.add_argument("--max_runs",   type=int, default=5,
                        help="Max runs per model to include (0 = unlimited, default: 5)")
    parser.add_argument("--extra_dirs", nargs="*", default=[],
                        help="Additional run directories (frankenstein, heterogeneous, "
                             "prompt-variant, etc.) to overlay as extra lines. "
                             "Each becomes a single line labelled by either an embedded "
                             "tag (gen-X_evo-Y_exe-Z, het-X, prompt-X, mode-X) or by the "
                             "directory's basename.")
    args = parser.parse_args()

    BASE     = args.base_dir
    REQUIRED = args.required
    # Default output location:
    #   - If extra dirs were provided, save into the first one so the comparison
    #     plot lives next to the data it characterises.
    #   - Otherwise save into base_dir (next to the baseline data).
    if args.output_dir:
        OUT_DIR = args.output_dir
    elif args.extra_dirs:
        OUT_DIR = args.extra_dirs[0]
    else:
        OUT_DIR = BASE

    model_files, timestamps = discover_model_files(BASE, required_substring=REQUIRED, max_runs=args.max_runs)

    # Add each extra directory as its own overlay line. Label-derivation order:
    #   1. Frankenstein role tag in the filename: gen-X_evo-Y_exe-Z
    #   2. Any of the other recognised tags in the filename: het-X, prompt-X, mode-X
    #   3. The directory's basename, with a leading "array_" or "array" stripped
    import re as _re
    _frank_re   = _re.compile(r'Final_Accum_(gen-[^_]+_evo-[^_]+_exe-[^_]+)_')
    _other_tag  = _re.compile(r'_(het-[^_]+|prompt-[^_]+|mode-[^_]+)')
    def _label_for_extra_dir(fdir, sample_filename):
        m = _frank_re.search(sample_filename)
        if m:
            return m.group(1)
        tags = _other_tag.findall(sample_filename)
        if tags:
            return "+".join(tags)
        return _re.sub(r'^array_?', '', os.path.basename(fdir.rstrip("/")))

    def _add_extra(label, fpaths):
        if not fpaths:
            return
        if args.max_runs > 0:
            fpaths = sorted(fpaths, key=_extract_timestamp, reverse=True)[:args.max_runs]
        model_files[label] = fpaths
        timestamps[label] = [_extract_timestamp(p) for p in fpaths]

    for fdir in args.extra_dirs:
        batch_name = os.path.basename(fdir.rstrip("/"))
        # If the dir uses the claude/gpt/gemini subfolder layout, register one
        # extra line per subfolder, labelled "<batch>_<model>".
        sub_paths = {sub: glob.glob(os.path.join(fdir, sub, "Final_Accum_*.json"))
                     for sub in _SUBDIR_TO_MODEL_LABEL}
        if any(sub_paths.values()):
            for sub, paths in sub_paths.items():
                if paths:
                    _add_extra(f"{batch_name} ({sub})", paths)
            continue

        # Flat layout — register a single extra line.
        fpaths = sorted(glob.glob(os.path.join(fdir, "Final_Accum_*.json")))
        if not fpaths:
            print(f"  [extra_dirs] no Final_Accum files in {fdir} — skipping")
            continue
        label = _label_for_extra_dir(fdir, os.path.basename(fpaths[0]))
        _add_extra(label, fpaths)

    if not model_files:
        print(f"No matching Final_Accum files found in {BASE} (required_substring={REQUIRED!r}).")
    else:
        print("Discovered files per model (newest -> oldest by filename timestamp):")
        for model, paths in model_files.items():
            ts = timestamps[model]
            print(f"- {model}: {len(paths)} (first={ts[0] if ts else 'n/a'} last={ts[-1] if ts else 'n/a'})")

        total_files = sum(len(files) for files in model_files.values())
        print(f"Processing {total_files} files...")
        overlay_model_results(model_files, output_dir=OUT_DIR)
