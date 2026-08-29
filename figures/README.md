# Figures Directory

This directory contains all visualization and analysis scripts for processing simulation results and generating plots.

## Cultural Evolution Plots

### **figs_avg_grouped.py**
- **Purpose**: Cultural evolution plotting for full simulation runs
- **Key Features**:
  - Average resources and donation percentages over generations
  - Data extraction from JSON files
- **Use Case**: Plotting multiple runs of a model over generations for original accumulated resource setting

### **figs_overlay.py**
- **Purpose**: Overlay multiple model results for comparison
- **Key Features**:
  - Multiple runs per model with error shading
  - Average lines with standard error bands
  - Color-coded by model family
- **Use Case**: Model comparison across multiple runs for original accumulated resource setting
- **Output**: Saves to `overlay_figures/` directory

### **figs_fixed_endowment.py**
- **Purpose**: Specialized plotting for fixed endowment results
- **Key Features**:
  - Modified donation percentage calculation (out of 10)
  - Fixed endowment data processing
- **Use Case**: Analysis of fixed endowment performance
- **Output**: Saves to both default directory and `overlay_figures/`

### **figs_compare_modes.py**
- **Purpose**: Compare donation percentages for different game modes
- **Key Features**:
  - Accumulated Endowment vs. Fixed Endowment vs. Single Trace History
  - Bar plots with error bars for donation percentages at end of generation 10
- **Use Case**: Plotting comparison of donation percentage using different game settings
- **Output**: Saves to `overlay_figures/`

### **figs_individual_runs.py**
- **Purpose**: Visualization of multiple individual runs
- **Key Features**:
  - 6 subplots (3 for resources, 3 for donations)
  - Individual runs in different colors
  - Black dotted average lines
- **Use Case**: Analysis of individual runs of models

## Strategy Analysis Plots

### **plot_strategy_combos.py**
- **Purpose**: Comprehensive plotting for strategy testing results
- **Key Features**:
  - Suitable for plotting results of strategy_response.py and prompt_strategy_combos.py
  - Strategy-centric plots (how each strategy performs across prompts)
  - Average performance with error bars
  - Support for multiple runs and models
  - Single-prompt and multi-prompt visualization
- **Output**: Multiple PNG files in `strategy_figures/`

### **prompt_heatmaps.py**
- **Purpose**: Heatmap visualizations for strategy analysis
- **Key Features**:
  - Strategy adherence heatmaps (MSE between intended and actual)
  - Full strategy text display
- **Use Case**: Analyzes prompt-strategy combinations and model behavior compared to intended strategy behavior 