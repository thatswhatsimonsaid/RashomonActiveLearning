### LIBRARIES ###
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.stats import chi2

### INPUT ###
METRICS_TO_EXCLUDE = ["selection_history", "elapsed_time"]
FIG_SIZE = (10, 4)
CONFIDENCE_INTERVAL_ALPHA = 0.15
CRITICAL_VALUE_Z = 1.96  


### AESTHETICS AND CONFIGURATION ###
## NOTE: This should ideally be kept in sync with 'experiments/master_config.py'!!! ##
METHOD_LEGEND_MAPPING = {
    "M1": "Passive (Random Forest)", 
    "M2": "Passive (GPC)", 
    "M3": "Passive (BNN)",
    "M4": "BALD (BNN)", 
    "M5": "BALD (GPC)", 
    "M6": "UNREAL (TreeFarms QBC-U)",
    "M7": "DUREAL (TreeFarms QBC-S)", 
    "M8": "QBC (Random Forest)",
}

## COLOR (Model Type) ##
METHOD_COLORS = {
    "M1": "gray",
    "M2": "mediumseagreen", 
    "M5": "darkgreen",
    "M3": "lightcoral", 
    "M4": "darkred",
    "M6": "blue", 
    "M7": "cornflowerblue",
    "M8": "black",
}

## LINESTYLE (Selector Type) ##
METHOD_LINESTYLES = {
    "M1": ":", 
    "M2": ":", 
    "M3": ":",
    "M4": "-", 
    "M5": "-",
    "M6": "--",  
    "M7": "-.",  
    "M8": "--",  
}

### PLOT MEAN ###
def create_mean_trace_plot(ax, results_dict, x_axis, y_label, title):
    """Generates the mean trace plot on a given matplotlib axis."""

    ## PLOT EACH METHOD ##
    for method_name, results_df in results_dict.items():

        # Mean and standard error #
        mean_trace = results_df.mean(axis=0)
        std_err = results_df.std(axis=0) / np.sqrt(len(results_df))
        
        # Aesthetics #
        legend_label = METHOD_LEGEND_MAPPING.get(method_name, method_name)
        color = METHOD_COLORS.get(method_name, None)
        linestyle = METHOD_LINESTYLES.get(method_name, '-')

        # Plot mean #
        ax.plot(x_axis, mean_trace, label=legend_label, color=color, linestyle=linestyle)
        ax.fill_between(
            x_axis,
            mean_trace - CRITICAL_VALUE_Z * std_err,
            mean_trace + CRITICAL_VALUE_Z * std_err,
            alpha=CONFIDENCE_INTERVAL_ALPHA,
            color=color
        )
    
    # Labels #
    ax.set_xlabel("Percent of Total Data Labeled for Training")
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=11)
    ax.legend(loc='lower right')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

### PLOT VARIANCE ###
def create_variance_trace_plot(ax, results_dict, x_axis, y_label, title):
    """Generates the variance trace plot on a given matplotlib axis."""

    ## PLOT EACH METHOD ##
    for method_name, results_df in results_dict.items():

        # Set up #
        n_simulations = len(results_df)
        variance_trace = results_df.var(axis=0)
        
        # Confidence Intervals #
        lower_chi2 = chi2.ppf(0.025, df=n_simulations - 1)
        upper_chi2 = chi2.ppf(0.975, df=n_simulations - 1)
        
        lower_bound = (n_simulations - 1) * variance_trace / upper_chi2
        upper_bound = (n_simulations - 1) * variance_trace / lower_chi2

        # Aesthetics #
        legend_label = METHOD_LEGEND_MAPPING.get(method_name, method_name)
        color = METHOD_COLORS.get(method_name, None)
        linestyle = METHOD_LINESTYLES.get(method_name, '-')

        ax.plot(x_axis, variance_trace, label=legend_label, color=color, linestyle=linestyle)
        ax.fill_between(x_axis, lower_bound, upper_bound, alpha=CONFIDENCE_INTERVAL_ALPHA, color=color)
    
    ## Labels ##
    ax.set_xlabel("Percent of Total Data Labeled for Training")
    ax.set_ylabel(f"Variance of {y_label}")
    ax.set_title(f"Variance of Performance on {title}", fontsize=9)
    ax.legend(loc='upper right')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

### MAIN SCRIPT LOGIC ###
def generate_plots_for_dataset(dataset_name: str, project_root: Path):
    """Finds all aggregated CSVs for a dataset, groups them by metric, and generates plots."""
    print(f"--- Generating plots for dataset: {dataset_name} ---")
    
    ## DIRECTORIES ##
    aggregated_csv_dir = project_root / "results" / dataset_name / "aggregated"
    output_image_dir = project_root / "results" / "images" / dataset_name
    output_image_dir.mkdir(parents=True, exist_ok=True)

    if not aggregated_csv_dir.exists():
        print(f"  > [ERROR] Aggregated results directory not found: {aggregated_csv_dir}")
        return

    ## FIND AND GROUP CSV FILES BY METRIC ##
    metric_files = {}
    for csv_file in aggregated_csv_dir.glob("M*.csv"):
        match = re.search(r'M\d+_(.+)\.csv', csv_file.name)
        if match:
            metric_name = match.group(1)
            if metric_name not in metric_files:
                metric_files[metric_name] = []
            metric_files[metric_name].append(csv_file)
    
    if not metric_files:
        print(f"  > No aggregated CSV files found in {aggregated_csv_dir}.")
        return

    ### PLOTTING PARAMETERS ###
    initial_train_prop = 0.16
    candidate_pool_prop = 0.64
    
    ## GENERATE PLOTS FOR EACH METRIC ##
    for metric, file_list in metric_files.items():
        if metric in METRICS_TO_EXCLUDE:
            print(f"  > Skipping metric: {metric}")
            continue
            
        print(f"  > Plotting metric: {metric}...")
        results_to_plot = {}
        for csv_path in file_list:
            method_name = csv_path.name.split('_')[0]
            results_to_plot[method_name] = pd.read_csv(csv_path, index_col=0)
        
        # Determine x-axis based on number of iterations in the first DataFrame
        num_iterations = results_to_plot[list(results_to_plot.keys())[0]].shape[1]
        x_axis = (initial_train_prop + (np.arange(num_iterations) / (num_iterations - 1)) * candidate_pool_prop) * 100

        # Create plot titles and labels
        plot_title = f"Active Learning Performance on {dataset_name} Dataset"
        y_label = " ".join(word.capitalize() for word in metric.split('_'))

        # Create Mean Plot
        fig_mean, ax_mean = plt.subplots(figsize=FIG_SIZE)
        create_mean_trace_plot(ax_mean, results_to_plot, x_axis, y_label, plot_title)
        output_path_mean = output_image_dir / f"{metric}_mean.png"
        fig_mean.savefig(output_path_mean, bbox_inches='tight', dpi=300)
        plt.close(fig_mean)
        print(f"    - Saved mean plot to {output_path_mean.relative_to(project_root)}")

        # Create Variance Plot
        fig_var, ax_var = plt.subplots(figsize=FIG_SIZE)
        create_variance_trace_plot(ax_var, results_to_plot, x_axis, y_label, dataset_name)
        output_path_var = output_image_dir / f"{metric}_variance.png"
        fig_var.savefig(output_path_var, bbox_inches='tight', dpi=300)
        plt.close(fig_var)
        print(f"    - Saved variance plot to {output_path_var.relative_to(project_root)}")

### SCRIPT EXECUTION ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate trace plots from aggregated simulation results.")
    parser.add_argument("--dataset", type=str, help="The name of a single dataset to plot.")
    parser.add_argument("--all", action="store_true", help="Flag to run plotting for all available datasets.")
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    RESULTS_DIR = PROJECT_ROOT / "results"

    if args.all:
        dataset_names = [d.name for d in RESULTS_DIR.iterdir() if (d / "aggregated").is_dir()]
        for name in dataset_names:
            generate_plots_for_dataset(name, PROJECT_ROOT)
    elif args.dataset:
        generate_plots_for_dataset(args.dataset, PROJECT_ROOT)
    else:
        print("Please specify a dataset to plot with --dataset <name> or use --all.")

