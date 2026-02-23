### Libraries ###
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

### Methods to Plot
# METHODS_TO_PLOT = ["M1", "M2", "M3", "M4", "M8", "M9", "M5", "M10"]
METHODS_TO_PLOT = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]

### XLIMS ###
DATASET_XLIMS = {
    # # Synthetic
    # "Synthetic_XOR_Baseline": (0,None),
    # "Synthetic_XOR_Alpha_25": (0,None),
    # "Synthetic_XOR_Alpha_50": (0,None),
    # "Synthetic_XOR_Alpha_75": (0,None),
    # "Synthetic_XOR_Alpha_100": (0,None),
    # "Synthetic_XOR_Phi_05": (0,None),
    # "Synthetic_XOR_Phi_10": (0,None),
    # "Synthetic_XOR_Phi_25": (0,None),
    # "Synthetic_XOR_Phi_45": (0,None),
    # # Real
    # "anneal": (0,None),
    # "bank_marketing": (0,1000),
    # "banknote_authentication": (0,400),
    # "bar-7": (0,400),
    # "biodeg": (0,300),
    # "breast_cancer_wisconsin": (0,None),
    # "car_evaluation": (0,None),
    # "cheap_restaurant": (0,None),
    # "coffee_house": (0,700),
    # "expensive_restaurant": (0,400),
    # "haberman": (0,100),
    # "hepatitis": (0,None),
    # "hypothyroid": (0,200),
    # "lymph": (0,None),
    # "monk1": (0,None),
    # "monk2": (0,None),
    # "monk3": (0,None),
    # "primary-tumor": (0,None),
    # "spect": (0,None),
    # "tic-tac-toe": (0,None),
    # "vote": (0,75),
    # "yeast": (0,600),
}
### CONFIGURATION ###
METHOD_LABELS = {
    "M1": "Random Sampling",
    "M2": "RF (Feat=3)",
    "M3": "RF (Feat=Sqrt)",
    "M4": "RF (Feat=All)",
    "M5": "UNREAL (Uniform)",
    "M6": "Uncertainty Sampling",
    "M7": "Coreset (Hamming)",
    "M8": "UNREAL (Bayesian)"
}

METHOD_COLORS = {
    "M1": "gray",
    "M2": "#2ca02c",
    "M3": "#1f77b4",
    "M4": "#d62728",
    "M5": "#ff7f0e",
    "M6": "#52b6c2",
    "M7": "#ffe600",
    "M8": "#14dbdeb6"
}

METHOD_STYLES = {
    "M1": "--",
    "M2": ":",
    "M3": ":",
    "M4": ":",
    "M5": "-",
    "M6": "-.",
    "M7": "-.",
    "M8": "-"
}

METRICS_TO_PLOT = [
    ("accuracy_history", "Accuracy", "lower right"),
    ("f1_history", "F1 Score", "lower right"),
    ("tree_edit_distance_history", "Tree Edit Distance (TED)", "upper right"),
    ("rashomon_size_history", "Rashomon Set Size", "upper right"),
    ("committee_size_history", r"Effective Committee Size ($\exp(H)$)", "upper right"),
    ("oracle_agreement_history", "Oracle Agreement", "lower right")
]

### Load aggregated data ###
def load_aggregated_data(aggregated_dir):
    
    ## Set up ##
    data = {}

    ## Open results each method ##
    for pkl_file in aggregated_dir.glob("M*_results.pkl"):
        method_key = pkl_file.stem.split("_")[0] 
        try:
            with open(pkl_file, "rb") as f:
                data[method_key] = pickle.load(f)
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
    return data

### Plot metric ###
def plot_metric(data, metric_key, y_label, save_path, dataset_name, legend_loc="best", show_legend=True):

    ## Set up ##
    # plt.figure(figsize=(10, 4))
    plt.figure(figsize=(5, 5))
    sorted_keys = sorted(data.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 99)
    has_data = False

    ## Plot each method ##
    for method in sorted_keys:

        # Set up #
        if method not in METHODS_TO_PLOT: continue

        # Restrict Size/Committee plots to only UNREAL variants (M5, M8)
        if metric_key in ["rashomon_size_history", "committee_size_history"]:
            if method not in ["M5", "M8"]:
                continue

        history = data[method].get(metric_key)        
        if history is None or (isinstance(history, np.ndarray) and history.size == 0): 
            continue
        has_data = True
        
        # Calculate stats #
        means = np.nanmean(history, axis=0)
        if metric_key == "tree_edit_distance_history":
            if np.all(means == -1.0): # Filter out methods that don't support TED (like RF or LogReg)
                continue
        stds = np.nanstd(history, axis=0) / np.sqrt(history.shape[0]) 
        iterations = np.arange(len(means)) + 1
        
        # Aesthetics #
        label = METHOD_LABELS.get(method, method)
        color = METHOD_COLORS.get(method, "black")
        style = METHOD_STYLES.get(method, "-")
        linewidth = 1.5
        alpha_fill = 0.1

        # Plot #
        plt.plot(iterations, means, label=label, color=color, linestyle=style, linewidth=linewidth)
        plt.fill_between(iterations, means - stds, means + stds, color=color, alpha=alpha_fill)

    ## If no data ##
    if not has_data:
        print(f"  -> Skipping {metric_key} (No data found)")
        plt.close()
        return

   ## Aesthetics ##
    plt.xlabel("Active Learning Iterations")
    plt.ylabel(y_label)
    # plt.title(f"{dataset_name}: {y_label}")    
    if dataset_name in DATASET_XLIMS:
        plt.xlim(DATASET_XLIMS[dataset_name])    
    if show_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  -> Saved {save_path.name}")

### Elapsed time bar chart ###
def plot_time_bar_chart(data, save_path, dataset_name):
    """
    Plots a bar chart comparing the average elapsed time + std dev errors.
    """

    ## Set up ##
    methods = []
    means = []
    stds = []
    colors = []
    sorted_keys = sorted(data.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 99)

    ## For each method ##
    for method in sorted_keys:

        # Calculate statistics #
        if method not in METHODS_TO_PLOT: continue
        time_mean = data[method].get("elapsed_time_mean")
        time_std = data[method].get("elapsed_time_std")
        if time_mean is not None:
            methods.append(METHOD_LABELS.get(method, method))
            means.append(time_mean)
            stds.append(time_std if time_std is not None else 0)
            colors.append(METHOD_COLORS.get(method, "gray"))

    ## If missing ##
    if not methods:
        return

    ## Plot ##
    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height,
                 f'{height:.1f}s', ha='center', va='bottom', fontweight='bold')

    ## Aesthetics ##
    plt.ylabel("Total Elapsed Time (seconds)")
    plt.title(f"{dataset_name}: Computational Cost")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  -> Saved {save_path.name}")

### Generate standalone legend ###
def generate_legend(output_path, ncol=4):
    handles = []
    labels = []
    for method in METHODS_TO_PLOT:
        display_name = METHOD_LABELS.get(method)
        color = METHOD_COLORS.get(method)
        ls = METHOD_STYLES.get(method, "-")
        if display_name is None or color is None:
            continue
        line = plt.Line2D([0], [0], color=color, linestyle=ls, linewidth=2.5)
        handles.append(line)
        labels.append(display_name)

    fig = plt.figure(figsize=(16, 2))
    fig_legend = fig.legend(handles, labels, loc="center", frameon=True,
                            ncol=ncol, fontsize=12, handlelength=3)
    plt.gca().axis("off")
    fig.savefig(output_path,
                bbox_inches=fig_legend.get_window_extent().transformed(
                    fig.dpi_scale_trans.inverted()),
                dpi=300, transparent=True)
    plt.close(fig)
    print(f"Legend saved to: {output_path}")


def plot_variance_metric(data, metric_key, y_label, save_path, dataset_name, show_legend=True):
    """
    Plots the variance across seeds for a specific metric history.
    """
    plt.figure(figsize=(5, 5))
    sorted_keys = sorted(data.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 99)
    has_data = False

    for method in sorted_keys:
        if method not in METHODS_TO_PLOT: continue
        history = data[method].get(metric_key)        
        if history is None or (isinstance(history, np.ndarray) and history.size == 0): 
            continue        
        history = np.array(history)
        if history.ndim == 1: history = history.reshape(1, -1) 
        has_data = True
        
        # Calculate Variance across seeds (axis 0)
        variances = np.nanvar(history, axis=0)
        iterations = np.arange(len(variances)) + 1
        
        label = METHOD_LABELS.get(method, method)
        color = METHOD_COLORS.get(method, "black")
        style = METHOD_STYLES.get(method, "-")

        plt.plot(iterations, variances, label=label, color=color, linestyle=style, linewidth=2)

    if not has_data:
        plt.close()
        return

    plt.xlabel("Active Learning Iterations")
    plt.ylabel(f"Variance of {y_label}")
    plt.title(f"{dataset_name}: {y_label} Stability (Variance)")    
    if dataset_name in DATASET_XLIMS:
        plt.xlim(DATASET_XLIMS[dataset_name])    
    if show_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  -> Saved Variance Plot: {save_path.name}")

### MAIN ###
def main():

    ## Arguments ##
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=False, help="...")
    parser.add_argument("--study_dir", type=str, default="study1_active_learning")
    parser.add_argument("--no-legend", dest="show_legend", action="store_false")
    parser.add_argument("--legend-only", action="store_true")
    args = parser.parse_args()

    ## Paths ##
    project_root = Path(__file__).resolve().parent.parent.parent

    ## Legend-only mode ##
    if args.legend_only:
        legend_dir = project_root / "results" / "study1_active_learning" / "PLOTS"
        legend_dir.mkdir(parents=True, exist_ok=True)
        generate_legend(output_path=legend_dir / "benchmark_legend.png")
        return

    ## Normal mode — dataset required ##
    if not args.dataset:
        parser.error("--dataset is required unless using --legend-only")

    dataset_dir = project_root / "results" / args.study_dir / args.dataset
    agg_dir = dataset_dir / "aggregated"
    img_dir = dataset_dir / "accuracy_images"
    if not agg_dir.exists():
        print(f"[ERROR] Aggregated results not found at: {agg_dir}")
        print("Did you run src/utils/aggregate_results.py first?")
        return
    img_dir.mkdir(parents=True, exist_ok=True)
    
    ## Dataset ##
    dataset_title = args.dataset
    print(f"--- Plotting Results for {dataset_title} ---")
    data = load_aggregated_data(agg_dir)
    if not data:
        print("No data found to plot.")
        return

    ## Trace plots ##
    for metric_key, y_label, leg_loc in METRICS_TO_PLOT:
        plot_metric(data, metric_key, y_label, img_dir / f"{metric_key}.png", 
                    dataset_title, leg_loc, show_legend=args.show_legend) 
        
    ## Accuracy bariance plots ##
    plot_variance_metric(data, "accuracy_history", "Accuracy", 
                         img_dir / "accuracy_variance.png", 
                         dataset_title, show_legend=args.show_legend)
        
    ## Bar charts ##
    plot_time_bar_chart(data, img_dir / "elapsed_time.png", dataset_title)

### Main ###
if __name__ == "__main__":
    main()