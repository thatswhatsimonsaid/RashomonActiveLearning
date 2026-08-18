### Libraries ###
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

### CONFIGURATION ###
FONT_SIZE = 14
METHODS_TO_PLOT = ["M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10"]

METHOD_LABELS = {
    "M2": "QBC-RF (sqrt)",
    "M3": r"UNREAL ($\epsilon=0.05$)",
    "M4": r"UNREAL ($\epsilon=0.20$)",
    "M5": r"UNREAL ($\epsilon=0.50$)",
    "M6": r"UNREAL ($\epsilon=1.00$)",
    "M7": r"BREAL ($\epsilon=0.05$)",
    "M8": r"BREAL ($\epsilon=0.20$)",
    "M9": r"BREAL ($\epsilon=0.50$)",
    "M10": r"BREAL ($\epsilon=1.00$)",
}

# --- HIGH CONTRAST COLOR PALETTE ---
E_005 = "#1f77b4" # Strong Blue
E_020 = "#17becf" # Cyan/Teal
E_050 = "#ff7f0e" # Bright Orange
E_100 = "#d62728" # Deep Red

METHOD_COLORS = {
    "M2": "#2ca02c", # Green
    "M3": E_005, "M7": E_005, 
    "M4": E_020, "M8": E_020,
    "M5": E_050, "M9": E_050,
    "M6": E_100, "M10": E_100
}

METHOD_STYLES = {
    "M2": "-.",     # QBC-RF: Dash-Dot
    "M3": "--",     # UNREAL: Dashed
    "M4": "--", 
    "M5": "--", 
    "M6": "--",
    "M7": "-",      # BREAL: Solid
    "M8": "-", 
    "M9": "-", 
    "M10": "-"
}

METRICS_TO_PLOT = [
    ("accuracy_history", "Test Accuracy", False, "1_accuracy_stability"),
    ("committee_size_history", "Effective Committee Size (ECS)", False, "2_version_space_filtering"),
    ("rashomon_size_history", "Total Rashomon Size $|R|$", True, "3_rashomon_volume")
]

def load_aggregated_data(aggregated_dir):
    data = {}
    for pkl_file in aggregated_dir.glob("M*_results.pkl"):
        method_key = pkl_file.stem.split("_")[0]
        try:
            with open(pkl_file, "rb") as f:
                data[method_key] = pickle.load(f)
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
    return data

def plot_metric(data, metric_key, y_label, use_log, save_path, dataset_name, show_legend=False):
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': FONT_SIZE})
    plt.figure(figsize=(8, 8))
    
    has_data = False

    for method in METHODS_TO_PLOT:
        if method not in data: continue
        history = data[method].get(metric_key)
        if history is None or (isinstance(history, np.ndarray) and history.size == 0): 
            continue
        has_data = True

        means = np.nanmean(history, axis=0)
        stds = np.nanstd(history, axis=0) / np.sqrt(history.shape[0])
        iters = np.arange(len(means)) + 1

        color = METHOD_COLORS.get(method, "black")
        style = METHOD_STYLES.get(method, "-")
        lw = 2.5 if method in ["M7", "M8", "M9", "M10"] else 1.5
        
        plt.plot(iters, means, label=METHOD_LABELS.get(method, method), 
                 color=color, linestyle=style, linewidth=lw)
        plt.fill_between(iters, means - stds, means + stds, color=color, alpha=0.08)

    if not has_data:
        print(f"  -> Skipping {metric_key} (No data found)")
        plt.close()
        return

    plt.xlabel("Active Learning Iterations")
    plt.ylabel(y_label)
    if use_log: plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.6)
    # plt.title(f"{y_label}\n({dataset_name})", fontweight='bold')
    
    if show_legend:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved {save_path.name}")

def generate_legend(output_path, ncol=3):
    sns.set_theme(style="white") 
    plt.rcParams.update({'font.size': FONT_SIZE})
    
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(111)
    ax.axis('off') 
    
    for method in METHODS_TO_PLOT:
        lw = 2.5 if method in ["M7", "M8", "M9", "M10"] else 1.5
        ax.plot([], [], 
                label=METHOD_LABELS[method], 
                color=METHOD_COLORS[method], 
                linestyle=METHOD_STYLES[method], 
                linewidth=lw)
    
    legend = ax.legend(loc='center', ncol=ncol, frameon=True, fancybox=True, shadow=False)
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    
    fig.savefig(output_path, dpi=300, bbox_inches=bbox.expanded(1.05, 1.1))
    plt.close(fig)
    print(f"Legend saved to: {output_path}")

### MAIN ###
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument("--study_dir", type=str, default="study2_ablation_study/ABS")
    parser.add_argument("--with-legend", dest="show_legend", action="store_true")
    parser.add_argument("--legend-only", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent

    if args.legend_only:
        legend_dir = project_root / "results" / "study2_ablation_study" / "PLOTS"
        legend_dir.mkdir(parents=True, exist_ok=True)
        generate_legend(output_path=legend_dir / "ablation_legend.png", ncol=3)
        return

    if not args.dataset:
        parser.error("--dataset is required unless using --legend-only")

    dataset_dir = project_root / "results" / args.study_dir / args.dataset
    agg_dir = dataset_dir / "aggregated"
    img_dir = dataset_dir / "ablation_plots"
    
    if not agg_dir.exists():
        print(f"[ERROR] Aggregated results not found at: {agg_dir}")
        return
        
    img_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_title = args.dataset
    print(f"--- Plotting Ablation Results for {dataset_title} ---")
    data = load_aggregated_data(agg_dir)
    if not data:
        print("No data found to plot.")
        return

    for metric_key, y_label, use_log, file_name in METRICS_TO_PLOT:
        plot_metric(data, metric_key, y_label, use_log, 
                    img_dir / f"{file_name}.png", 
                    dataset_title, show_legend=args.show_legend) 

if __name__ == "__main__":
    main()