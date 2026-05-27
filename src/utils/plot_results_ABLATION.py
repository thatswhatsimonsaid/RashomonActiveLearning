### Summary ###
"""
Specialized plotting script for the Epsilon Sensitivity Ablation Study.
Generates THREE separate plots for Accuracy, ECS, and Rashomon Size.
"""

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
    "M1": "Random",
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
    "M1": "#7f7f7f",  "M2": "#2ca02c", # Grey and Green
    "M3": E_005, "M7": E_005, 
    "M4": E_020, "M8": E_020,
    "M5": E_050, "M9": E_050,
    "M6": E_100, "M10": E_100
}

# --- DISTINCT LINE STYLES ---
METHOD_STYLES = {
    "M1": ":",      # Random: Dotted
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

plt.rcParams.update({'font.size': FONT_SIZE})

def load_aggregated_data(aggregated_dir):
    data = {}
    for pkl_file in aggregated_dir.glob("M*_results.pkl"):
        method_key = pkl_file.stem.replace("_results", "")
        try:
            with open(pkl_file, "rb") as f:
                data[method_key] = pickle.load(f)
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
    return data

def plot_ablation_diagnostic(data, dataset_name, save_dir):
    """Generates three individual standalone figures."""
    
    metrics = [
        ("accuracy_history", "Test Accuracy", False, "1_accuracy_stability"),
        ("committee_size_history", "Effective Committee Size (ECS)", False, "2_version_space_filtering"),
        ("rashomon_size_history", "Total Rashomon Size $|R|$", True, "3_rashomon_volume")
    ]

    for metric_key, y_label, use_log, file_name in metrics:
        # Create a fresh figure for each metric
        plt.figure(figsize=(8, 6))
        
        for method in METHODS_TO_PLOT:
            if method not in data: continue
            history = data[method].get(metric_key)
            if history is None or len(history) == 0: continue

            means = np.nanmean(history, axis=0)
            stds = np.nanstd(history, axis=0) / np.sqrt(history.shape[0])
            iters = np.arange(len(means)) + 1

            color = METHOD_COLORS.get(method, "black")
            style = METHOD_STYLES.get(method, "-")
            lw = 2.5 if method in ["M7", "M8", "M9", "M10"] else 1.5
            
            plt.plot(iters, means, label=METHOD_LABELS.get(method, method), 
                     color=color, linestyle=style, linewidth=lw)
            plt.fill_between(iters, means - stds, means + stds, color=color, alpha=0.08)

        plt.xlabel("Active Learning Iterations")
        plt.ylabel(y_label)
        if use_log: plt.yscale('log')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.title(f"{y_label}\n({dataset_name})", fontweight='bold')
        
        # Place legend outside to the right
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        plt.tight_layout()
        save_path = save_dir / f"{file_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> Generated: {save_path.name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--study_dir", type=str, required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    agg_dir = project_root / "results" / args.study_dir / args.dataset / "aggregated"
    img_dir = project_root / "results" / args.study_dir / args.dataset / "ablation_plots"
    img_dir.mkdir(parents=True, exist_ok=True)

    data = load_aggregated_data(agg_dir)
    sns.set_theme(style="whitegrid")
    plot_ablation_diagnostic(data, args.dataset, img_dir)

if __name__ == "__main__":
    main()