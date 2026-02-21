import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from pathlib import Path

### CONFIGURATION ###
METRICS_TO_AUC = [
    "accuracy_history", 
    "oracle_agreement_history", 
    "tree_edit_distance_history",
    "oracle_agreement_history"
]

# Reference baseline for the FINAL PLOT ratios
BASELINE_ID = "M5"  # Comparing everyone to Random Sampling

# Simple Budget Truncation Configuration
BUDGET_FRACTION = 0.9

METHOD_LABELS = {
    "M1": "Random",
    "M2": "RF (p=3)",
    "M3": "RF (p=Sqrt)",
    "M4": "RF (p=All)",
    "M5": "UNREAL",
    "M6": "Uncertainty",
    "M7": "Coreset"
}

def calculate_auc(history_arr, fraction):
    """
    Calculates AUC truncated at a fixed percentage of the total budget.
    """
    if history_arr is None or len(history_arr) == 0:
        return None
    
    # 1. Calculate Mean Curve
    mean_curve = np.nanmean(history_arr, axis=0)
    
    # 2. Determine Truncation Point (% of total iterations)
    total_iters = len(mean_curve)
    t_max = int(total_iters * fraction)
    
    # Ensure we have at least some points to integrate
    t_max = max(2, t_max)
    
    truncated_curve = mean_curve[:t_max]
    x = np.arange(len(truncated_curve))
    
    # 3. Integration
    try:
        auc_val = np.trapz(truncated_curve, x)
    except AttributeError:
        auc_val = np.trapezoid(truncated_curve, x)
        
    return auc_val, t_max

def load_auc_data(study_root, metric_key):
    """
    Discovers datasets and computes AUC using the % budget rule.
    """
    records = []
    all_dirs = [d for d in study_root.iterdir() if d.is_dir()]
    dataset_dirs = sorted([d.name for d in all_dirs if (d / "aggregated").exists() and d.name != "PLOTS"])

    for ds_name in dataset_dirs:
        ds_path = study_root / ds_name / "aggregated"
        
        for pkl_file in ds_path.glob("M*_results.pkl"):
            method_id = pkl_file.stem.split("_")[0]
            if method_id not in METHOD_LABELS: continue
                
            try:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)
                
                history = data.get(metric_key)
                res = calculate_auc(history, fraction=BUDGET_FRACTION)
                
                if res is not None:
                    auc_val, t_max = res
                    records.append({
                        "Dataset": ds_name,
                        "Method": METHOD_LABELS[method_id],
                        "MethodID": method_id,
                        "AUC": auc_val,
                        "t_max": t_max
                    })
            except Exception:
                pass 
                
    return pd.DataFrame(records)

def plot_relative_heatmap(auc_df, output_path, metric_key):
    """Generates the heatmap using the % budget AUCs."""
    lower_is_better = "distance" in metric_key.lower()
    pivot = auc_df.pivot(index='Method', columns='Dataset', values='AUC')
    
    baseline_label = METHOD_LABELS.get(BASELINE_ID)
    if baseline_label not in pivot.index:
        print(f"Error: Baseline {baseline_label} not found.")
        return

    # Calculate the Ratio relative to Random Sampling
    relative_pivot = pivot.div(pivot.loc[baseline_label], axis=1)
    
    # Sort methods for the manuscript story
    preferred_order = ["M1", "M5", "M6", "M10", "M8", "M9", "M4", "M3", "M2"]
    unique_order = [METHOD_LABELS[m] for m in preferred_order if m in METHOD_LABELS and METHOD_LABELS[m] in relative_pivot.index]
    relative_pivot = relative_pivot.reindex(unique_order)

    plt.figure(figsize=(24, 11))
    
    if lower_is_better:
        # Distance: Green for < 1.0 (Lower is better)
        cmap = sns.diverging_palette(130, 10, as_cmap=True, s=90, l=60, center="light")
        vmin, vmax = 0.4, 1.6 
    else:
        # Accuracy: Green for > 1.0 (Higher is better)
        cmap = sns.diverging_palette(10, 130, as_cmap=True, s=90, l=60, center="light")
        vmin, vmax = 0.85, 1.15

    sns.heatmap(
        relative_pivot, annot=True, fmt=".3f", cmap=cmap, center=1.0,
        vmin=vmin, vmax=vmax, linewidths=0.5,
        cbar_kws={'label': f'Efficiency Ratio (vs {baseline_label})'}
    )
    
    display_metric = metric_key.replace("_history", "").replace("_", " ").title()
    plt.title(f"Active Learning Efficiency: {display_metric} AUC ({int(BUDGET_FRACTION*100)}% Budget Truncation)\n(Green = Better than {baseline_label})", fontsize=18, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  -> Saved {metric_key} heatmap to: {output_path.name}")

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    study_root = project_root / "results" / "study1_active_learning/tree_predictor"
    output_dir = project_root / "results"/ "study1_active_learning" / "PLOTS" / "AUC_Plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting Budget-Based AUC Analysis ({int(int(BUDGET_FRACTION*100))}% Truncation)...")
    
    for metric in METRICS_TO_AUC:
        print(f"--- Processing {metric} ---")
        auc_df = load_auc_data(study_root, metric)
        if not auc_df.empty:
            filename = f"AUC_Heatmap_{metric}_{int(BUDGET_FRACTION*100)}PCT_vs_{BASELINE_ID}.png"
            plot_relative_heatmap(auc_df, output_dir / filename, metric)

if __name__ == "__main__":
    main()