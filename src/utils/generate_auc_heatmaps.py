### Libraries ###
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import argparse
from pathlib import Path

### CONFIGURATION ###
METRICS_TO_AUC = [
    "accuracy_history", 
    "oracle_agreement_history", 
    "tree_edit_distance_history"
]
BASELINE_ID = "M8"  

# Strict column order for the heatmap
DATASET_ORDER = [
    "Synthetic_XOR_Baseline", 
    "Synthetic_XOR_Phi_05",
    "Synthetic_XOR_Phi_10",
    "Synthetic_XOR_Phi_25", 
    "Synthetic_XOR_Phi_45",
    "Synthetic_XOR_Alpha_50",
    "Synthetic_XOR_Alpha_25", 
    "Synthetic_XOR_Alpha_75", 
    "Synthetic_XOR_Alpha_100",
    "anneal",
     "bank_marketing", 
    "bar-7", 
    "breast_cancer_wisconsin",
    "car_evaluation", 
    "cheap_restaurant", 
    "coffee_house", 
    "expensive_restaurant", 
    "haberman", 
    "hepatitis", 
    "hypothyroid", 
    "lymph", 
    "monk1", 
    "monk2", 
    "monk3", 
    "primary-tumor", 
    "spect", 
    "tic-tac-toe", 
    "vote", 
    "yeast"
] 

METHOD_LABELS = {
    "M1": "Random",
    "M2": "RF (p=3)",
    "M3": "RF (p=Sqrt)",
    "M4": "RF (p=All)",
    "M5": "UNREAL (Uniform)",
    "M6": "Uncertainty",
    "M7": "Coreset",
    "M8": "UNREAL (Bayesian)" 
}
PREFERRED_METHOD_ORDER = ["M8", "M5", "M1", "M6", "M7", "M4", "M3", "M2"]

def calculate_auc(history_arr, fraction):
    if history_arr is None or len(history_arr) == 0:
        return None
    mean_curve = np.nanmean(history_arr, axis=0)
    total_iters = len(mean_curve)
    t_max = int(total_iters * fraction)
    t_max = max(2, t_max)
    truncated_curve = mean_curve[:t_max]
    x = np.arange(len(truncated_curve))
    try:
        auc_val = np.trapezoid(truncated_curve, x)
    except AttributeError:
        auc_val = np.trapezoid(truncated_curve, x)
    return auc_val, t_max

def load_auc_data(study_root, metric_key, budget_fraction): 
    records = []
    for ds_name in DATASET_ORDER:
        ds_path = study_root / ds_name / "aggregated"
        
        if not ds_path.exists():
            print(f"  [Warning] Directory not found: {ds_name}")
            continue

        for pkl_file in ds_path.glob("M*_results.pkl"):
            method_id = pkl_file.stem.split("_")[0]
            if method_id not in METHOD_LABELS: continue
            try:
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)
                history = data.get(metric_key)
                res = calculate_auc(history, fraction=budget_fraction)
                if res is not None:
                    auc_val, t_max = res
                    records.append({
                        "Dataset": ds_name,
                        "Method": METHOD_LABELS[method_id],
                        "MethodID": method_id,
                        "AUC": auc_val
                    })
            except Exception:
                pass 
    return pd.DataFrame(records)

def plot_relative_heatmap(auc_df, output_path, metric_key, budget_fraction): 
    lower_is_better = "distance" in metric_key.lower()
    pivot = auc_df.pivot(index='Method', columns='Dataset', values='AUC')
    
    existing_columns = [ds for ds in DATASET_ORDER if ds in pivot.columns]
    pivot = pivot[existing_columns]
    baseline_label = METHOD_LABELS.get(BASELINE_ID)

    if baseline_label not in pivot.index: return
    relative_pivot = pivot.div(pivot.loc[baseline_label], axis=1)    
    unique_method_labels = [METHOD_LABELS[m] for m in PREFERRED_METHOD_ORDER if m in METHOD_LABELS and METHOD_LABELS[m] in relative_pivot.index]
    relative_pivot = relative_pivot.reindex(unique_method_labels)

    plt.figure(figsize=(26, 10))

    if lower_is_better:
        # Green if Other/UNREAL > 1.0 (UNREAL is smaller/better)
        cmap = sns.diverging_palette(10, 130, as_cmap=True, s=90, l=60, center="light")
        vmin, vmax = 0.4, 1.6 
    else:
        # Green if Other/UNREAL < 1.0 (UNREAL is larger/better)
        cmap = sns.diverging_palette(130, 10, as_cmap=True, s=90, l=60, center="light")
        vmin, vmax = 0.85, 1.15

    sns.heatmap(
        relative_pivot, annot=True, fmt=".3f", cmap=cmap, center=1.0,
        vmin=vmin, vmax=vmax, linewidths=0.5,
        cbar_kws={'label': f'Efficiency Ratio (vs {baseline_label})'}
    )
    
    # display_metric = metric_key.replace("_history", "").replace("_", " ").title()
    # plt.title(f"Efficiency Ratios: {display_metric} ({int(budget_fraction*100)}% Budget)\n"
            #   f"Green = {baseline_label} Wins | Red = Baseline Wins", fontsize=18, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget_fraction", type=float, default=1.0, help="Fraction of the budget to truncate AUC at (0.0 to 1.0)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    study_root = project_root / "results" / "study1_active_learning/tree_predictor"
    output_dir = project_root / "results"/ "study1_active_learning" / "PLOTS" / "AUC_Plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    budget_pct = int(args.budget_fraction * 100)
    print(f" Starting Budget-Based AUC Analysis ({budget_pct}% Truncation)...")
    
    for metric in METRICS_TO_AUC:
        print(f"--- Processing {metric} ---")
        auc_df = load_auc_data(study_root, metric, args.budget_fraction)
        if not auc_df.empty:
            filename = f"AUC_Heatmap_{metric}_{budget_pct}PCT_vs_{BASELINE_ID}.png"
            plot_relative_heatmap(auc_df, output_dir / filename, metric, args.budget_fraction)

if __name__ == "__main__":
    main()
