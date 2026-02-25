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
    "anneal",                               # Dataset 1
    "bank_marketing",                       # Dataset 2
    "bar-7",                                # Dataset 3
    "breast_cancer_wisconsin",              # Dataset 4
    "car_evaluation",                       # Dataset 5
    "cheap_restaurant",                     # Dataset 6
    "coffee_house",                         # Dataset 7
    "compas",
    "expensive_restaurant",                 # Dataset 8
    "haberman",                             # Dataset 9
    "hepatitis",                            # Dataset 10
    "hypothyroid",                          # Dataset 11
    "kr-vs-kp",                             # Dataset 12
    "lymph",                                # Dataset 13
    # "monk1",                                # Dataset 14 (replace with FICO [or bank_note/spect] if finish on time)
    "monk2",                                # Dataset 15 
    # "monk3",                                # Dataset 16 (replace with COMPAS [or bank_note/spect] if finish on time)
    "primary-tumor",                        # Dataset 17
    "spect",
    "tic-tac-toe",                          # Dataset 19
    "vote",                                 # Dataset 20
    "yeast"                                 # Dataset 21
] 
METHOD_LABELS = {
    "M1": "Random",
    # "M2": "QBC-RF (p=3)",
    "M3": "QBC-RF (p=sqrt)",
    "M4": "QBC-RF (p=d)",
    "M5": "UNREAL",
    "M6": "Uncertainty",
    "M7": "Coreset",
    "M8": "BREAL",
    "M9": "QBC-RF (Weighted, p=sqrt)",
    "M10": "QBC-RF (Weighted, p=d)",
}
PREFERRED_METHOD_ORDER = ["M8", "M5", "M3", "M4", "M9", "M10", "M6", "M7", "M1",]

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
    LABEL_SIZE = 16 
    CELL_SIZE = 12   
    CBAR_SIZE = 16   
    
    # 1. Pivot and Filter
    pivot = auc_df.pivot(index='Method', columns='Dataset', values='AUC')
    existing_columns = [ds for ds in DATASET_ORDER if ds in pivot.columns]
    pivot = pivot[existing_columns]
    
    # 2. Calculate Relative Efficiency
    baseline_label = METHOD_LABELS.get(BASELINE_ID)
    if baseline_label not in pivot.index: return
    
    relative_pivot = pivot.div(pivot.loc[baseline_label], axis=1)    
    unique_method_labels = [METHOD_LABELS[m] for m in PREFERRED_METHOD_ORDER if m in METHOD_LABELS and METHOD_LABELS[m] in relative_pivot.index]
    relative_pivot = relative_pivot.reindex(unique_method_labels)

    # 3. Define Aesthetics based on Metric
    lower_is_better = "distance" in metric_key.lower()
    if lower_is_better:
        # Green is good (below baseline), Red is bad
        cmap = sns.diverging_palette(10, 130, as_cmap=True, s=90, l=60, center="light")
        vmin, vmax = 0.4, 1.6 
    else:
        # Green is good (above baseline), Red is bad
        cmap = sns.diverging_palette(130, 10, as_cmap=True, s=90, l=60, center="light")
        vmin, vmax = 0.85, 1.15

    # 4. Plot
    fig, ax = plt.subplots(figsize=(24, 10))

    sns.heatmap(
        relative_pivot, 
        annot=True, 
        fmt=".3f", 
        cmap=cmap, 
        center=1.0,
        vmin=vmin, 
        vmax=vmax, 
        linewidths=3,    
        linecolor='white',
        annot_kws={
            "size": 10, 
            "weight": "bold",
            "family": "sans-serif" 
        }, 
        cbar_kws={
            'label': f'Efficiency Ratio against {baseline_label}',
            'shrink': 0.7, 
            'pad': 0.01   
        },
        ax=ax
    )
    
    # 5. Final Polish
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.xticks(rotation=45, ha='right', fontsize=LABEL_SIZE)
    plt.yticks(fontsize=LABEL_SIZE)

    # Clean up Colorbar text
    cbar_axis = ax.collections[0].colorbar.ax
    cbar_axis.yaxis.label.set_size(CBAR_SIZE)
    cbar_axis.tick_params(labelsize=CBAR_SIZE - 2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
