import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path

### CONFIGURATION ###
# To change comparison, swap BASELINE_ID between "M1" (Random) or "M5" (UNREAL)
BASELINE_ID = "M1"  
TARGET_PERCENTAGES = [0.7, 0.8, 0.9] 

# TOGGLE: Set to False to exclude "Synthetic_*" datasets from the analysis
INCLUDE_SYNTHETIC = False

NAME_MAPPING = {
    "M1": "Random Sampling",
    "M2": "QBC-RF (F=3)",
    "M3": "QBC-RF (F=Sqrt)",
    "M4": "QBC-RF (F=All)",
    "M5": "UNREAL (0.05)",
    "M6": "UNREAL (0.10)",
    "M8": "Uncertainty",
    "M9": "Coreset",
    "M10": "Rashomon-EMC"
}

ORDER = ["UNREAL (0.05)", "UNREAL (0.10)", "Rashomon-EMC", "Uncertainty", "Coreset", "QBC-RF (F=All)", "QBC-RF (F=Sqrt)", "QBC-RF (F=3)"]

def calculate_n_rel(study_root, include_synthetic=True):
    if not study_root.exists():
        print(f"Error: {study_root} not found.")
        return pd.DataFrame()

    dataset_dirs = [d for d in study_root.iterdir() if d.is_dir() and (d / "aggregated").exists()]
    
    # Filter synthetic datasets if requested
    if not include_synthetic:
        dataset_dirs = [d for d in dataset_dirs if "Synthetic" not in d.name]
        
    baseline_label = NAME_MAPPING.get(BASELINE_ID, BASELINE_ID)
    print(f"Found {len(dataset_dirs)} datasets with aggregated results relative to {baseline_label}.")
    
    efficiency_data = []
    method_counts = {label: 0 for label in NAME_MAPPING.values()}

    for ds_dir in dataset_dirs:
        agg_dir = ds_dir / "aggregated"
        
        # 1. Load Baseline results to establish crossing thresholds
        bl_path = agg_dir / f"{BASELINE_ID}_results.pkl"
        if not bl_path.exists():
            continue
        
        try:
            with open(bl_path, "rb") as f:
                bl_data = pickle.load(f)
            
            bl_trace = np.nanmean(bl_data["accuracy_history"], axis=0)
            acc_start = bl_trace[0]
            acc_final = bl_trace[-1]
            total_growth = acc_final - acc_start
            
            # Skip if dataset has no learning signal (growth < 1%)
            if total_growth < 0.01 or np.isnan(total_growth):
                continue

            # 2. Process each method relative to the baseline
            for method_id, method_label in NAME_MAPPING.items():
                if method_id == BASELINE_ID: continue
                
                m_path = agg_dir / f"{method_id}_results.pkl"
                if not m_path.exists(): continue
                
                with open(m_path, "rb") as f:
                    m_data = pickle.load(f)
                
                m_hist = m_data.get("accuracy_history")
                if m_hist is None or np.all(np.isnan(m_hist)): continue
                
                m_trace = np.nanmean(m_hist, axis=0)
                
                # 3. Calculate N_rel for each target milestone
                found_points = False
                for k in TARGET_PERCENTAGES:
                    target_acc = acc_start + (k * total_growth)
                    
                    bl_crossings = np.where(bl_trace >= target_acc)[0]
                    n_baseline = bl_crossings[0] if len(bl_crossings) > 0 else len(bl_trace)
                    
                    m_crossings = np.where(m_trace >= target_acc)[0]
                    n_method = m_crossings[0] if len(m_crossings) > 0 else len(m_trace)
                    
                    if n_baseline > 0:
                        n_rel = n_method / n_baseline
                        efficiency_data.append({
                            "Dataset": ds_dir.name,
                            "Method": method_label,
                            "Target": f"{int(k*100)}% Growth",
                            "N_rel": n_rel
                        })
                        found_points = True
                
                if found_points:
                    method_counts[method_label] += 1
                    
        except Exception as e:
            print(f"      [Warning] Error processing {ds_dir.name}: {e}")

    print("\nProcessing Summary (Valid Datasets per Method):")
    for method, count in method_counts.items():
        if method != baseline_label:
            print(f"  - {method}: {count} datasets")
            
    return pd.DataFrame(efficiency_data)

def plot_efficiency_boxplot(df, output_path):
    if df.empty:
        print("No valid data points collected.")
        return

    plot_df = df[df["Method"].isin(ORDER)].copy()
    plot_df['N_rel_clipped'] = plot_df['N_rel'].clip(upper=2.5)

    plt.figure(figsize=(14, 10))
    sns.set_style("whitegrid")
    palette = sns.color_palette("viridis_r", len(TARGET_PERCENTAGES))
    
    actual_order = [m for m in ORDER if m in plot_df["Method"].unique()]
    
    sns.boxplot(
        data=plot_df, y="Method", x="N_rel_clipped", hue="Target",   
        order=actual_order,
        orient="h", palette=palette, showfliers=True, width=0.7, linewidth=2,
        flierprops={"marker": "x", "markersize": 5, "alpha": 0.5}
    )
    
    baseline_label = NAME_MAPPING.get(BASELINE_ID, BASELINE_ID)
    plt.axvline(1.0, color="#c0392b", linestyle="--", linewidth=2.5, label=f"{baseline_label} (1.0)")
    
    plt.title("Relative Label Efficiency ($N_{rel}$) Across Benchmarks", fontsize=20, pad=20)
    plt.xlabel(f"Labels Required Relative to {baseline_label} (Clipped at 2.5)", fontsize=16)
    plt.ylabel("")
    plt.legend(title="Accuracy Target", loc="upper right", fontsize=12, title_fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"\nPlot saved to: {output_path}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    study_root = project_root / "results" / "study1_active_learning" / "tree_predictor"
    output_dir = project_root / "results" / "study1_active_learning" / "PLOTS" / "Efficiency_Plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use the configuration toggle
    df_results = calculate_n_rel(study_root, include_synthetic=INCLUDE_SYNTHETIC)
    
    if not df_results.empty:
        suffix = "" if INCLUDE_SYNTHETIC else "_NoSynth"
        plot_efficiency_boxplot(df_results, output_dir / f"Relative_Label_Efficiency{suffix}.png")
        
        unreal_id = "M5"
        unreal_label = NAME_MAPPING[unreal_id]
        baseline_label = NAME_MAPPING[BASELINE_ID]
        
        if BASELINE_ID == unreal_id:
            print(f"\nSummary for {unreal_label}:")
            print(f"  This method is currently set as the Baseline.")
            random_data = df_results[df_results["Method"] == NAME_MAPPING["M1"]]
            if not random_data.empty:
                r_stats = random_data["N_rel"]
                print(f"  Random needs {r_stats.median():.3f}x labels compared to UNREAL.")
        else:
            unreal_data = df_results[df_results["Method"] == unreal_label]
            if not unreal_data.empty:
                stats = unreal_data["N_rel"]
                median_val = stats.median()
                print(f"\nSummary for {unreal_label}:")
                print(f"  Median N_rel: {median_val:.3f} ({(1 - median_val)*100:+.1f}% Typical Savings)")
                print(f"  Range: [{stats.min():.2f}, {stats.max():.2f}]")
                worst_idx = stats.idxmax()
                print(f"  Worst Case: {df_results.loc[worst_idx, 'Dataset']} (N_rel={stats.max():.2f})")
                best_idx = stats.idxmin()
                print(f"  Best Case:  {df_results.loc[best_idx, 'Dataset']} (N_rel={stats.min():.2f})")