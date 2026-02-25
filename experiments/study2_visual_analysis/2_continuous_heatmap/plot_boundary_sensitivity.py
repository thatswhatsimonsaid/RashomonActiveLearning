import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import warnings
from scipy import stats
from pathlib import Path

### PATH SETUP ###
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

### IMPORTS ###
from experiments.study2_visual_analysis.utils import (
    distance_to_boundary,
    run_localized_analysis 
)
from continuous_task import GRID_RESOLUTION


### CONFIGURATION ###
BASE_RESULTS_DIR = PROJECT_ROOT / "results" / "study2_visual_analysis" / "2_continuous_heatmap"
RAW_RESULTS_DIR = BASE_RESULTS_DIR / "raw"
PLOT_OUTPUT_DIR = BASE_RESULTS_DIR / "plots"
PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Localization Parameters
DISTANCE_METRIC = 'axis' 
NEAR_THRESHOLD = 0.2  
FAR_THRESHOLD = 0.2   

plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings("ignore")

def load_and_average_data() -> pd.DataFrame:
    """Loads all raw heatmaps, aggregates V.E., and calculates distance."""
    pkl_files = list(RAW_RESULTS_DIR.glob("task_*.pkl"))
    if not pkl_files:
        return pd.DataFrame()

    # Added bma_heatmaps list
    rf_heatmaps, bma_heatmaps, unreal_heatmaps = [], [], []
    
    for f in pkl_files:
        with open(f, 'rb') as file:
            data = pickle.load(file)
            rf_heatmaps.append(data["QBC-RF"])
            if "BMA-RF" in data:
                bma_heatmaps.append(data["BMA-RF"])
            unreal_heatmaps.append(data["UNREAL"])

    # Build the DataFrame including 've_bma'
    df = pd.DataFrame({
        've_rf': np.mean(rf_heatmaps, axis=0).ravel(),
        've_bma': np.mean(bma_heatmaps, axis=0).ravel(), 
        've_unreal': np.mean(unreal_heatmaps, axis=0).ravel() 
    })
    
    # Grid coordinates
    x = np.linspace(0, 1, GRID_RESOLUTION)
    y = np.linspace(0, 1, GRID_RESOLUTION)
    xx, yy = np.meshgrid(x, y)
    df['X1'], df['X2'] = xx.ravel(), yy.ravel()
    
    # Signed Distance
    df['dist_to_boundary'] = df[['X1', 'X2']].apply(
        lambda x: distance_to_boundary(x, metric=DISTANCE_METRIC), axis=1
    )
    return df

def plot_boundary_sensitivity(df_data: pd.DataFrame):
    """Generates the line plot with highlighted Localization Zones."""
    fig, ax = plt.subplots(figsize=(10, 4))

    # Binning for smooth lines
    df_binned = df_data.assign(
        Dist_Bin=pd.cut(df_data['dist_to_boundary'], bins=60)
    ).groupby('Dist_Bin', observed=True).agg(
        Mean_Dist=('dist_to_boundary', 'mean'),
        Mean_VE_RF=('ve_rf', 'mean'),
        Mean_VE_BMA=('ve_bma', 'mean'),     
        Mean_VE_UNREAL=('ve_unreal', 'mean')
    ).reset_index()

    # Plot lines 
    ax.plot(df_binned['Mean_Dist'], df_binned['Mean_VE_UNREAL'], label='UNREAL', color='darkblue', linewidth=3.5)
    ax.plot(df_binned['Mean_Dist'], df_binned['Mean_VE_BMA'], label='QBC-RF-Uniform', color='forestgreen', linewidth=2.5, linestyle='--')
    ax.plot(df_binned['Mean_Dist'], df_binned['Mean_VE_RF'], label='QBC-RF-Weighted', color='darkred', linewidth=3.5)
    
    # Highlight Zones
    ax.axvspan(-NEAR_THRESHOLD, NEAR_THRESHOLD, color='green', alpha=0.1, label='Boundary Zone')
    ax.axvspan(df_data['dist_to_boundary'].min(), -FAR_THRESHOLD, color='gray', alpha=0.1, label='Distal Region')
    ax.axvspan(FAR_THRESHOLD, df_data['dist_to_boundary'].max(), color='gray', alpha=0.1)
    ax.axvline(0.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel("Signed Distance from Boundary (d)", fontsize=14)
    ax.set_ylabel("Average Vote Entropy (V.E.)", fontsize=14)
    ax.legend(loc='upper right', frameon=True, fontsize=12)
    
    filename = PLOT_OUTPUT_DIR / f"localized_sensitivity_{DISTANCE_METRIC}.png"
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Generated Plot: {filename.name}")

def save_stats_table(df_results, stats_dict, output_dir: Path, near_thresh=0.1, far_thresh=0.2):
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "boundary_stats_table.tex"

    # 1. Define Zones
    near_mask = df_results['dist_to_boundary'].abs() <= near_thresh
    far_mask = df_results['dist_to_boundary'].abs() >= far_thresh
    
    # 2. Calculate Means for all three methods
    rf_near = df_results.loc[near_mask, 've_rf'].mean()
    rf_far = df_results.loc[far_mask, 've_rf'].mean()
    
    bma_near = df_results.loc[near_mask, 've_bma'].mean()
    bma_far = df_results.loc[far_mask, 've_bma'].mean()
    bma_bci = bma_near / bma_far if bma_far != 0 else 1.0

    unreal_near = df_results.loc[near_mask, 've_unreal'].mean()
    unreal_far = df_results.loc[far_mask, 've_unreal'].mean()
    p_val = stats_dict['Interaction_Pval']
    if p_val < 0.0001:
        p_str = f"p < 10^{{{int(np.floor(np.log10(p_val)))}}}"
    else:
        p_str = f"p = {p_val:.4f}"

    # 4. Build Table Rows
    row_rf = f"QBC-RF-Uniform & {rf_near:.3f} & {rf_far:.3f} & {stats_dict['BCI_RF']:.3f} & -- \\\\"
    row_bma = f"RF-Weighted & {bma_near:.3f} & {bma_far:.3f} & {bma_bci:.3f} & -- \\\\"
    row_unreal = f"UNREAL & {unreal_near:.3f} & {unreal_far:.3f} & \\textbf{{{stats_dict['BCI_UNREAL']:.3f}}} & {stats_dict['Interaction_Beta']:.3f} (${p_str}$) \\\\"

    # 5. Assemble and Save
    latex_table = r'''\begin{table}[htbp]
\centering
\caption{Concentration of Disagreement and Decay Gradients.}
\label{tab:BoundaryStats}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
\textbf{Method} & \textbf{$H_{\text{man.}}$} & \textbf{$H_{\text{dist.}}$} & \textbf{BCI} & \textbf{Decay $\beta_3$ ($p$)} \\ 
\midrule
''' + row_rf + '\n' + row_bma + '\n' + row_unreal + r'''
\bottomrule
\end{tabular}%
}
\end{table}'''

    with open(file_path, "w") as f:
        f.write(latex_table)
    
    print(f"\n--- LaTeX Table Generated: {file_path.name} ---")
### MAIN ###    
if __name__ == "__main__":
    print("--- Starting Boundary Sensitivity Analysis ---")
    df_results = load_and_average_data()

    if not df_results.empty:
        # 1. Run the statistical analysis
        stats_results = run_localized_analysis(
            df_results,
            near_thresh=NEAR_THRESHOLD,
            far_thresh=FAR_THRESHOLD
        )
        
        # 2. Save the LaTeX table
        save_stats_table(
            df_results, 
            stats_results, 
            PLOT_OUTPUT_DIR,
            near_thresh=NEAR_THRESHOLD,
            far_thresh=FAR_THRESHOLD
        )
        
        # 3. Generate the plot
        plot_boundary_sensitivity(df_results)
    else:
        print("No data found to analyze.")

    print("--- Analysis Complete ---")