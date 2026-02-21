# ==============================================================================
# --- SET UP ---
# ==============================================================================

### LIBRARIES ###
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import warnings
import matplotlib.patheffects as pe

### SET UP SYS ###
SCRIPT_DIR = Path(__file__).resolve().parent
VISUAL_ANALYSIS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = VISUAL_ANALYSIS_DIR.parent.parent
sys.path.append(str(VISUAL_ANALYSIS_DIR))
from utils import create_data_pool, true_dgp
from heatmap_task import N_POINTS, POOL_SEED

### CONFIGURATION ###
BASE_RESULTS_DIR = PROJECT_ROOT / "results" / "study2_visual_analysis" / "2_heat_bubble"
RAW_RESULTS_DIR = BASE_RESULTS_DIR / "raw"
PLOT_OUTPUT_DIR = BASE_RESULTS_DIR / "plots"
PLOT_OUTPUT_DIR.mkdir(exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid') 
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# --- PLOTTING SCRIPT ---
# ==============================================================================

def load_all_results() -> pd.DataFrame:
    """
    Loads all raw .pkl files and combines them into one master DataFrame.
    """
    pkl_files = list(RAW_RESULTS_DIR.glob("task_*.pkl"))
    if not pkl_files:
        print(f"Error: No .pkl files found in {RAW_RESULTS_DIR}")
        print("Please run the 'submit_heatmap.sbatch' script first.")
        return pd.DataFrame()
        
    print(f"Found {len(pkl_files)} result files. Aggregating...")
    
    all_results = []
    for f in pkl_files:
        with open(f, 'rb') as file:
            try:
                task_id = int(f.stem.split('_')[-1])
                df = pickle.load(file)
                df['task_id'] = task_id
                df['point_id'] = df.index
                all_results.append(df)
            except Exception as e:
                print(f"Warning: Could not load or parse {f.name}. Error: {e}")
    
    df_long = pd.concat(all_results, ignore_index=True)
    return df_long

def plot_side_by_side_bubbles(df_pool, df_agg_results, n_sims):
    """
    Generates a side-by-side comparison (QBC-RF vs UNREAL) using the 
    user's preferred aesthetics (Rank, ID, Avg VE text).
    """
    # 1. Setup Figure (1 Row, 2 Cols)
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharey=True)
    
    # 2. Grid for background decision boundary
    xx, yy = np.meshgrid(np.linspace(-0.05, 1.05, 200), np.linspace(-0.05, 1.05, 200))
    grid_points_cont = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([true_dgp(x) for x in grid_points_cont])
    Z = Z.reshape(xx.shape)

    # 3. Define the configuration for the loop
    configs = [
        (axes[0], "QBC-RF", "QBC-RF"),
        (axes[1], "UNREAL", "UNREAL")
    ]

    for ax, selector, title in configs:
        print(f"Plotting {selector}...")
        
        # --- Background: True Boundary ---
        ax.contour(xx, yy, Z, colors='black', linestyles='--', levels=[0.5], linewidths=1.5, alpha=0.6)
        
        # --- Data Prep ---
        df_selector = df_agg_results[df_agg_results['selector'] == selector]
        if df_selector.empty:
            print(f"  Warning: No data for {selector}")
            continue
            
        # Calculate stats
        avg_entropies = df_selector.groupby('point_id')['ve'].mean()
        ranks = avg_entropies.rank(method='first', ascending=False).astype(int)
        
        # --- Plot Points ---
        for idx, row in df_pool.iterrows():
            avg_ve = avg_entropies.get(idx, np.nan)
            rank = ranks.get(idx, 0)
            
            # --- Aesthetics (Matches Original Request) ---
            marker = 'P' if row['label'] == 1 else 'x'      
            color_val = 0.0 if np.isnan(avg_ve) else avg_ve 
            color = plt.cm.coolwarm(color_val)              
            size = 50 + (color_val * 450)                   
            
            # Scatter Bubble
            ax.scatter(
                row['X1_cont'], row['X2_cont'],
                marker=marker,
                c=[color], 
                s=size,
                edgecolors='black',
                linewidth=0.5,
                zorder=10,
                alpha=0.9
            )
            
            # Text Label (Rank, ID, Avg VE)
            text_label = f"Rank: {rank}\nAvg V.E: {avg_ve:.2f}"
            # text_label = f"ID: {idx}\nRank: {rank}\nAvg V.E: {avg_ve:.2f}"
            
            # Add a white outline to text for readability over grid lines
            ax.text(
                row['X1_cont'] + 0.015, row['X2_cont'] + 0.015,
                text_label,
                fontsize=9,
                zorder=11,
                path_effects=[pe.withStroke(linewidth=2, foreground="white", alpha=0.7)]
            )

        # --- Axis Styling ---
        ax.set_title(title, fontsize=18, weight='bold', pad=15)
        ax.set_xlabel("X1", fontsize=14)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)        
        ax.grid(True, linestyle=':', zorder=0)

    axes[0].set_ylabel("X2", fontsize=14)

    # 4. Shared Colorbar
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1.0))
    sm.set_array([])    
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.9, pad=0.05)
    cbar.set_label('Average Vote Entropy (V.E.)', rotation=270, labelpad=20, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # 5. Global Title
    # fig.suptitle(f"Discrete Selection Ranking: Stability over {n_sims} Permutations", fontsize=22, y=1.02)

    # 6. Save
    output_path = PLOT_OUTPUT_DIR / "bubble_comparison_side_by_side.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Comparison plot saved to {output_path.relative_to(PROJECT_ROOT)}")

def main():
    print("--- Starting Heat-Bubble Plot Aggregation ---")

    # 1. Load all result files
    df_long = load_all_results()
    if df_long.empty:
        return
    
    # 2. Get the actual number of sims that ran
    n_sims_found = df_long['task_id'].nunique()
    print(f"Aggregating {n_sims_found} unique simulation results.")

    # 3. "Melt" the dataframe
    df_melted = df_long.melt(
        id_vars=['task_id', 'point_id'],
        value_vars=['QBC-RF', 'UNREAL', 'DUREAL'],
        var_name='selector',
        value_name='ve'
    )

    # 4. Get the "ground truth" data pool
    df_pool = create_data_pool(n_points=N_POINTS, pool_seed=POOL_SEED) 

    # 5. Generate the side-by-side plot
    plot_side_by_side_bubbles(df_pool, df_melted, n_sims_found)
        
    print("\n" + "=" * 30)
    print("=== Plotting Complete ===")
    print(f"All plots saved to: {PLOT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()