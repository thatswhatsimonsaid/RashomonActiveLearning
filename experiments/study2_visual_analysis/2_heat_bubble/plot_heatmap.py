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
# Ensure project root is in path for utils and src imports
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(VISUAL_ANALYSIS_DIR))

from utils import create_data_pool, true_dgp
# Import N_POINTS and POOL_SEED to ensure alignment with the task script
try:
    from heatmap_task import N_POINTS, POOL_SEED
except ImportError:
    # Fallback if running standalone
    N_POINTS, POOL_SEED = 10, 0

### CONFIGURATION ###
BASE_RESULTS_DIR = PROJECT_ROOT / "results" / "study2_visual_analysis" / "2_heat_bubble"
RAW_RESULTS_DIR = BASE_RESULTS_DIR / "raw"
PLOT_OUTPUT_DIR = BASE_RESULTS_DIR / "plots"
PLOT_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
plt.style.use('seaborn-v0_8-whitegrid') 
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# --- PLOTTING SCRIPT ---
# ==============================================================================

def load_all_results() -> pd.DataFrame:
    """Loads all raw .pkl files and combines them into one master DataFrame."""
    pkl_files = list(RAW_RESULTS_DIR.glob("task_*.pkl"))
    if not pkl_files:
        print(f"Error: No .pkl files found in {RAW_RESULTS_DIR}")
        return pd.DataFrame()
        
    print(f"Found {len(pkl_files)} result files. Aggregating...")
    
    all_results = []
    for f in pkl_files:
        with open(f, 'rb') as file:
            try:
                # Extract task_id from filename (e.g., task_001.pkl)
                task_id_str = f.stem.split('_')[-1]
                task_id = int(task_id_str)
                df = pickle.load(file)
                df['task_id'] = task_id
                df['point_id'] = df.index
                all_results.append(df)
            except Exception as e:
                print(f"Warning: Could not load {f.name}. Error: {e}")
    
    return pd.concat(all_results, ignore_index=True)

def plot_comparison_bubbles(df_pool, df_agg_results, n_sims):
    """
    Generates a 3-panel comparison: QBC-RF (Uniform) vs BMA-RF (Weighted) vs UNREAL.
    """
    # 1. Setup Figure (1 Row, 3 Cols for the full ablation)
    fig, axes = plt.subplots(1, 3, figsize=(25, 8), sharey=True)
    
    # 2. Grid for background decision boundary
    xx, yy = np.meshgrid(np.linspace(-0.05, 1.05, 200), np.linspace(-0.05, 1.05, 200))
    grid_points_cont = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([true_dgp(x) for x in grid_points_cont]).reshape(xx.shape)

    # 3. Define the configuration for the loop (Matching your new heatmap_task.py keys)
    configs = [
        (axes[0], "QBC-RF", "QBC-RF (Uniform)"),
        (axes[1], "BMA-RF", "BMA-RF (Weighted)"),
        (axes[2], "UNREAL", "UNREAL (Rashomon)")
    ]

    for ax, selector, title in configs:
        print(f"Plotting {selector}...")
        
        # --- Background: True Boundary ---
        ax.contour(xx, yy, Z, colors='black', linestyles='--', levels=[0.5], linewidths=1.5, alpha=0.3)
        
        # --- Data Prep ---
        df_selector = df_agg_results[df_agg_results['selector'] == selector]
        if df_selector.empty:
            ax.text(0.5, 0.5, "No Data", ha='center')
            continue
            
        # Calculate stats: Mean uncertainty per point across all combinations
        avg_entropies = df_selector.groupby('point_id')['ve'].mean()
        # Rank 1 = Highest uncertainty (most important to label)
        ranks = avg_entropies.rank(method='first', ascending=False).astype(int)
        
        # --- Plot Points ---
        for idx, row in df_pool.iterrows():
            avg_ve = avg_entropies.get(idx, 0.0)
            rank = ranks.get(idx, 0)
            
            # Marker logic: Positive (+) for Class 1, Cross (x) for Class 0
            marker = 'P' if row['label'] == 1 else 'x'      
            color = plt.cm.coolwarm(avg_ve)              
            size = 100 + (avg_ve * 800) # Scale size by uncertainty
            
            ax.scatter(
                row['X1_cont'], row['X2_cont'],
                marker=marker, c=[color], s=size,
                edgecolors='black', linewidth=0.8, zorder=10, alpha=0.9
            )
            
            # Label: Rank and Score
            text_label = f"R:{rank}\n{avg_ve:.2f}"
            ax.text(
                row['X1_cont'] + 0.02, row['X2_cont'] + 0.02,
                text_label, fontsize=10, zorder=11, weight='bold',
                path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.8)]
            )

        # --- Styling ---
        ax.set_title(title, fontsize=20, weight='bold', pad=15)
        ax.set_xlabel("X1 (Signal)", fontsize=14)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)        
        ax.grid(True, linestyle=':', alpha=0.6)

    axes[0].set_ylabel("X2 (Signal)", fontsize=14)

    # 4. Colorbar and Legend
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=1.0))
    sm.set_array([])    
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
    cbar.set_label('Mean Weighted Entropy', fontsize=16, labelpad=15)

    # 5. Save
    output_path = PLOT_OUTPUT_DIR / "bubble_ablation_comparison.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {output_path}")

def main():
    print("--- Starting Heat-Bubble Plot Aggregation ---")
    df_long = load_all_results()
    if df_long.empty: return
    
    # Aligning the melted variables with your heatmap_task.py keys
    df_melted = df_long.melt(
        id_vars=['task_id', 'point_id'],
        value_vars=['QBC-RF', 'BMA-RF', 'UNREAL'],
        var_name='selector',
        value_name='ve'
    )

    df_pool = create_data_pool(n_points=N_POINTS, pool_seed=POOL_SEED) 
    plot_comparison_bubbles(df_pool, df_melted, df_long['task_id'].nunique())
    print("=== Plotting Complete ===")

if __name__ == "__main__":
    main()