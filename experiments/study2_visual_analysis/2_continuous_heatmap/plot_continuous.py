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

### PATH SETUP ###
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(SCRIPT_DIR.parent)) 

from utils import true_dgp
from continuous_task import GRID_RESOLUTION

### CONFIGURATION ###
BASE_STUDY_DIR = PROJECT_ROOT / "results" / "study2_visual_analysis" / "2_continuous_heatmap"
INPUT_DIR = BASE_STUDY_DIR / "raw"
PLOT_DIR = BASE_STUDY_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings("ignore")

def load_and_average_results():
    """
    Loads all 'task_*.pkl' files and averages them for all three methods.
    Returns a dict: {'QBC-RF': avg, 'BMA-RF': avg, 'UNREAL': avg}
    """
    files = list(INPUT_DIR.glob("task_*.pkl"))
    
    if not files:
        print(f"Error: No .pkl files found in {INPUT_DIR}")
        return None, 0
        
    print(f"Found {len(files)} result files. Aggregating all three models...")
    
    rf_heatmaps = []
    bma_heatmaps = []   
    unreal_heatmaps = []
    
    for f in files:
        with open(f, 'rb') as file:
            try:
                data = pickle.load(file)
                if "QBC-RF" in data: rf_heatmaps.append(data["QBC-RF"])
                if "BMA-RF" in data: bma_heatmaps.append(data["BMA-RF"])
                if "UNREAL" in data: unreal_heatmaps.append(data["UNREAL"])
            except Exception as e:
                print(f"Warning: Could not load {f.name}: {e}")
    
    avg_heatmaps = {}
    if rf_heatmaps: avg_heatmaps["QBC-RF"] = np.mean(rf_heatmaps, axis=0)
    if bma_heatmaps: avg_heatmaps["BMA-RF"] = np.mean(bma_heatmaps, axis=0)
    if unreal_heatmaps: avg_heatmaps["UNREAL"] = np.mean(unreal_heatmaps, axis=0)
        
    return avg_heatmaps, len(files)




def plot_side_by_side_heatmaps(avg_heatmaps, n_sims):
    """
    Generates a 1x3 side-by-side heatmap comparison (QBC-RF vs BMA-RF vs UNREAL).
    """
    # 1. Setup Figure (1 Row, 3 Cols)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True) 

    # 2. Setup Grids
    x = np.linspace(0, 1, GRID_RESOLUTION)
    y = np.linspace(0, 1, GRID_RESOLUTION)
    xx, yy = np.meshgrid(x, y)
    
    # Ground Truth Boundary
    xx_fine, yy_fine = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
    grid_points_fine = np.c_[xx_fine.ravel(), yy_fine.ravel()]
    Z_true = np.array([true_dgp(p) for p in grid_points_fine]).reshape(xx_fine.shape)

    # 3. Define Plot Configs for all three methods
    configs = [
        (axes[0], "QBC-RF", "QBC-RF (Uniform)"),
        (axes[1], "BMA-RF", "QBC-RF (Weighted)"),
        (axes[2], "UNREAL", "UNREAL (Rashomon Set)")
    ]

    vmin, vmax = 0.0, 1.0 

    # 4. Plotting Loop
    images = [] 
    for ax, key, title in configs:
        if key not in avg_heatmaps:
            ax.set_visible(False)
            continue
            
        heatmap_data = avg_heatmaps[key]
        
        im = ax.contourf(
            xx, yy, heatmap_data,
            levels=np.linspace(0, 1.0, 101), 
            cmap='viridis',
            extend='both' 
        )
        images.append(im)
        
        ax.contour(
            xx_fine, yy_fine, Z_true, 
            colors='white', linestyles='--', 
            levels=[0.5], linewidths=2.5, alpha=0.9
        )
        
        ax.contour(
            xx_fine, yy_fine, Z_true, 
            colors='white', linestyles='--', 
            levels=[0.5], linewidths=2.5, alpha=0.9
        )
        
        label_coords = [0.165, 0.5, 0.835]
        
        for r_idx, y_c in enumerate(label_coords):
            for c_idx, x_c in enumerate(label_coords):
                r_int = int(y_c * (GRID_RESOLUTION - 1))
                c_int = int(x_c * (GRID_RESOLUTION - 1))
                
                try:
                    val = heatmap_data[r_int, c_int]
                except IndexError:
                    val = 0.0                
                text_color = 'white' if val < 0.6 else 'black'
                
                ax.text(
                    x_c, y_c, f"{val:.2f}",
                    color=text_color,
                    ha='center', va='center',
                    fontsize=14, weight='bold',
                    path_effects=[pe.withStroke(linewidth=2, foreground="black" if text_color == 'white' else 'white', alpha=0.5)]
                )

        ax.set_title(title, fontsize=18, weight='bold', pad=15)
        ax.set_xlabel("X1", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(False) 

    axes[0].set_ylabel("X2", fontsize=14)

    # 5. Shared Colorbar
    cbar = fig.colorbar(images[0], ax=axes.ravel().tolist(), shrink=0.8, pad=0.03)
    cbar.set_label('Average Vote Entropy (V.E.)', rotation=270, labelpad=20, fontsize=14)

    # 6. Global Title
    # fig.suptitle(f"Continuous Version Space Uncertainty\n(Averaged over {n_sims} random initializations)", fontsize=22, y=1.02)

    # 7. Save
    output_path = PLOT_DIR / "continuous_heatmap_comparison.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Comparison plot saved to {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    avg_maps, n_runs = load_and_average_results()
    if avg_maps:
        plot_side_by_side_heatmaps(avg_maps, n_runs)
        print(f"\nStudy 3 Plotting Complete. Check: {PLOT_DIR}")