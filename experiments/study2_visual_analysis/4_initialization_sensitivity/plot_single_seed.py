### LIBRARIES ###
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import warnings

### PATH SETUP ###
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import helpers
from experiments.study2_visual_analysis.utils import (
    true_dgp, GRID_RESOLUTION, create_data_pool, 
    POOL_SEED
)

### CONFIGURATION ###
BASE_STUDY_DIR = PROJECT_ROOT / "results" / "study2_visual_analysis" / "4_initialization_sensitivity"
INPUT_DIR = BASE_STUDY_DIR / "raw"
PLOT_DIR = BASE_STUDY_DIR / "plots" / "single_seed_examples"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Define N_POOL_POINTS locally to match task script
N_POOL_POINTS = 500 

### SELECT SEEDS HERE ###
# Dictionary mapping Scenario Name -> Task ID (Seed)
# If a scenario isn't listed here, the script will skip it.
TARGET_SEEDS = {
    "A_Random_Corners": 4,        # Try 0, 1, 2...
    "B_Random_Boundary": 104,     # Try 101, 102 (100 might be invalid/skipped)
    "C_Margin_0.85": 204,          # 200 is the first run for this scenario
    "C_Margin_0.55": 304          # 300 is the first run for this scenario
}

plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings("ignore")

def get_target_files():
    """
    Finds the specific .pkl files requested in TARGET_SEEDS.
    """
    if not INPUT_DIR.exists():
        print(f"Error: {INPUT_DIR} does not exist.")
        return {}

    found_files = {}
    
    for scenario, task_id in TARGET_SEEDS.items():
        # Construct the expected filename pattern
        # Format: init_task_{ID}_ScenarioName.pkl
        # We use glob because the ID is zero-padded (0000)
        pattern = f"init_task_{task_id:04d}_{scenario}.pkl"
        
        matches = list(INPUT_DIR.glob(pattern))
        
        if matches:
            found_files[scenario] = matches[0]
            print(f"Found Seed {task_id} for {scenario}")
        else:
            print(f"Warning: Could not find file for {scenario} with Seed {task_id}. Skipping.")
            
    return found_files

def plot_single_run(filepath, df_pool):
    """
    Plots the heatmap for a single run, overlays training points, AND adds numerical labels.
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
        
    scenario = data['scenario']
    run_idx = data['run_idx']
    train_indices = data['train_indices']
    rf_grid = data['QBC-RF']
    unreal_grid = data['UNREAL']
    
    # Get coordinates of the specific 5 points used in this run
    training_points = df_pool.loc[train_indices]
    
    # Setup Grid for Background
    xx, yy = np.meshgrid(np.linspace(0, 1, GRID_RESOLUTION), np.linspace(0, 1, GRID_RESOLUTION))
    
    # Setup for 9 Region Labels (Coordinates)
    x_coords = [0.165, 0.50, 0.835] 
    y_coords = [0.165, 0.50, 0.835]
    
    # Corresponding array indices for a 50x50 grid to sample the value
    array_indices = [
        (8, 8), (8, 25), (8, 41),     # Bottom Row 
        (25, 8), (25, 25), (25, 41),  # Middle Row
        (41, 8), (41, 25), (41, 41)   # Top Row 
    ]

    # Create Figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    configs = [
        (axes[0], rf_grid, "QBC-RF (Random Forest)"),
        (axes[1], unreal_grid, "UNREAL (LFR)")
    ]
    
    for ax, grid_data, title in configs:
        # 1. Plot Heatmap
        im = ax.contourf(
            xx, yy, grid_data, 
            levels=np.linspace(0, 1.01, 100), 
            cmap='viridis', vmin=0.0, vmax=1.0
        )
        
        # 2. Plot True Boundary
        grid_flat = np.c_[xx.ravel(), yy.ravel()]
        Z_true = np.array([true_dgp(x) for x in grid_flat]).reshape(xx.shape)
        ax.contour(xx, yy, Z_true, colors='white', linestyles='--', levels=[0.5], linewidths=2)
        
        # 3. Add Numerical Labels (NEW)
        for i, (y_idx, x_idx) in enumerate(array_indices):
            try:
                value = grid_data[y_idx, x_idx]
            except IndexError:
                value = 0.0
            
            # Calculate specific coordinates for this label
            x_c = x_coords[i % 3]
            y_c = y_coords[i // 3]

            text_color = 'black' if value < 0.5 else 'white'
            
            ax.text(
                x_c, y_c, 
                f"{value:.2f}",
                fontsize=14,
                color=text_color,
                ha='center', va='center',
                weight='bold',
                zorder=9 # Below points, above heatmap
            )

        # 4. OVERLAY TRAINING POINTS
        class_0 = training_points[training_points['label'] == 0]
        class_1 = training_points[training_points['label'] == 1]
        
        # Halo effect for visibility
        ax.scatter(class_0['X1_cont'], class_0['X2_cont'], c='white', s=150, marker='o', edgecolors='black', zorder=10)
        ax.scatter(class_0['X1_cont'], class_0['X2_cont'], c='blue', s=100, marker='o', edgecolors='black', label='Class 0', zorder=11)
        
        ax.scatter(class_1['X1_cont'], class_1['X2_cont'], c='white', s=150, marker='s', edgecolors='black', zorder=10)
        ax.scatter(class_1['X1_cont'], class_1['X2_cont'], c='red', s=100, marker='s', edgecolors='black', label='Class 1', zorder=11)

        ax.set_title(title, fontsize=14, weight='bold')
        ax.set_xlabel("X1")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if title.startswith("UNREAL"):
            ax.legend(loc='upper right', frameon=True, framealpha=0.9)

    axes[0].set_ylabel("X2")
    
    # Super Title
    clean_name = scenario.replace('_', ' ')
    fig.suptitle(f"Single Seed Example: {clean_name}\n(Run ID: {run_idx})", fontsize=18, y=1.05)
    
    # Save
    out_path = PLOT_DIR / f"single_seed_{scenario}_run_{run_idx}.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  > Saved example to: {out_path.name}")

def main():
    print("--- Generating Single Seed Examples ---")
    
    # 1. Re-create the exact data pool used in simulations
    df_pool = create_data_pool(N_POOL_POINTS, POOL_SEED)
    
    # 2. Find specific files
    examples = get_target_files()
    if not examples:
        print("No matching files found.")
        return

    print(f"Plotting {len(examples)} examples...")
    
    # 3. Plot
    for scen, filepath in examples.items():
        plot_single_run(filepath, df_pool)
        
    print(f"\nDone. Check folder: {PLOT_DIR}")

if __name__ == "__main__":
    main()