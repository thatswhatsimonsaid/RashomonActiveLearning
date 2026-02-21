### LIBRARIES ###
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import pickle
import itertools

### PATH SETUP ###
SCRIPT_DIR = Path(__file__).resolve().parent
VISUAL_ANALYSIS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = VISUAL_ANALYSIS_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(VISUAL_ANALYSIS_DIR))

from utils import (
    create_data_pool,
    get_rf_committee,
    get_rashomon_committee,
    get_entropies_for_grid,
    BINARIZED_FEATURES
)

### PARAMETERS ####
N_POOL_POINTS = 500      
N_INITIAL_POINTS = 3  
N_COMMITTEE = 10
POOL_SEED = 0     
MODEL_SEED = 0 
GRID_RESOLUTION = 50
REGULARIZATION = 0.001
THRESHOLD = 0.01 

### OUTPUT SETUP ###
BASE_OUTPUT_DIR = PROJECT_ROOT / "results" / "study2_visual_analysis" / "3_continuous_heatmap"
RAW_RESULTS_DIR = BASE_OUTPUT_DIR / "raw"
RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True) 
warnings.filterwarnings("ignore", category=UserWarning) 

# ==============================================================================
# --- MAIN TASK LOGIC ---
# ==============================================================================
def run_task(task_id: int):
    """
    Runs one simulation to map the uncertainty landscape.
    """
    print(f"--- Running Continuous Heatmap Task ID {task_id} ---")
    
    # 1. Create a large data pool (consistent pool for all runs)
    df_pool = create_data_pool(n_points=N_POOL_POINTS, pool_seed=POOL_SEED)
    
    # 2. Get the initial training set (unique set per task/seed)
    np.random.seed(task_id)
    train_indices = np.random.choice(
        df_pool.index, 
        size=N_INITIAL_POINTS, 
        replace=False
    )
    df_train = df_pool.loc[train_indices]
    X_train_bin = df_train[BINARIZED_FEATURES]
    y_train = df_train["label"]

    # Cold start safety check: Need both classes to train meaningful boundaries
    if len(y_train.unique()) < 2:
        print(f"  [Task {task_id} SKIPPED] Single-class training set.")        
        results = {k: np.zeros((GRID_RESOLUTION, GRID_RESOLUTION)) for k in ["QBC-RF", "UNREAL", "DUREAL"]}
        output_path = RAW_RESULTS_DIR / f"task_{task_id:03d}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        return 
    
    # 3. Create the 50x50 query grid for the unit square [0,1]^2
    xx, yy = np.meshgrid(
        np.linspace(-0.05, 1.05, GRID_RESOLUTION), 
        np.linspace(-0.05, 1.05, GRID_RESOLUTION)
    )
    grid_points_cont = pd.DataFrame(
        np.c_[xx.ravel(), yy.ravel()], 
        columns=['X1_cont', 'X2_cont']
    )

    # Binarize grid features using the 0.33/0.66 thresholds
    grid_points_bin = pd.DataFrame(index=grid_points_cont.index)
    grid_points_bin['X1_bin_0.33'] = (grid_points_cont['X1_cont'] > 0.33).astype(int)
    grid_points_bin['X1_bin_0.66'] = (grid_points_cont['X1_cont'] > 0.66).astype(int)
    grid_points_bin['X2_bin_0.33'] = (grid_points_cont['X2_cont'] > 0.33).astype(int)
    grid_points_bin['X2_bin_0.66'] = (grid_points_cont['X2_cont'] > 0.66).astype(int)
    
    # 4. Train committees 
    print(f"  Training committees...")
    rf_committee = get_rf_committee(X_train_bin, y_train, N_COMMITTEE, MODEL_SEED)
    rashomon_committee = get_rashomon_committee(
        X_train_bin, y_train, N_COMMITTEE, MODEL_SEED,
        reg=REGULARIZATION,
        thresh=THRESHOLD
    )

    # 5. Calculate Vote Entropies across the grid
    print(f"  Calculating grid entropies...")
    
    # Baseline: QBC with Random Forest
    rf_entropies = get_entropies_for_grid(
        rf_committee, df_train, 
        use_unique_trees=False, 
        grid_points_bin=grid_points_bin
    )
    
    # UNREAL
    unreal_entropies = get_entropies_for_grid(
        rashomon_committee, df_train, 
        use_unique_trees=True, 
        grid_points_bin=grid_points_bin
    )
    
    # DUREAL
    dureal_entropies = get_entropies_for_grid(
        rashomon_committee, df_train, 
        use_unique_trees=False, 
        grid_points_bin=grid_points_bin
    )
    
    # 6. Reshape results for heatmap plotting
    results = {
        "QBC-RF": rf_entropies.values.reshape(GRID_RESOLUTION, GRID_RESOLUTION),
        "UNREAL": unreal_entropies.values.reshape(GRID_RESOLUTION, GRID_RESOLUTION),
        "DUREAL": dureal_entropies.values.reshape(GRID_RESOLUTION, GRID_RESOLUTION)
    }
    
    # 7. SAVE 
    output_path = RAW_RESULTS_DIR / f"task_{task_id:03d}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
        
    print(f"  Task {task_id} complete. Heatmap saved to results.")

### MAIN ENTRY POINT ###
if __name__ == "__main__":
    if len(sys.argv) == 2:
        run_task(int(sys.argv[1]))
    else:
        print("Usage: python continuous_task.py <task_id>")