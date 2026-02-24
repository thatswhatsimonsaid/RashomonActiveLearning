### LIBRARIES ###
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import pickle

### PATH SETUP ###
SCRIPT_DIR = Path(__file__).resolve().parent
STUDY_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = STUDY_ROOT.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(STUDY_ROOT))

from utils import (
    create_data_pool,
    get_weighted_entropies,
    BINARIZED_FEATURES
)
from src.utils.models import BMARandomForestWrapper, PySORTDWrapper, RandomForestWrapper

### PARAMETERS ####
N_POOL_POINTS = 500      
N_INITIAL_POINTS = 3  
N_COMMITTEE = 10     
POOL_SEED = 0     
MODEL_SEED = 0 
GRID_RESOLUTION = 50
REGULARIZATION = 0.001
THRESHOLD = 500      
BETA = 50.0         

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
    Benchmarks QBC-RF (Uniform) vs BMA-RF (Weighted) vs UNREAL (Rashomon).
    """
    print(f"--- Running Continuous Heatmap Task ID {task_id} ---")
    
    # 1. Create a large data pool
    df_pool = create_data_pool(n_points=N_POOL_POINTS, pool_seed=POOL_SEED)
    
    # 2. Get the initial training set
    np.random.seed(task_id)
    train_indices = np.random.choice(df_pool.index, size=N_INITIAL_POINTS, replace=False)
    df_train = df_pool.loc[train_indices]
    X_train_bin = df_train[BINARIZED_FEATURES]
    y_train = df_train["label"]

    if len(y_train.unique()) < 2:
        print(f"  [Task {task_id} SKIPPED] Single-class training set.")        
        results = {k: np.zeros((GRID_RESOLUTION, GRID_RESOLUTION)) for k in ["QBC-RF", "BMA-RF", "UNREAL"]}
        output_path = RAW_RESULTS_DIR / f"task_{task_id:03d}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        return 
    
   # 3. Create the grid for the unit square [0,1]^2
    xx, yy = np.meshgrid(
        np.linspace(-0.05, 1.05, GRID_RESOLUTION), 
        np.linspace(-0.05, 1.05, GRID_RESOLUTION)
    )
    grid_points_cont = pd.DataFrame(
        np.c_[xx.ravel(), yy.ravel()], 
        columns=['X1_cont', 'X2_cont']
    )

    # Binarize grid features
    grid_points_bin = pd.DataFrame(index=grid_points_cont.index)
    for val in [0.33, 0.66]:
        grid_points_bin[f'X1_bin_{val}'] = (grid_points_cont['X1_cont'] > val).astype(int)
        grid_points_bin[f'X2_bin_{val}'] = (grid_points_cont['X2_cont'] > val).astype(int)
    
    # Ensure grid matches training features exactly
    grid_points_bin = grid_points_bin[BINARIZED_FEATURES]
    
    # 4. Instantiate and Fit Production Wrappers
    print(f"  Training and fitting production models...")
    rf_u_model = RandomForestWrapper(n_estimators=N_COMMITTEE, max_depth=3)
    rf_b_model = BMARandomForestWrapper(n_estimators=N_COMMITTEE, max_depth=3)
    unreal_model = PySORTDWrapper(
        max_depth=3, regularization=REGULARIZATION, 
        rashomon_multiplier=1.1, max_num_trees=N_COMMITTEE
    )

    for m in [rf_u_model, rf_b_model, unreal_model]:
        m.fit(X_train_bin, y_train)

    # 5. Calculate Grid Entropies
    print(f"  Calculating grid entropies (BETA={BETA})...")
    
    # M4: QBC-RF (Uniform)
    rf_u_ent = get_weighted_entropies(rf_u_model, df_train, grid_points_bin, beta=0.0)
    # M9: BMA-RF (Weighted)
    rf_b_ent = get_weighted_entropies(rf_b_model, df_train, grid_points_bin, beta=BETA)
    # M8: UNREAL (Rashomon + Weighted)
    unreal_ent = get_weighted_entropies(unreal_model, df_train, grid_points_bin, beta=BETA)
    
    # 6. Reshape results
    results = {
        "QBC-RF": rf_u_ent.values.reshape(GRID_RESOLUTION, GRID_RESOLUTION),
        "BMA-RF": rf_b_ent.values.reshape(GRID_RESOLUTION, GRID_RESOLUTION),
        "UNREAL": unreal_ent.values.reshape(GRID_RESOLUTION, GRID_RESOLUTION)
    }
    
    # 7. SAVE 
    output_path = RAW_RESULTS_DIR / f"task_{task_id:03d}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
        
    print(f"  Task {task_id} complete. Appendix heatmaps saved.")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        run_task(int(sys.argv[1]))
    else:
        print("Usage: python continuous_task.py <task_id>")