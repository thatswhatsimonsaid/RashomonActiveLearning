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
    BINARIZED_FEATURES,
    N_NOISE_FOR_EXPERIMENT
)
from src.utils.models import BMARandomForestWrapper, PySORTDWrapper, RandomForestWrapper

### PARAMETERS ####
N_POOL_POINTS = 500      
N_INITIAL_POINTS = 20  
N_COMMITTEE = 10     
POOL_SEED = 0     
MODEL_SEED = 0 
GRID_RESOLUTION = 50
REGULARIZATION = 0.001
BETA = 0 
RASHOMON_MULTIPLIER  = 1000
MAX_NUM_TREES = 100000
MAX_DEPTH = 5   
RF_MAX_FEATURES = 9

### OUTPUT SETUP ###
BASE_OUTPUT_DIR = PROJECT_ROOT / "results" / "study2_visual_analysis" / "2_continuous_heatmap"
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
    
    # 2. Get a stratified initial training set
    np.random.seed(task_id)
    
    # Separate the pool by class
    df_class_0 = df_pool[df_pool["label"] == 0]
    df_class_1 = df_pool[df_pool["label"] == 1]
    
    # Sample 1 from one class and 2 from the other (total 3)
    n_class_0 = 2 if task_id % 2 == 0 else 1
    n_class_1 = 3 - n_class_0
    
    idx_0 = np.random.choice(df_class_0.index, size=n_class_0, replace=False)
    idx_1 = np.random.choice(df_class_1.index, size=n_class_1, replace=False)
    
    train_indices = np.concatenate([idx_0, idx_1])
    df_train = df_pool.loc[train_indices]
    
    X_train_bin = df_train[BINARIZED_FEATURES]
    y_train = df_train["label"]
    
   # 3. Create the grid for the unit square [0,1]^2
    xx, yy = np.meshgrid(
        np.linspace(-0.05, 1.05, GRID_RESOLUTION), 
        np.linspace(-0.05, 1.05, GRID_RESOLUTION)
    )
    grid_points_cont = pd.DataFrame(
        np.c_[xx.ravel(), yy.ravel()], 
        columns=['X1_cont', 'X2_cont']
    )

    # Binarize signal grid features
    grid_points_bin = pd.DataFrame(index=grid_points_cont.index)
    for val in [0.33, 0.66]:
        grid_points_bin[f'X1_bin_{val}'] = (grid_points_cont['X1_cont'] > val).astype(int)
        grid_points_bin[f'X2_bin_{val}'] = (grid_points_cont['X2_cont'] > val).astype(int)
    
    for i in range(N_NOISE_FOR_EXPERIMENT):
        grid_points_bin[f'N{i}_bin_0.5'] = 0
    grid_points_bin = grid_points_bin[BINARIZED_FEATURES]
    
    # 4. Instantiate and Fit Production Wrappers
    print(f"  Training and fitting production models...")
    rf_u_model = RandomForestWrapper(n_estimators=N_COMMITTEE, 
                                     max_depth=MAX_DEPTH,
                                     max_features = RF_MAX_FEATURES)
    rf_b_model = BMARandomForestWrapper(n_estimators=N_COMMITTEE, max_depth=MAX_DEPTH)
    unreal_model = PySORTDWrapper(
        max_depth=MAX_DEPTH, 
        regularization=REGULARIZATION, 
        rashomon_multiplier=RASHOMON_MULTIPLIER, 
        max_num_trees=MAX_NUM_TREES
    )

    for m in [rf_u_model, rf_b_model, unreal_model]:
        m.fit(X_train_bin, y_train)

    # # --- RASHOMON DEBUGGER ---
    # train_losses = unreal_model.get_ensemble_losses(X_train_bin, y_train)
    # best_err = np.min(train_losses)
    # bound_limit = best_err * RASHOMON_MULTIPLIER
    
    # print(f"\n" + "="*50)
    # print(f"DEBUGGER: UNREAL (Task {task_id})")
    # print(f"  Training Points (n): {len(y_train)}")
    # print(f"  Classes in Train: {y_train.value_counts().to_dict()}")
    # print(f"  Best Cost-Complexity Loss: {best_err:.6f}")
    # print(f"  Rashomon Bound (Multiplier {RASHOMON_MULTIPLIER}x): {bound_limit:.6f}")
    
    # r_size = unreal_model.get_rashomon_size()
    # print(f"  Models found in Rashomon Set: {r_size}")
    
    # if best_err == 0:
    #     print("  WARNING: Best Error is 0. Multiplier has no effect.")
    #     print("  Try adding a small constant to RASHOMON_MULTIPLIER or increasing N_INITIAL_POINTS.")
    # print("="*50 + "\n")
    
    # if best_err == 0:
    #     print("  WARNING: Best Error is 0. Multiplier has no effect.")
    # print("="*50 + "\n")

    # # 5. Calculate Grid Entropies
    # print(f"  Calculating grid entropies (BETA={BETA})...")
    
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