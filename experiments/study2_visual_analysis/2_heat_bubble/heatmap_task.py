### LIBRARIES ###
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import itertools 
import pickle 

### PATH SETUP ###
SCRIPT_DIR = Path(__file__).resolve().parent
VISUAL_ANALYSIS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = VISUAL_ANALYSIS_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(VISUAL_ANALYSIS_DIR))

from utils import (
    create_data_pool,
    get_weighted_entropies, # Updated production bridge
    BINARIZED_FEATURES
)
from src.utils.models import BMARandomForestWrapper, PySORTDWrapper, RandomForestWrapper

# ==============================================================================
# --- STUDY 2 PARAMETERS
# ==============================================================================
N_POINTS = 10          
N_INITIAL_POINTS = 3    
N_COMMITTEE = 5
POOL_SEED = 0
MODEL_SEED = 0
REGULARIZATION = 0.01
THRESHOLD = 0.5
BETA = 50.0  # Aligning with Appendix/M9 logic

### CONFIGURATION ###
BASE_OUTPUT_DIR = PROJECT_ROOT / "results" / "study2_visual_analysis" / "2_heat_bubble"
RAW_RESULTS_DIR = BASE_OUTPUT_DIR / "raw"
RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True) 
warnings.filterwarnings("ignore", category=UserWarning) 

# ==============================================================================
# --- SIMULATION LOGIC ---
# ==============================================================================

def get_simulation_combination(task_id: int):
    all_point_indices = list(range(N_POINTS))
    all_combinations = list(itertools.combinations(all_point_indices, N_INITIAL_POINTS))
    if not (0 <= task_id < len(all_combinations)):
        raise IndexError(f"Task ID {task_id} out of bounds.")
    
    train_indices = list(all_combinations[task_id])
    return train_indices

def run_task(task_id: int):
    print(f"--- Running Study 2 Bubble Task ID {task_id} ---")
    
    # 1. Create the fixed 10-point data pool
    df_pool = create_data_pool(N_POINTS, POOL_SEED)
    
    # 2. Get the specific combination for this task
    train_indices = get_simulation_combination(task_id)
    df_train = df_pool.loc[train_indices]
    
    # In bubble tasks, we calculate entropy for ALL points in the pool 
    # to see how uncertainty shifts globally.
    df_candidate = df_pool.copy()
    
    X_train_bin = df_train[BINARIZED_FEATURES]
    y_train = df_train["label"]

    if len(y_train.unique()) < 2:
        print(f"  [Task {task_id} SKIPPED] Training set has only one class.")
        final_report = pd.DataFrame(0.0, index=df_pool.index, columns=["QBC-RF", "BMA-RF", "UNREAL"])
        output_path = RAW_RESULTS_DIR / f"task_{task_id:03d}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(final_report, f)
        return 
    
    # 3. Instantiate and Fit Production Wrappers
    print(f"  Training production wrappers...")
    rf_u_model = RandomForestWrapper(n_estimators=N_COMMITTEE, max_depth=3)
    rf_b_model = BMARandomForestWrapper(n_estimators=N_COMMITTEE, max_depth=3)
    unreal_model = PySORTDWrapper(
        max_depth=3, regularization=REGULARIZATION, 
        rashomon_multiplier=1.1, max_num_trees=N_COMMITTEE
    )

    for m in [rf_u_model, rf_b_model, unreal_model]:
        m.fit(X_train_bin, y_train)

    # 4. Calculate Weighted Entropies
    print(f"  Calculating weighted entropies (Beta={BETA})...")
    results = {}
    
    # M4: QBC-RF (Uniform Weights)
    results["QBC-RF"] = get_weighted_entropies(rf_u_model, df_train, df_candidate, beta=0.0) 
    
    # M9: BMA-RF (Bayesian Weights)
    results["BMA-RF"] = get_weighted_entropies(rf_b_model, df_train, df_candidate, beta=BETA)
    
    # M8: UNREAL (Rashomon + Bayesian Weights)
    results["UNREAL"] = get_weighted_entropies(unreal_model, df_train, df_candidate, beta=BETA)
    
    # 5. Format and Save
    final_report = pd.DataFrame(index=df_pool.index)
    for name, series in results.items():
        final_report[name] = series 
    
    final_report = final_report.fillna(0.0)
    
    output_path = RAW_RESULTS_DIR / f"task_{task_id:03d}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(final_report, f)
    
    print(f"  Task {task_id} complete.")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        run_task(int(sys.argv[1]))
    else:
        print("Usage: python heatmap_task.py <task_id>")