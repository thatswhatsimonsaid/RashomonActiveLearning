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
    get_rf_committee,
    get_rashomon_committee,
    get_qbc_selection,
    BINARIZED_FEATURES
)

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

### CONFIGURATION ###
BASE_OUTPUT_DIR = PROJECT_ROOT / "results" / "study2_visual_analysis" / "2_heat_bubble"
RAW_RESULTS_DIR = BASE_OUTPUT_DIR / "raw"
RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True) 
warnings.filterwarnings("ignore", category=UserWarning) 

# ==============================================================================
# --- SIMULATION LOGIC ---
# ==============================================================================

def get_simulation_combination(task_id: int):
    """
    Finds the specific combination of 3 points for this task ID.
    10 choose 3 = 120 possible tasks.
    """
    all_point_indices = list(range(N_POINTS))
    all_combinations = list(itertools.combinations(all_point_indices, N_INITIAL_POINTS))
    if not (0 <= task_id < len(all_combinations)):
        raise IndexError(f"Task ID {task_id} out of bounds. Max task ID is {len(all_combinations)-1}")
    
    train_indices = list(all_combinations[task_id])
    candidate_indices = [i for i in all_point_indices if i not in train_indices]
    return train_indices, candidate_indices

def run_task(task_id: int):
    print(f"--- Running Study 2 Bubble Task ID {task_id} ---")
    
    # 1. Create the fixed 10-point data pool
    df_pool = create_data_pool(N_POINTS, POOL_SEED)
    
    # 2. Get the specific combination for this task
    train_indices, candidate_indices = get_simulation_combination(task_id)
    df_train = df_pool.loc[train_indices]
    df_candidate = df_pool.loc[candidate_indices]
    
    X_train_bin = df_train[BINARIZED_FEATURES]
    y_train = df_train["label"]

    # Cold start check
    if len(y_train.unique()) < 2:
        print(f"  [Task {task_id} SKIPPED] Training set has only one class.")
        final_report = pd.DataFrame(0.0, index=df_pool.index, columns=["QBC-RF", "UNREAL", "DUREAL"])
        output_path = RAW_RESULTS_DIR / f"task_{task_id:03d}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(final_report, f)
        return 
    
    # 3. Train committees
    rf_committee = get_rf_committee(X_train_bin, y_train, N_COMMITTEE, MODEL_SEED)
    rashomon_committee = get_rashomon_committee(
        X_train_bin, y_train, N_COMMITTEE, MODEL_SEED, 
        REGULARIZATION, THRESHOLD
    )

    # 4. Calculate Vote Entropies
    results = {}
    results["QBC-RF"] = get_qbc_selection(rf_committee, df_train, df_candidate, False, MODEL_SEED)[1] 
    results["UNREAL"] = get_qbc_selection(rashomon_committee, df_train, df_candidate, True, MODEL_SEED)[1]
    results["DUREAL"] = get_qbc_selection(rashomon_committee, df_train, df_candidate, False, MODEL_SEED)[1]
    
    # 5. Format and Save
    final_report = pd.DataFrame(index=df_pool.index)
    for name, series in results.items():
        final_report[name] = series 
    
    final_report = final_report.fillna(0.0)
    
    output_path = RAW_RESULTS_DIR / f"task_{task_id:03d}.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(final_report, f)
    
    print(f"  Task {task_id} complete. Results saved.")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        run_task(int(sys.argv[1]))
    else:
        print("Usage: python heatmap_task.py <task_id>")