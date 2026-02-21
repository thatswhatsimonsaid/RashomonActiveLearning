# experiments/study2_visual_analysis/4_initialization_sensitivity/initialization_task.py

import sys
import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.spatial.distance import cdist

# --- PATH SETUP ---
SCRIPT_DIR = Path(__file__).resolve().parent
VISUAL_ANALYSIS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = VISUAL_ANALYSIS_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(VISUAL_ANALYSIS_DIR))

from utils import (
    create_data_pool, get_rf_committee, get_lfr_committee,
    get_entropies_for_grid, distance_to_boundary,
    BINARIZED_FEATURES, N_COMMITTEE, MODEL_SEED, POOL_SEED,
    LFR_REGULARIZATION, LFR_THRESHOLD, GRID_RESOLUTION, USE_ALL_LFR_MODELS
)

# --- CONFIGURATION ---
BASE_OUTPUT_DIR = PROJECT_ROOT / "results" / "study2_visual_analysis" / "4_initialization_sensitivity" / "raw"
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
N_POOL_POINTS = 500
N_INITIAL = 5 

# --- SCENARIO C CONFIGURATION ---
MARGIN_LIST = [0.4, 0.2, 0.05] 

# ==============================================================================
# --- INITIALIZATION LOGIC ---
# ==============================================================================

def check_data_validity(X_bin, y):
    """Checks if the dataset contains conflicting samples."""
    df_check = X_bin.copy()
    df_check['label'] = y
    feature_cols = X_bin.columns.tolist()
    conflicts = df_check.groupby(feature_cols)['label'].nunique()
    if (conflicts > 1).any():
        return False, "Identical features with conflicting labels detected."
    return True, "OK"

def get_scenario_c_indices(df_pool, margin):
    """Fixed geometry for Closing Vise."""
    lower = 0.5 - margin
    upper = 0.5 + margin
    p1 = [lower, lower]                 
    p2 = [lower, lower - 0.05]          
    p3 = [lower - 0.05, lower]          
    p4 = [upper, upper]                 
    p5 = [upper + 0.05, upper + 0.05]   
    target_coords = np.array([p1, p2, p3, p4, p5])
    target_coords = np.clip(target_coords, 0.01, 0.99)
    
    pool_coords = df_pool[['X1_cont', 'X2_cont']].values
    selected_indices = []
    for target in target_coords:
        dists = cdist([target], pool_coords)[0]
        sorted_args = np.argsort(dists)
        for idx in sorted_args:
            real_idx = df_pool.index[idx]
            if real_idx not in selected_indices:
                selected_indices.append(real_idx)
                break
    return np.array(selected_indices) 

def get_stratified_sample(df_pool, candidates_idx, n_total, rng_seed):
    """
    Helper to ensure at least 1 point from Class 0 and 1 from Class 1,
    then sample the rest randomly from the candidates.
    """
    np.random.seed(rng_seed)
    
    # Subset the pool to just the valid region candidates
    df_cand = df_pool.loc[candidates_idx]
    
    idx_0 = df_cand[df_cand['label'] == 0].index
    idx_1 = df_cand[df_cand['label'] == 1].index
    
    # Verify we can actually do stratified sampling
    if len(idx_0) == 0 or len(idx_1) == 0:
        # Fallback: If geometry forces only one class, we can't stratify.
        # Just return random (and it will likely fail the later check).
        return np.random.choice(candidates_idx, size=n_total, replace=False)
    
    # 1. Pick one mandatory point from each class
    p0 = np.random.choice(idx_0, 1, replace=False)
    p1 = np.random.choice(idx_1, 1, replace=False)
    
    # 2. Pick the remaining points from EVERYONE else
    current_selection = np.concatenate([p0, p1])
    remaining_candidates = [i for i in candidates_idx if i not in current_selection]
    
    if len(remaining_candidates) < (n_total - 2):
        raise ValueError("Not enough points in region to stratify.")
        
    p_rest = np.random.choice(remaining_candidates, size=n_total - 2, replace=False)
    
    return np.concatenate([current_selection, p_rest])

def get_initial_indices(task_id, scenario_type, df_pool):
    """
    Selects indices based on the Scenario with STRATIFIED sampling for A and B.
    """
    if scenario_type == 'A_Random_Corners':
        condition = (
            ((df_pool['X1_cont'] < 0.2) | (df_pool['X1_cont'] > 0.8)) & 
            ((df_pool['X2_cont'] < 0.2) | (df_pool['X2_cont'] > 0.8))
        )
        candidates = df_pool[condition].index
        # Use Stratified helper
        return get_stratified_sample(df_pool, candidates, N_INITIAL, task_id)

    elif scenario_type == 'B_Random_Boundary':
        dists = df_pool[['X1_cont', 'X2_cont']].apply(
            lambda x: abs(distance_to_boundary(x)), axis=1
        )
        candidates = df_pool[dists < 0.1].index
        # Use Stratified helper
        return get_stratified_sample(df_pool, candidates, N_INITIAL, task_id)

    elif scenario_type.startswith('C_Margin_'):
        margin_val = float(scenario_type.split('_')[-1])
        return get_scenario_c_indices(df_pool, margin_val)
    
    else:
        raise ValueError(f"Unknown scenario: {scenario_type}")

# ==============================================================================
# --- MAIN TASK ---
# ==============================================================================

def run_task(array_id):
    scenarios = ['A_Random_Corners', 'B_Random_Boundary'] + \
                [f'C_Margin_{m}' for m in MARGIN_LIST]
    
    runs_per_scenario = 100
    scenario_idx = array_id // runs_per_scenario
    run_idx = array_id % runs_per_scenario
    
    if scenario_idx >= len(scenarios):
        return

    current_scenario = scenarios[scenario_idx]
    print(f"--- Running Task {array_id} ({current_scenario}) ---")

    # 2. Create Pool
    df_pool = create_data_pool(N_POOL_POINTS, POOL_SEED)
    
    # 3. Get Initial Points (Now Stratified!)
    # train_indices = get_initial_indices(array_id, current_scenario, df_pool)
    train_indices = get_initial_indices(run_idx, current_scenario, df_pool)
    
    df_train = df_pool.loc[train_indices]
    X_train_bin = df_train[BINARIZED_FEATURES]
    y_train = df_train["label"]
    
    # Safety Check (Should basically never fail now for A and B)
    if len(y_train.unique()) < 2:
        print("    [WARNING] Only one class in training set. Skipping.")
        return

    # 4. Prepare Grid
    xx, yy = np.meshgrid(np.linspace(0, 1, GRID_RESOLUTION), np.linspace(0, 1, GRID_RESOLUTION))
    grid_points_cont = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=['X1_cont', 'X2_cont'])
    grid_points_bin = pd.DataFrame(index=grid_points_cont.index)
    grid_points_bin['X1_bin_0.33'] = (grid_points_cont['X1_cont'] > 0.33).astype(int)
    grid_points_bin['X1_bin_0.66'] = (grid_points_cont['X1_cont'] > 0.66).astype(int)
    grid_points_bin['X2_bin_0.33'] = (grid_points_cont['X2_cont'] > 0.33).astype(int)
    grid_points_bin['X2_bin_0.66'] = (grid_points_cont['X2_cont'] > 0.66).astype(int)

    # 5. Train & Calculate Entropy
    # QBC-RF
    try:
        rf_committee = get_rf_committee(X_train_bin, y_train, N_COMMITTEE, MODEL_SEED)
        rf_ent = get_entropies_for_grid(rf_committee, df_train, False, grid_points_bin)
        rf_grid = rf_ent.values.reshape(GRID_RESOLUTION, GRID_RESOLUTION)
    except Exception as e:
        print(f"    [ERROR] RF Failed: {e}")
        rf_grid = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION))
    
    # UNREAL
    is_valid, msg = check_data_validity(X_train_bin, y_train)
    if not is_valid:
        print(f"    [SKIP] LFR skipped: {msg}")
        unreal_grid = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION))
    else:
        try:
            lfr_committee = get_lfr_committee(
                X_train_bin, y_train, N_COMMITTEE, MODEL_SEED, 
                LFR_REGULARIZATION, LFR_THRESHOLD, use_all_models=True
            )
            unreal_ent = get_entropies_for_grid(lfr_committee, df_train, True, grid_points_bin)
            unreal_grid = unreal_ent.values.reshape(GRID_RESOLUTION, GRID_RESOLUTION)
        except Exception as e:
            print(f"    [ERROR] LFR Failed: {e}")
            unreal_grid = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION))

    # 6. Save
    results = {
        "scenario": current_scenario,
        "run_idx": run_idx,
        "train_indices": list(train_indices),
        "QBC-RF": rf_grid,
        "UNREAL": unreal_grid
    }
    
    filename = f"init_task_{array_id:04d}_{current_scenario}.pkl"
    with open(BASE_OUTPUT_DIR / filename, 'wb') as f:
        pickle.dump(results, f)
        
    print(f"    Saved to {filename}")

if __name__ == "__main__":
    task_id = int(sys.argv[1])
    run_task(task_id)