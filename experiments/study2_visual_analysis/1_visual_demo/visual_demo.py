### LIBRARIES ###
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

### PATHS ###
SCRIPT_DIR = Path(__file__).resolve().parent
VISUAL_ANALYSIS_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = VISUAL_ANALYSIS_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(VISUAL_ANALYSIS_DIR))

### IMPORTS ###
from utils import (
    create_data_pool,
    get_rf_committee,
    get_lfr_committee,
    get_qbc_selection,
    plot_committee_partitions,
    BINARIZED_FEATURES,
    N_POINTS,
    N_INITIAL_POINTS,
    N_COMMITTEE,
    POOL_SEED,
    INIT_SET_SEED,
    MODEL_SEED,
    LFR_REGULARIZATION,
    LFR_THRESHOLD,
    USE_ALL_LFR_MODELS
)

### CONFIGURATION & OUTPUT ###
BASE_OUTPUT_DIR = PROJECT_ROOT / "results" / "study2_visual_analysis" / "1_visual_demo"
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True) 

# ==============================================================================
# --- MAIN SIMULATION SCRIPT ---
# ==============================================================================

def main():
    """
    Runs the full visual active learning demonstration.
    """
    print("=== Starting Visual Active Learning Simulation ===")
    
    # 1. Get the 10 data points and the initial labeled points
    df_pool = create_data_pool(N_POINTS, POOL_SEED)
    
    # Select initial points at random
    np.random.seed(INIT_SET_SEED)
    initial_train_indices = np.random.choice(df_pool.index, size=N_INITIAL_POINTS, replace=False)
    initial_candidate_indices = [i for i in df_pool.index if i not in initial_train_indices]
    
    print(f"Initial Labeled Set ({N_INITIAL_POINTS} points): {initial_train_indices}. Seed = {INIT_SET_SEED}")
    print(f"Candidate Set ({10 - N_INITIAL_POINTS} points): {initial_candidate_indices}")
    print("-" * 30)

    # 2. Set up the timelines.
    timelines = {
        "QBC-RF": list(initial_train_indices),
        "UNREAL": list(initial_train_indices),
        # "DUREAL": list(initial_train_indices),
    }
    
    # 3. Loop from Iterations
    n_iterations = len(initial_candidate_indices)
    
    for i in range(n_iterations):
        current_iteration = i + N_INITIAL_POINTS + 1
        print(f"\n--- Starting Iteration {current_iteration} (Adding {current_iteration}th point) ---")

        for selector_name in timelines.keys():
            
            # 1. Get this timeline's current data
            current_train_indices = timelines[selector_name]
            current_candidate_indices = [idx for idx in df_pool.index if idx not in current_train_indices]
            if not current_candidate_indices:
                continue
                
            df_train = df_pool.loc[current_train_indices]
            df_candidate = df_pool.loc[current_candidate_indices]

            X_train_bin = df_train[BINARIZED_FEATURES]
            y_train = df_train["label"]

            # 2. Get the committee for this selector
            committee = []
            use_unique = False
            
            if selector_name == "QBC-RF":
                committee = get_rf_committee(
                    X_train_bin, 
                    y_train, 
                    N_COMMITTEE,
                    MODEL_SEED
                )
                
            elif selector_name == "UNREAL":
                committee = get_lfr_committee(
                    X_train_bin, 
                    y_train, 
                    N_COMMITTEE, 
                    MODEL_SEED,
                    LFR_REGULARIZATION, 
                    LFR_THRESHOLD,
                    USE_ALL_LFR_MODELS
                )
                use_unique = True
                
            elif selector_name == "DUREAL":
                committee = get_lfr_committee(
                    X_train_bin, 
                    y_train, 
                    N_COMMITTEE, 
                    MODEL_SEED,
                    LFR_REGULARIZATION, 
                    LFR_THRESHOLD,
                    USE_ALL_LFR_MODELS
                )
                use_unique = False
            
            if not committee:
                print(f"  > [{selector_name}] No committee models found. Skipping.")
                continue

            # 3. Select the next point
            selected_point_index, candidate_entropies = get_qbc_selection(
                committee, 
                df_train, 
                df_candidate, 
                use_unique, 
                MODEL_SEED
            )
            
            print(f"  > [{selector_name}] Selected Point: {selected_point_index} (V.E. = {candidate_entropies.get(selected_point_index, 0.0):.2f})")
            selector_output_dir = BASE_OUTPUT_DIR / selector_name
            selector_output_dir.mkdir(exist_ok=True)

            # 4. Generate the plot for this state
            plot_committee_partitions(
                committee,
                selector_name,
                current_iteration,
                df_pool,
                current_train_indices,
                selected_point_index,
                candidate_entropies,
                selector_output_dir 
            )
            
            # 5. Add the selected point to this timeline's training set
            timelines[selector_name].append(selected_point_index)
            
    print("\n" + "=" * 30)
    print("=== Simulation Complete ===")
    print(f"All visual results saved to: {BASE_OUTPUT_DIR}")


if __name__ == "__main__":
    main()