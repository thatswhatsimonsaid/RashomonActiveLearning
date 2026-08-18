### Summary ###
"""
Aggregates Active Learning simulation results across multiple methods and seeds.
'Safe' version: Only aggregates seeds that completed successfully for ALL methods.
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
from collections import defaultdict
import re

### Robust Value Getter ###
def get_metric_value(result_obj, metric_key):
    """Retrieves a metric whether it is a Dictionary or a Class Object."""
    if isinstance(result_obj, dict):
        return result_obj.get(metric_key)
    return getattr(result_obj, metric_key, None)

def extract_seed_id(filename: str) -> str:
    """Extracts the seed identifier (e.g., 'S0', 'S99') from a filename."""
    match = re.search(r'_(S\d+)\.pkl$', filename)
    return match.group(1) if match else None

### Aggregate results functions ###
def aggregate_results(dataset_subdir: str, project_root: Path, study_dir: str):
    """
    Reads individual seed .pkl files, identifies common seeds across all methods,
    and aggregates them into summary files.
    """

    ALLOWED_METHODS = {"M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10"}    

    ## 1. Define paths ##
    results_dir = project_root / "results" / study_dir / dataset_subdir
    output_dir = results_dir / "aggregated"
    
    print(f"--- Safe Aggregation for {dataset_subdir} ---")
    if not results_dir.exists():
        print(f"  > [ERROR] Directory not found: {results_dir}")
        return

    ## 2. Identify Methods and Common Seeds ##
    method_dirs = sorted([
        d for d in results_dir.iterdir() 
        if d.is_dir() and d.name in ALLOWED_METHODS
    ])
    
    if not method_dirs:
        print(f"  > No valid method directories found in {results_dir}")
        return

    # Map method name to set of seed IDs found
    method_seeds = {}
    method_file_map = {} # method -> seed_id -> filepath

    for md in method_dirs:
        m_name = md.name
        files = list(md.glob("*.pkl"))
        seeds = {}
        for f in files:
            sid = extract_seed_id(f.name)
            if sid:
                seeds[sid] = f
        
        method_seeds[m_name] = set(seeds.keys())
        method_file_map[m_name] = seeds

    # Calculate Intersection (Seeds present in EVERY found method)
    if not method_seeds:
        print("  > No seeds found.")
        return

    common_seeds = set.intersection(*method_seeds.values())
    common_seeds = sorted(list(common_seeds), key=lambda x: int(x[1:]))

    print(f"  > Detected {len(method_dirs)} methods.")
    print(f"  > Consistency Check:")
    for m in method_seeds:
        count = len(method_seeds[m])
        status = "OK" if count == len(common_seeds) else f"DIFF ({count})"
        print(f"    - {m:<4}: {count:<3} seeds found | Status: {status}")
    
    print(f"\n  > Found {len(common_seeds)} common seeds across all methods.")
    if not common_seeds:
        print("  > [ABORT] No common seeds found. Check for failed simulations.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    ## 3. Define Metrics to Aggregate ##
    METRICS = [
        "accuracy_history",
        "f1_history",
        "mse_history",
        "rmse_history",
        "r2_history",
        "oracle_agreement_history",
        "rashomon_size_history",
        "committee_size_history",        
        "tree_edit_distance_history",
        "runtime_history"     
    ]

    ## 4. Run Aggregation ##
    for method_name in [md.name for md in method_dirs]:
        print(f"  > Aggregating {method_name} (using {len(common_seeds)} common seeds)...")

        aggregated_data = defaultdict(list)
        valid_seeds_count = 0

        for sid in common_seeds:
            pkl_file = method_file_map[method_name][sid]
            try:
                with open(pkl_file, "rb") as f:
                    result = pickle.load(f)
                
                # Use accuracy_history (classification) or mse_history (regression) as validation
                acc = get_metric_value(result, "accuracy_history")
                mse = get_metric_value(result, "mse_history")
                if acc is None and mse is None:
                    continue

                for metric in METRICS:
                    val = get_metric_value(result, metric)
                    if val is not None:
                        if isinstance(val, (list, np.ndarray)):
                            # Convert everything to list for consistency in first pass
                            aggregated_data[metric].append(np.array(val).tolist())
                        elif isinstance(val, (int, float)):
                            aggregated_data[metric].append(val)
                            
                valid_seeds_count += 1
            except Exception as e:
                print(f"    - Error loading {pkl_file.name}: {e}")

        final_dict = {}
        for metric, data_list in aggregated_data.items():
            if not data_list: continue
            
            # Aggregate Histories (List of Lists) -> (Seeds, Iterations)
            if isinstance(data_list[0], list):
                # Ensure all histories have same length by truncating to minimum found length
                min_len = min(len(x) for x in data_list)
                truncated_data = [x[:min_len] for x in data_list]
                arr = np.array(truncated_data)
                
                final_dict[metric] = arr
                final_dict[f"{metric}_mean"] = np.nanmean(arr, axis=0)
                final_dict[f"{metric}_std"] = np.nanstd(arr, axis=0)
            
            # Aggregate Scalars
            else:
                arr = np.array(data_list)
                final_dict[metric] = arr
                final_dict[f"{metric}_mean"] = np.nanmean(arr)
                final_dict[f"{metric}_std"] = np.nanstd(arr)

        if valid_seeds_count > 0:
            final_dict["n_seeds"] = valid_seeds_count
            final_dict["common_seeds"] = common_seeds
            
            output_file = output_dir / f"{method_name}_results.pkl"
            with open(output_file, "wb") as f:
                pickle.dump(final_dict, f)
        else:
            print(f"    - No valid data found for {method_name}.")

    print(f"\n✨ Aggregation Complete. Summaries saved to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--study_dir", type=str, required=True) 
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    aggregate_results(args.dataset, PROJECT_ROOT, args.study_dir)