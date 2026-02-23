import argparse
import pickle
from pathlib import Path
import numpy as np
from collections import defaultdict

### Robust Value Getter ###
def get_metric_value(result_obj, metric_key):
    """
    Retrieves a metric from the result object whether it is 
    a Dictionary or a Class Object.
    """
    if isinstance(result_obj, dict):
        return result_obj.get(metric_key)
    return getattr(result_obj, metric_key, None)

### Aggregate results functions ###
def aggregate_results(dataset_subdir: str, project_root: Path, study_dir: str):
    """
    Reads individual seed .pkl files for specific finished methods, 
    aggregates histories, and saves summary .pkl files.
    """
    
    ## 1. Define paths ##
    results_dir = project_root / "results" / study_dir / dataset_subdir
    output_dir = results_dir / "aggregated"
    
    # Define the specific methods that finished/we want to aggregate
    ALLOWED_METHODS = {"M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"}    
    print(f"Aggregating results in: {results_dir}")
    if not results_dir.exists():
        print(f"  > [ERROR] Directory not found: {results_dir}")
        return

    ## 2. Identify Methods ##
    method_dirs = sorted([
        d for d in results_dir.iterdir() 
        if d.is_dir() and d.name in ALLOWED_METHODS
    ])
    
    if not method_dirs:
        print(f"  > No valid method directories found in {results_dir}")
        return

    ## SAFETY CHECK ##
    print(f"  > Performing consistency check on finished methods: {sorted(list(ALLOWED_METHODS))}...")
    run_counts = {}
    for md in method_dirs:
        count = len(list(md.glob("*.pkl")))
        run_counts[md.name] = count
    distinct_counts = set(run_counts.values())

    if len(distinct_counts) > 1:
        print(f"\n [ABORT] Inconsistent simulation counts detected among specified methods.")
        print(f"     Aggregation requires an equal number of runs across selected methods.")
        print(f"     --------------------------------------")
        print(f"     {'METHOD':<10} | {'RUNS FOUND':<10}")
        print(f"     {'-'*23}")
        for method, count in run_counts.items():
            marker = "<<" if count != max(run_counts.values()) else ""
            print(f"     {method:<10} | {count:<10} {marker}")
        print(f"     --------------------------------------")
        return 

    common_count = list(distinct_counts)[0]
    if common_count == 0:
        print(" [STOP] No result files found in any selected method directory.")
        return

    print(f" Check Passed: All {len(method_dirs)} methods have {common_count} runs.")
    output_dir.mkdir(parents=True, exist_ok=True)

    ## 3. Define Metrics to Aggregate ##
    METRICS = [
        "accuracy_history",
        "f1_history",
        "oracle_agreement_history",
        "rashomon_size_history",
        "committee_size_history",        
        "tree_edit_distance_history",
        "elapsed_time",
        "feature_rank_correlation_history",
        "feature_jaccard_history"      
    ]

    ## 4. Run Aggregation ##
    for method_dir in method_dirs:
        method_name = method_dir.name
        print(f"  > Processing {method_name}...")

        seed_files = sorted(list(method_dir.glob("*.pkl")))
        aggregated_data = defaultdict(list)
        valid_seeds_count = 0

        for pkl_file in seed_files:
            try:
                with open(pkl_file, "rb") as f:
                    result = pickle.load(f)
                
                if get_metric_value(result, "accuracy_history") is None:
                    print(f"    - Skipping {pkl_file.name}: Missing accuracy_history.")
                    continue

                for metric in METRICS:
                    val = get_metric_value(result, metric)
                    if val is not None:
                        if isinstance(val, list):
                            aggregated_data[metric].append(val)
                        elif isinstance(val, np.ndarray):
                            aggregated_data[metric].append(val.tolist())
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
                min_len = min(len(x) for x in data_list)
                truncated_data = [x[:min_len] for x in data_list]
                arr = np.array(truncated_data)
                
                final_dict[metric] = arr
                final_dict[f"{metric}_mean_trace"] = np.mean(arr, axis=0)
                final_dict[f"{metric}_std_trace"] = np.std(arr, axis=0)
            
            # Aggregate Scalars (Time/Global Metrics)
            else:
                arr = np.array(data_list)
                final_dict[metric] = arr
                final_dict[f"{metric}_mean"] = np.mean(arr)
                final_dict[f"{metric}_std"] = np.std(arr)
                final_dict[f"{metric}_sem_trace"] = np.std(arr, axis=0) / np.sqrt(valid_seeds_count)

        if valid_seeds_count > 0:
            output_file = output_dir / f"{method_name}_results.pkl"
            with open(output_file, "wb") as f:
                pickle.dump(final_dict, f)
        else:
            print(f"    - No valid data found for {method_name}.")

    print(f"  ✨ Aggregation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--study_dir", type=str, required=True) 
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    aggregate_results(args.dataset, PROJECT_ROOT, args.study_dir)