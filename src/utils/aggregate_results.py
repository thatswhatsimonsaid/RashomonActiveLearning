### LIBRARIES ###
import pickle
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import re
import argparse

### AGGREGATE RESULTS FUNCTION ###
def aggregate_results(results_base_dir: Path, target_dataset: str = None):
    """
    Aggregates raw .pkl simulation results into compiled .csv files for analysis.

    Args:
        results_base_dir: The top-level directory containing all result subdirectories.
        target_dataset: Optional. If provided, only aggregate results for this specific dataset.
    """
    # print("--- Starting Aggregation of Raw Results ---")

    ### SPECIFY DATASETS TO PROCESS ###
    if target_dataset:
        dataset_paths = [results_base_dir / target_dataset]
        if not dataset_paths[0].exists():
            print(f"Error: Dataset directory not found at '{dataset_paths[0]}'")
            return
    else:
        dataset_paths = [p for p in results_base_dir.iterdir() if p.is_dir()]

    ### LOOP THROUGH DATASETS ###
    for dataset_path in dataset_paths:
        # print(f"\nProcessing dataset: {dataset_path.name}...")

        ## 1. Create the 'aggregated' output directory ##
        aggregated_dir = dataset_path / "aggregated"
        aggregated_dir.mkdir(exist_ok=True)

        ## 2. Find all method subdirectories ##
        method_paths = sorted(
            [p for p in dataset_path.iterdir() if p.is_dir() and p.name.startswith('M')],
            key=lambda p: int(re.search(r'\d+', p.name).group())
        )

        if not method_paths:
            print(f"  > No method results found for {dataset_path.name}. Skipping.")
            continue

        ## 3. Loop through each method folder ##
        for method_path in tqdm(method_paths, desc=f"Aggregating methods for {dataset_path.name}"):
            method_name = method_path.name
            
            pkl_files = list(method_path.glob("*.pkl"))
            if not pkl_files:
                continue

            ## 4. Peek at the first result file ##
            with open(pkl_files[0], 'rb') as f:
                sample_result = pickle.load(f)
            
            result_keys = [key for key in vars(sample_result) if not key.startswith('_')]
            
            ## 5. Initialize storage ##
            aggregated_data = {key: [] for key in result_keys}

            ## 6. Loop through all simulation files ##
            for pkl_file in sorted(pkl_files):
                with open(pkl_file, 'rb') as f:
                    result = pickle.load(f)
                
                for key in result_keys:
                    aggregated_data[key].append(getattr(result, key))

            ## 7. Create and save a DataFrame for each key ##
            for key, data_list in aggregated_data.items():
                if not data_list: continue

                if isinstance(data_list[0], list):
                    df = pd.DataFrame(data_list)
                    df.columns = [f"Iter_{i}" for i in range(df.shape[1])]
                else:
                    df = pd.DataFrame({key: data_list})

                df.index.name = "Simulation"
                
                output_filename = f"{method_name}_{key}.csv"
                output_path = aggregated_dir / output_filename
                df.to_csv(output_path)
        
        # print(f"  > Aggregation for {dataset_path.name} complete. Results are in '{aggregated_dir.relative_to(Path.cwd())}/'")

    # print("\n--- All Datasets Aggregated Successfully ---")

### MAIN ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate simulation results for one or all datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional: The name of a single dataset to aggregate."
    )
    args = parser.parse_args()

    ## DIRECTORY ##
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    RESULTS_DIR = PROJECT_ROOT / "results"
    
    ## RUN AGGREGATION ##
    aggregate_results(results_base_dir=RESULTS_DIR, target_dataset=args.dataset)
