### Summary ###
"""
Utility script to collect plots from deep dataset directories into a 
centralized 'PLOTS' folder, organized by metric type.
Supports recursive searching to handle varying directory structures.
"""

import shutil
from pathlib import Path
import argparse

def collect_plots(study_name: str):
    # Determine Project Root (assumes script is in src/utils/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    
    # Define paths
    results_dir = project_root / "results" / study_name
    plots_dir = results_dir / "COLLECTED_PLOTS"

    if not results_dir.exists():
        print(f"[ERROR] Study directory not found: {results_dir}")
        print(f"Looked at: {results_dir}")
        return

    print(f"Scanning study directory: {results_dir}")
    print(f"Target directory: {plots_dir}")

    # Counter for feedback
    count = 0

    # Iterate through each dataset folder in the study
    # We skip 'aggregated' and 'COLLECTED_PLOTS'
    for dataset_dir in sorted(results_dir.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name in ["aggregated", "COLLECTED_PLOTS"]:
            continue
        
        print(f"  > Checking {dataset_dir.name}...")
        
        # Robust search: Find all images recursively within this dataset's folder
        found_in_dataset = 0
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            for img in dataset_dir.rglob(ext):
                # Skip anything already inside a COLLECTED_PLOTS folder if it exists
                if "COLLECTED_PLOTS" in img.parts:
                    continue
                    
                # Metric name is the filename (e.g., '1_accuracy_stability')
                plot_type = img.stem 
                
                # Create a dedicated folder for this metric
                dest_dir = plots_dir / plot_type
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy and rename to the dataset name (preserving extension)
                dest_file = dest_dir / f"{dataset_dir.name}{img.suffix}"
                shutil.copy2(img, dest_file)
                
                found_in_dataset += 1
                count += 1
        
        if found_in_dataset > 0:
            print(f"    - Found {found_in_dataset} images.")
        
    if count > 0:
        print(f"\n✨ Done! Successfully collected {count} plots.")
        print(f"Organized by metric in: {plots_dir}")
    else:
        print(f"\n⚠️ No plots found. Please verify that plots have been generated in the dataset subdirectories.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=str, default="ABS", help="Study path relative to results/ (e.g., 'study1_active_learning/ABS')")
    args = parser.parse_args()
    
    collect_plots(args.study)