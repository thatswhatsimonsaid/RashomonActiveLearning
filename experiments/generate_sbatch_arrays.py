### LIBRARIES ###
import os
import stat
from pathlib import Path
from master_config import EXPERIMENT_CONFIGS, N_REPLICATIONS, SLURM_CONFIG

### PATHS ###
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOG_DIR = SCRIPT_DIR / "job_logs"
SBATCH_DIR = SCRIPT_DIR / "job_scripts"
DATA_DIR = PROJECT_ROOT / "src" / "data" / "processed"

### FUNCTIONS ###
def create_sbatch_file(dataset_name: str, config: dict, method_number: int):
    """Generates a dedicated sbatch script for a single experiment configuration."""
    
    ## CONFIGURATION ##
    model_name = config["model"]
    selector_name = config["selector"]
    params = config["params"]    
    job_name = f"{dataset_name}_M{method_number}"
    params_str = " ".join([f"{k}={v}" for k, v in params.items()])
    python_executable = PROJECT_ROOT / ".RAL/bin/python"
    
    ## COMMANDS ##
    python_command = f"""
{python_executable} -m src.utils.run_experiment \\
    --dataset {dataset_name} \\
    --model {model_name} \\
    --selector {selector_name} \\
    --seed $SLURM_ARRAY_TASK_ID \\
    --method_number {method_number} \\
    {params_str}
"""

    ## SBATCH SCRIPT CONTENT ##
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --array=0-{N_REPLICATIONS - 1}
#SBATCH --output={LOG_DIR}/{dataset_name}/out/M{method_number}_S%a.out
#SBATCH --error={LOG_DIR}/{dataset_name}/error/M{method_number}_S%a.err
#SBATCH --time={SLURM_CONFIG['time']}
#SBATCH --mem-per-cpu={SLURM_CONFIG['mem_per_cpu']}
#SBATCH --mail-type={SLURM_CONFIG['mail_type']}
#SBATCH --mail-user={SLURM_CONFIG['mail_user']}

cd {PROJECT_ROOT}

echo "Running {job_name} | Seed (Task ID): $SLURM_ARRAY_TASK_ID"

{python_command}
"""
    
    ## WRITE FILE ##
    dataset_sbatch_dir = SBATCH_DIR / dataset_name
    sbatch_path = dataset_sbatch_dir / f"submit_{job_name}.sbatch"
    
    with open(sbatch_path, 'w') as f:
        f.write(sbatch_content)
    
    os.chmod(sbatch_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    print(f"  -> Generated script for {job_name}")

### MAIN SCRIPT ###
if __name__ == "__main__":
    datasets = sorted([f.stem for f in DATA_DIR.glob("*.pkl")])
    if not datasets:
        print(f"Warning: No .pkl files found in {DATA_DIR}. No sbatch files will be generated.")

    print("--- Generating sbatch job array files ---")

    ## CREATE DIRECTORIES ##
    for dataset in datasets:
        dataset_sbatch_dir = SBATCH_DIR / dataset
        dataset_log_dir = LOG_DIR / dataset
        
        dataset_sbatch_dir.mkdir(parents=True, exist_ok=True)
        (dataset_log_dir / "out").mkdir(parents=True, exist_ok=True)
        (dataset_log_dir / "error").mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating scripts for dataset: {dataset}")
        for method_idx, exp_config in enumerate(EXPERIMENT_CONFIGS):
            method_number = method_idx + 1
            create_sbatch_file(dataset, config=exp_config, method_number=method_number)

        # 1. RUN ALL SCRIPT #
        run_all_content = "#!/bin/bash\nfor file in ./*.sbatch; do sbatch \"$file\"; done"
        run_all_path = dataset_sbatch_dir / "1_run_all.sh"
        with open(run_all_path, 'w') as f:
            f.write(run_all_content)
        os.chmod(run_all_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        print(f"  -> Generated '1_run_all.sh' in {dataset_sbatch_dir}")

        # 2. AGGREGATE RESULTS SCRIPT #
        aggregate_results_content = f"""#!/bin/bash
# This script aggregates results ONLY for the {dataset} dataset.
SCRIPT_DIR=$( cd -- "$( dirname -- "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( cd -- "${{SCRIPT_DIR}}/../../../" &> /dev/null && pwd )
DATASET_NAME=$(basename "$SCRIPT_DIR")
PYTHON_EXEC="${{PROJECT_ROOT}}/.RAL/bin/python"

echo "Aggregating results for dataset: ${{DATASET_NAME}}"

# Change to the project root to ensure python -m works
cd "${{PROJECT_ROOT}}"

# Run the aggregation script with the correct python, targeting only this dataset
"${{PYTHON_EXEC}}" -m src.utils.aggregate_results --dataset "${{DATASET_NAME}}"

echo "Aggregation complete for ${{DATASET_NAME}}."
"""
        aggregate_results_path = dataset_sbatch_dir / "2_aggregate_results.sh"
        with open(aggregate_results_path, 'w') as f:
            f.write(aggregate_results_content)
        os.chmod(aggregate_results_path, stat.S_IRWXU)
        print(f"  -> Generated '2_aggregate_results.sh' in {dataset_sbatch_dir}")
        
        # 3. PLOT RESULTS SCRIPT #
        plot_results_content = f"""#!/bin/bash
# This is a placeholder for your plotting script.
# For now, it just provides instructions.
echo "Plotting for dataset: {dataset}"
echo "To analyze and plot these results, you can start a Jupyter Notebook."
echo "From the project root, run:"
echo "  jupyter notebook"
echo "Then open a notebook and load the aggregated CSV files from:"
echo "  results/{dataset}/aggregated/"
"""
        plot_results_path = dataset_sbatch_dir / "3_plot_results.sh"
        with open(plot_results_path, 'w') as f:
            f.write(plot_results_content)
        os.chmod(plot_results_path, stat.S_IRWXU)
        print(f"  -> Generated '3_plot_results.sh' in {dataset_sbatch_dir}")

        # 4. CLEANUP LOGS SCRIPT #
        cleanup_logs_content = f"""#!/bin/bash
# This script cleans up logs and sbatch files for the {dataset} dataset.
SCRIPT_DIR=$( cd -- "$( dirname -- "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( cd -- "${{SCRIPT_DIR}}/../../../" &> /dev/null && pwd )
DATASET_NAME=$(basename "$SCRIPT_DIR")
DATASET_LOG_DIR="${{PROJECT_ROOT}}/experiments/job_logs/${{DATASET_NAME}}"
echo "Cleaning up logs and sbatch files for dataset: ${{DATASET_NAME}}"
echo "  -> Deleting .sbatch files in current directory..."
find . -maxdepth 1 -type f -name "*.sbatch" -delete
echo "  -> Deleting log directory: ${{DATASET_LOG_DIR}}"
if [ -d "$DATASET_LOG_DIR" ]; then rm -rf "$DATASET_LOG_DIR"; echo "     Log directory deleted."; else echo "     Log directory not found."; fi
echo "  -> Deleting helper scripts..."
rm -- 1_run_all.sh 2_aggregate_results.sh 3_plot_results.sh 5_cleanup_results.sh "$0"
echo "Log cleanup complete."
"""
        cleanup_logs_path = dataset_sbatch_dir / "4_cleanup_logs.sh"
        with open(cleanup_logs_path, 'w') as f:
            f.write(cleanup_logs_content)
        os.chmod(cleanup_logs_path, stat.S_IRWXU)
        print(f"  -> Generated '4_cleanup_logs.sh' in {dataset_sbatch_dir}")

        # 5. CLEANUP RESULTS SCRIPT #
        cleanup_results_content = f"""#!/bin/bash
# This script cleans up ONLY the raw result files for the {dataset} dataset, leaving the 'aggregated' folder intact.
SCRIPT_DIR=$( cd -- "$( dirname -- "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( cd -- "${{SCRIPT_DIR}}/../../../" &> /dev/null && pwd )
DATASET_NAME=$(basename "$SCRIPT_DIR")
DATASET_RESULTS_DIR="${{PROJECT_ROOT}}/results/${{DATASET_NAME}}"

echo "Cleaning up raw result files for dataset: ${{DATASET_NAME}} (leaving 'aggregated' folder)"
echo "  -> Targeting directory: ${{DATASET_RESULTS_DIR}}"

# --- Safety Checks ---
if [ -z "$DATASET_RESULTS_DIR" ] || [ "$DATASET_RESULTS_DIR" == "$PROJECT_ROOT" ] || [ "$DATASET_RESULTS_DIR" == "/" ]; then
    echo "     [SAFETY] Path is empty or points to a critical directory. Aborting."
    exit 1
fi
if [[ "$DATASET_RESULTS_DIR" != *"/results/"* ]]; then
    echo "     [SAFETY] Path does not appear to be a valid results directory. Aborting."
    exit 1
fi

# --- Deletion Logic ---
if [ -d "$DATASET_RESULTS_DIR" ]; then
    echo "     Directory found. Starting multi-step cleanup..."
    
    # Step 1: Delete all .pkl files inside the method subdirectories (M1, M2, etc.)
    echo "       - Deleting raw .pkl result files..."
    find "$DATASET_RESULTS_DIR" -type f -name "*.pkl" -delete
    
    # Step 2: Delete the now-empty method subdirectories (M1, M2, etc.)
    # This will not affect the 'aggregated' folder as it is not empty.
    echo "       - Deleting empty M* subdirectories..."
    find "$DATASET_RESULTS_DIR" -mindepth 1 -type d -empty -delete

else
    echo "     Result directory not found. Nothing to do."
fi
echo "Result cleanup complete."
"""
        cleanup_results_path = dataset_sbatch_dir / "5_cleanup_results.sh"
        with open(cleanup_results_path, 'w') as f:
            f.write(cleanup_results_content)
        os.chmod(cleanup_results_path, stat.S_IRWXU)
        print(f"  -> Generated '5_cleanup_results.sh' in {dataset_sbatch_dir}")
        
    print("\n--- Done ---")

