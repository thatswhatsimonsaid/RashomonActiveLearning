# experiments/generate_sbatch_arrays.py

import os
import stat
from pathlib import Path
from master_config import EXPERIMENT_CONFIGS, N_REPLICATIONS, SLURM_CONFIG

# --- DYNAMIC PATHS (ROBUST VERSION) ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOG_DIR = SCRIPT_DIR / "job_logs"
SBATCH_DIR = SCRIPT_DIR / "job_scripts"
DATA_DIR = PROJECT_ROOT / "src" / "data" / "processed"

def create_sbatch_file(dataset_name: str, config: dict):
    # This function remains the same
    model_name = config["model"]
    selector_name = config["selector"]
    params = config["params"]
    job_name = f"{dataset_name}_{model_name}_{selector_name}"
    if model_name == "TreeFarms" and selector_name == "QBC":
        if params.get("use_unique_trees"):
            job_name += "_UNREAL"
        else:
            job_name += "_DUREAL"
    params_str = " ".join([f"{k}={v}" for k, v in params.items()])
    python_executable = PROJECT_ROOT / "RAL/bin/python"
    python_command = f"""
{python_executable} -m src.utils.run_experiment \\
    --dataset {dataset_name} \\
    --model {model_name} \\
    --selector {selector_name} \\
    --seed $SLURM_ARRAY_TASK_ID \\
    {params_str}
"""
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --array=0-{N_REPLICATIONS - 1}
#SBATCH --output={LOG_DIR}/{dataset_name}/out/{job_name}_%a.out
#SBATCH --error={LOG_DIR}/{dataset_name}/error/{job_name}_%a.err
#SBATCH --time={SLURM_CONFIG['time']}
#SBATCH --mem-per-cpu={SLURM_CONFIG['mem_per_cpu']}
#SBATCH --mail-type={SLURM_CONFIG['mail_type']}
#SBATCH --mail-user={SLURM_CONFIG['mail_user']}

cd {PROJECT_ROOT}
echo "Running {job_name} | Seed (Task ID): $SLURM_ARRAY_TASK_ID"
{python_command}
"""
    dataset_sbatch_dir = SBATCH_DIR / dataset_name
    sbatch_path = dataset_sbatch_dir / f"submit_{job_name}.sbatch"
    with open(sbatch_path, 'w') as f:
        f.write(sbatch_content)
    os.chmod(sbatch_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    print(f"  -> Generated script for {job_name}")


if __name__ == "__main__":
    datasets = sorted([f.stem for f in DATA_DIR.glob("*.pkl")])
    if not datasets:
        print(f"Warning: No .pkl files found in {DATA_DIR}. No sbatch files will be generated.")

    print("--- Generating sbatch job array files ---")
    for dataset in datasets:
        dataset_sbatch_dir = SBATCH_DIR / dataset
        dataset_log_dir = LOG_DIR / dataset
        
        dataset_sbatch_dir.mkdir(parents=True, exist_ok=True)
        (dataset_log_dir / "out").mkdir(parents=True, exist_ok=True)
        (dataset_log_dir / "error").mkdir(parents=True, exist_ok=True)
        
        print(f"\nGenerating scripts for dataset: {dataset}")
        for exp_config in EXPERIMENT_CONFIGS:
            create_sbatch_file(dataset, config=exp_config)

        run_all_content = "#!/bin/bash\nfor file in ./*.sbatch; do sbatch \"$file\"; done"
        run_all_path = dataset_sbatch_dir / "run_all.sh"
        with open(run_all_path, 'w') as f:
            f.write(run_all_content)
        os.chmod(run_all_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        print(f"  -> Generated 'run_all.sh' in {dataset_sbatch_dir}")

        # --- UPDATED: Create two separate cleanup scripts ---
        
        # Script 1: Cleans logs and sbatch files
        cleanup_logs_content = f"""#!/bin/bash
# This script cleans up logs and sbatch files for the {dataset} dataset.
SCRIPT_DIR=$( cd -- "$( dirname -- "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( cd -- "${SCRIPT_DIR}/../../" &> /dev/null && pwd )
DATASET_NAME=$(basename "$SCRIPT_DIR")
DATASET_LOG_DIR="${{PROJECT_ROOT}}/experiments/job_logs/${{DATASET_NAME}}"
echo "Cleaning up logs and sbatch files for dataset: ${{DATASET_NAME}}"
echo "  -> Deleting .sbatch files in current directory..."
find . -maxdepth 1 -type f -name "*.sbatch" -delete
echo "  -> Deleting log directory: ${{DATASET_LOG_DIR}}"
if [ -d "$DATASET_LOG_DIR" ]; then rm -rf "$DATASET_LOG_DIR"; echo "     Log directory deleted."; else echo "     Log directory not found."; fi
echo "  -> Deleting helper scripts (run_all.sh and this script)..."
rm -- "$0"
rm -- "run_all.sh"
echo "Log cleanup complete."
"""
        cleanup_logs_path = dataset_sbatch_dir / "cleanup_logs.sh"
        with open(cleanup_logs_path, 'w') as f:
            f.write(cleanup_logs_content)
        os.chmod(cleanup_logs_path, stat.S_IRWXU)
        print(f"  -> Generated 'cleanup_logs.sh' in {dataset_sbatch_dir}")

        # Script 2: Cleans ONLY the .pkl result files
        cleanup_results_content = f"""#!/bin/bash
# This script cleans up ONLY the result .pkl files for the {dataset} dataset.
SCRIPT_DIR=$( cd -- "$( dirname -- "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( cd -- "${{SCRIPT_DIR}}/../../" &> /dev/null && pwd )
DATASET_NAME=$(basename "$SCRIPT_DIR")
DATASET_RESULTS_DIR="${{PROJECT_ROOT}}/src/results/${{DATASET_NAME}}"
echo "Cleaning up result files for dataset: ${{DATASET_NAME}}"
echo "  -> Deleting results directory: ${{DATASET_RESULTS_DIR}}"
if [ -d "$DATASET_RESULTS_DIR" ]; then
    rm -rf "$DATASET_RESULTS_DIR"
    echo "     Result directory deleted."
else
    echo "     Result directory not found."
fi
echo "Result cleanup complete."
"""
        cleanup_results_path = dataset_sbatch_dir / "cleanup_results.sh"
        with open(cleanup_results_path, 'w') as f:
            f.write(cleanup_results_content)
        os.chmod(cleanup_results_path, stat.S_IRWXU)
        print(f"  -> Generated 'cleanup_results.sh' in {dataset_sbatch_dir}")

    print("\n--- Done ---")