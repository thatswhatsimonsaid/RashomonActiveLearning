### SUMMARY ###
"""
Generator script for the DGP entropy estimation hyperparameter sweep.

Creates a complete, self-contained workflow for each Rashomon threshold specified.
Each workflow includes a SLURM job array script and numbered helper scripts
to manage the simulation lifecycle (run, aggregate, plot, cleanup).
"""

### LIBRARIES ###
import stat
from pathlib import Path

### INPUPT ###
N_REPLICATIONS = 50
RASHOMON_THRESHOLDS = [0.01, 0.02, 0.05, 0.075]
SIMULATION_CONFIG = {
    "n_train": 200,
    "n_candidate": 1000,
    "rf_estimators": 100,
    "tf_regularization": 0.005,
    "noise_level": 0.1
}

SLURM_CONFIG = {
    "partition": "largemem",
    "time": "11:59:00",
    "mem_per_cpu": "100GB",
    "mail_type": "ALL",
    "mail_user": "simondn@uw.edu",
}

# --- Paths ---
STUDY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STUDY_DIR.parent.parent
WORKFLOWS_DIR = STUDY_DIR / "job_workflows"


def generate_workflow_scripts(threshold: float):
    """Generates all .sbatch and helper .sh scripts for a single threshold."""
    
    threshold_str = str(threshold).replace('.', '_')
    output_dir = WORKFLOWS_DIR / f"thresh_{threshold_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logs_out_dir = output_dir / "logs" / "out"
    logs_err_dir = output_dir / "logs" / "error"
    logs_out_dir.mkdir(parents=True, exist_ok=True)
    logs_err_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Create the main .sbatch array script ---
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name=S3_ENT_{threshold_str}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --time={SLURM_CONFIG['time']}
#SBATCH --mem-per-cpu={SLURM_CONFIG['mem_per_cpu']}
#SBATCH --array=1-{N_REPLICATIONS}
#SBATCH --mail-type={SLURM_CONFIG['mail_type']}
#SBATCH --mail-user={SLURM_CONFIG['mail_user']}
#SBATCH --output={logs_out_dir}/dgp_entropy_seed_%a.out
#SBATCH --error={logs_err_dir}/dgp_entropy_seed_%a.err

PROJECT_ROOT="{PROJECT_ROOT}"
cd $PROJECT_ROOT
source .RAL/bin/activate

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "--- Running DGP Entropy Study | Threshold: {threshold}, Seed: $SLURM_ARRAY_TASK_ID ---"

python -m experiments.study3_entropy_validation.entropy_estimation_study run \\
    --n_train {SIMULATION_CONFIG['n_train']} \\
    --n_candidate {SIMULATION_CONFIG['n_candidate']} \\
    --rf_estimators {SIMULATION_CONFIG['rf_estimators']} \\
    --tf_regularization {SIMULATION_CONFIG['tf_regularization']} \\
    --tf_rashomon_threshold {threshold} \\
    --noise_level {SIMULATION_CONFIG['noise_level']} \\
    --seed $SLURM_ARRAY_TASK_ID
"""
    sbatch_path = output_dir / "submit_entropy_study_array.sbatch"
    with open(sbatch_path, 'w') as f: f.write(sbatch_content)

    # --- 2. Create Helper Scripts ---
    run_all_content = f"""#!/bin/bash
sbatch submit_entropy_study_array.sbatch
"""
    (output_dir / "1_run_all.sh").write_text(run_all_content)

    aggregate_content = f"""#!/bin/bash
PROJECT_ROOT="{PROJECT_ROOT}"
cd $PROJECT_ROOT
source .RAL/bin/activate
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
python -m experiments.study3_entropy_validation.entropy_estimation_study aggregate --tf_rashomon_threshold {threshold}
"""
    (output_dir / "2_aggregate_results.sh").write_text(aggregate_content)

    plot_content = f"""#!/bin/bash
PROJECT_ROOT="{PROJECT_ROOT}"
cd $PROJECT_ROOT
source .RAL/bin/activate
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
python -m experiments.study3_entropy_validation.entropy_estimation_study plot --tf_rashomon_threshold {threshold}
"""
    (output_dir / "3_generate_image.sh").write_text(plot_content)

    cleanup_results_content = f"""#!/bin/bash
RAW_RESULTS_DIR="{PROJECT_ROOT}/results/study3_entropy_validation/raw/thresh_{threshold_str}"
echo "Deleting raw .pkl files from: $RAW_RESULTS_DIR"
rm -rf "$RAW_RESULTS_DIR"
echo "Raw results cleanup complete."
"""
    (output_dir / "4_cleanup_results.sh").write_text(cleanup_results_content)

    cleanup_logs_content = f"""#!/bin/bash
echo "Cleaning up log files from ./logs..."
if [ -d "./logs" ]; then
    rm -rf ./logs
    echo "Log directory deleted."
else
    echo "Log directory not found. Nothing to do."
fi
echo "Log cleanup complete."
"""
    (output_dir / "5_cleanup_logs.sh").write_text(cleanup_logs_content)
    
    for sh_file in output_dir.glob("*.sh"):
        sh_file.chmod(sh_file.stat().st_mode | stat.S_IEXEC)
    
    print(f"  -> Generated workflow for threshold {threshold} in {output_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    print("--- Generating Workflows for DGP Entropy Study Hyperparameter Sweep ---")
    
    for threshold in RASHOMON_THRESHOLDS:
        generate_workflow_scripts(threshold)
        
    print(f"\n--- Generation complete. Sub-directories created in '{WORKFLOWS_DIR.relative_to(PROJECT_ROOT)}'. ---")