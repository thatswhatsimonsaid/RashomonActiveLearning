### LIBRARIES ###
import os
import stat
from pathlib import Path
from master_config import (
    BASE_SELECTORS,
    DATASETS,
    N_REPLICATIONS,
    RASHOMON_THRESHOLD,
    SLURM_CONFIG,
    STUDIES,
    TASK_TYPE,
)

### PATHS ###
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
LOG_DIR = SCRIPT_DIR / "slurm_logs"
SBATCH_ROOT_DIR = SCRIPT_DIR / "job_scripts"


def create_sbatch_file(dataset_name, config, method_number, full_study_path, sbatch_dir):
    params = config["params"]
    params_str = " ".join([f"{key}={value}" for key, value in params.items()])
    python_executable = PROJECT_ROOT / ".RAL_CL/bin/python"
    job_name = f"{dataset_name}_M{method_number}"
    python_command = f"""
{python_executable} src/utils/run_experiment.py \\
    --dataset {dataset_name} \\
    --selector_model {config["selector_model"]} \\
    --predictor_model {config["predictor_model"]} \\
    --selector {config["selector"]} \\
    --seed $SLURM_ARRAY_TASK_ID \\
    --method_number {method_number} \\
    --rashomon_threshold {RASHOMON_THRESHOLD} \\
    --task_type {TASK_TYPE} \\
    --study_dir {full_study_path} \\
    {params_str}
"""
    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --array=0-{N_REPLICATIONS - 1}
#SBATCH --output={LOG_DIR}/{full_study_path.split('/')[-1]}/{dataset_name}/out/M{method_number}_S%a.out
#SBATCH --error={LOG_DIR}/{full_study_path.split('/')[-1]}/{dataset_name}/error/M{method_number}_S%a.err
#SBATCH --time={SLURM_CONFIG['time']}
#SBATCH --mem-per-cpu={SLURM_CONFIG['mem_per_cpu']}
#SBATCH --mail-type={SLURM_CONFIG['mail_type']}
#SBATCH --mail-user={SLURM_CONFIG['mail_user']}

cd {PROJECT_ROOT}

# Batch shells on this cluster do not define `module`. Point at the same
# EasyBuild Python that built .RAL_CL, then load the module when it exists.
export LD_LIBRARY_PATH=/sw/ebpkgs/software/Python/3.10.8-GCCcore-12.2.0/lib:${{LD_LIBRARY_PATH}}
export PATH=/sw/ebpkgs/software/Python/3.10.8-GCCcore-12.2.0/bin:${{PATH}}
if type module >/dev/null 2>&1; then
    module load Python/3.10.8-GCCcore-12.2.0
fi
source .RAL_CL/bin/activate
export PYTHONPATH=$PYTHONPATH:.
export PYTHONDONTWRITEBYTECODE=1

echo "Running {job_name} | Seed (Task ID): $SLURM_ARRAY_TASK_ID"
{python_command}
"""
    sbatch_path = sbatch_dir / f"submit_{job_name}.sbatch"
    with open(sbatch_path, "w") as handle:
        handle.write(sbatch_content)
    os.chmod(sbatch_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)


if __name__ == "__main__":
    for study in STUDIES:
        study_name = study["name"]
        full_study_path = f"study3_regression/{study_name}"
        study_sbatch_dir = SBATCH_ROOT_DIR / study_name
        print(f"\n=== STUDY: {study_name} ===")
        for dataset in DATASETS:
            print(f"  Generating: {dataset}")
            dataset_sbatch_dir = study_sbatch_dir / "datasets" / dataset
            dataset_log_dir = LOG_DIR / study_name / dataset
            dataset_sbatch_dir.mkdir(parents=True, exist_ok=True)
            (dataset_log_dir / "out").mkdir(parents=True, exist_ok=True)
            (dataset_log_dir / "error").mkdir(parents=True, exist_ok=True)
            for idx, selector_config in enumerate(BASE_SELECTORS):
                full_config = selector_config.copy()
                full_config["predictor_model"] = study["predictor"]
                create_sbatch_file(dataset, full_config, idx + 1, full_study_path, dataset_sbatch_dir)
    print("\n--- Regression job scripts generated. Nothing was submitted. ---")
