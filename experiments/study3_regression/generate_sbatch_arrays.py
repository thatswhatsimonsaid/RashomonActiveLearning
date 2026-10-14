### LIBRARIES ###
import os
import stat
from pathlib import Path
from master_config import (
    BASE_SELECTORS,
    MAX_QUEUED_JOBS,
    N_REPLICATIONS,
    RASHOMON_THRESHOLD,
    SLURM_CONFIG,
    STUDIES,
    SUBMIT_DATASETS,
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
#SBATCH --account={SLURM_CONFIG['account']}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --array=0-{N_REPLICATIONS - 1}
#SBATCH --output={LOG_DIR}/{full_study_path.split('/')[-1]}/{dataset_name}/out/M{method_number}_S%a.out
#SBATCH --error={LOG_DIR}/{full_study_path.split('/')[-1]}/{dataset_name}/error/M{method_number}_S%a.err
#SBATCH --time={SLURM_CONFIG['time']}
#SBATCH --mem={SLURM_CONFIG['mem']}
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
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
echo "Sleeping to stagger start times..."
sleep $((RANDOM % 30 + 1))
{python_command}
"""
    sbatch_path = sbatch_dir / f"submit_{job_name}.sbatch"
    with open(sbatch_path, "w") as handle:
        handle.write(sbatch_content)
    os.chmod(sbatch_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)


def generate_master_scripts(full_study_path: str, study_sbatch_dir: Path):
    """Write the queue-capped launcher used by the classification study."""
    dataset_list = " ".join(f'"{name}"' for name in SUBMIT_DATASETS)
    n_methods = len(BASE_SELECTORS)
    smart_run = f"""#!/bin/bash
MAX_JOBS={MAX_QUEUED_JOBS}
CHECK_INTERVAL=60
STUDY_DIR="{full_study_path}"
N_REPLICATIONS={N_REPLICATIONS}

SCRIPT_DIR=$( cd -- "$( dirname -- "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../../../../")
DATASETS_DIR="$SCRIPT_DIR/datasets"
PYTHON="$PROJECT_ROOT/.RAL_CL/bin/python"

cd "$PROJECT_ROOT"
export LD_LIBRARY_PATH=/sw/ebpkgs/software/Python/3.10.8-GCCcore-12.2.0/lib:${{LD_LIBRARY_PATH}}
export PATH=/sw/ebpkgs/software/Python/3.10.8-GCCcore-12.2.0/bin:${{PATH}}
export PYTHONPATH="$PROJECT_ROOT:${{PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1

if [ ! -x "$PYTHON" ]; then
    echo "Python not found at $PYTHON"
    exit 1
fi

echo "--------------------------------------------------------"
echo "Smart Launcher: regression"
echo "   Target: $STUDY_DIR | Max queued jobs: $MAX_JOBS"
echo "--------------------------------------------------------"

get_job_count() {{
    squeue -u "$USER" -h -r | wc -l
}}

DATASETS=({dataset_list})
for dataset_name in "${{DATASETS[@]}}"; do
    dataset_path="$DATASETS_DIR/$dataset_name"
    if [ ! -d "$dataset_path" ]; then
        echo "Missing scripts for $dataset_name"
        continue
    fi

    pending_files=()
    for sbatch_file in "$dataset_path"/submit_*.sbatch; do
        [ -e "$sbatch_file" ] || continue
        method_number=$(sed -n 's/.*_M\\([0-9][0-9]*\\)\\.sbatch/\\1/p' <<< "$sbatch_file")
        if squeue -u "$USER" -h -o %j | grep -qx "${{dataset_name}}_M${{method_number}}"; then
            echo "   Already queued: $dataset_name M$method_number"
            continue
        fi
        if "$PYTHON" src/utils/method_complete.py \\
            --dataset "$dataset_name" \\
            --method_number "$method_number" \\
            --study_dir "$STUDY_DIR" \\
            --n_replications "$N_REPLICATIONS"; then
            echo "   Already complete: $dataset_name M$method_number"
        else
            pending_files+=("$sbatch_file")
        fi
    done

    num_scripts=${{#pending_files[@]}}
    if [ "$num_scripts" -eq 0 ]; then
        echo "Skipping $dataset_name (nothing left to run)"
        continue
    fi
    total_tasks=$(( num_scripts * N_REPLICATIONS ))
    echo "Processing: $dataset_name (Needs $total_tasks slots, $num_scripts arrays)"

    while true; do
        current_jobs=$(get_job_count)
        if [ $(( current_jobs + total_tasks )) -le "$MAX_JOBS" ]; then
            echo "   Launching $dataset_name (Load: $current_jobs + $total_tasks <= $MAX_JOBS)"
            for sbatch_file in "${{pending_files[@]}}"; do
                sbatch "$sbatch_file"
            done
            sleep 15
            break
        else
            echo "   Queue full ($current_jobs/$MAX_JOBS). Need $total_tasks. Sleeping ${{CHECK_INTERVAL}}s..."
            sleep "$CHECK_INTERVAL"
        fi
    done
done

echo "--------------------------------------------------------"
echo "All requested datasets submitted."
echo "--------------------------------------------------------"
"""
    ignite = """#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"
if pgrep -f "1_smart_run.sh" > /dev/null; then
    echo "Smart Run is already running."
else
    echo "Starting Smart Run in the background..."
    nohup ./1_smart_run.sh > smart_run_log.txt 2>&1 &
    echo "Process ID: $!"
    echo "Logs: tail -f smart_run_log.txt"
fi
"""
    stop = """#!/bin/bash
if pgrep -f "1_smart_run.sh" > /dev/null; then
    pkill -f "1_smart_run.sh"
    echo "Smart Run stopped. Jobs already in the queue are still running."
else
    echo "Smart Run is not running."
fi
"""
    scripts = {
        "0a_ignite.sh": ignite,
        "0b_stop_submitter.sh": stop,
        "1_smart_run.sh": smart_run,
    }
    for name, content in scripts.items():
        path = study_sbatch_dir / name
        with open(path, "w") as handle:
            handle.write(content)
        os.chmod(path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    print(f"  Master scripts written ({n_methods} methods, cap {MAX_QUEUED_JOBS}).")


if __name__ == "__main__":
    for study in STUDIES:
        study_name = study["name"]
        full_study_path = f"study3_regression/{study_name}"
        study_sbatch_dir = SBATCH_ROOT_DIR / study_name
        study_sbatch_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== STUDY: {study_name} ===")
        for dataset in SUBMIT_DATASETS:
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
        generate_master_scripts(full_study_path, study_sbatch_dir)
    print("\n--- Regression job scripts generated. Nothing was submitted. ---")
