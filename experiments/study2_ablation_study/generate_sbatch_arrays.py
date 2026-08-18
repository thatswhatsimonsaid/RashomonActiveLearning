### Summary ###
"""
Job factory for Study 2: Epsilon Sensitivity Ablation.
Generates .sbatch array jobs and master orchestrator scripts.
"""

import os
import stat
from pathlib import Path
from master_config import (BASE_SELECTORS, N_REPLICATIONS, SLURM_CONFIG, STUDIES, TASK_TYPE, DATASETS)

### PATHS ###
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent 
LOG_DIR = SCRIPT_DIR / "slurm_logs"
SBATCH_ROOT_DIR = SCRIPT_DIR / "job_scripts"
DATA_DIR = PROJECT_ROOT / "src" / "data" / "classification"

### HELPER: CREATE SBATCH FILE ###
def create_sbatch_file(dataset_name: str, 
                       method_label: str,
                       method_number: int,
                       threshold: float, 
                       config: dict, 
                       full_study_path: str,
                       sbatch_dir: Path): 
    
    selector_model_name = config["selector_model"]
    predictor_model_name = config["predictor_model"]
    selector_name = config["selector"]
    params = config["params"]    
    
    job_name = f"{dataset_name}_{method_label}"    
    params_str = " ".join([f"{k}={v}" for k, v in params.items()])
    python_executable = PROJECT_ROOT / ".RAL_CL/bin/python"    
    python_command = f"""
{python_executable} src/utils/run_experiment.py \\
    --dataset {dataset_name} \\
    --selector_model {selector_model_name} \\
    --predictor_model {predictor_model_name} \\
    --selector {selector_name} \\
    --seed $SLURM_ARRAY_TASK_ID \\
    --method_number {method_number} \\
    --rashomon_threshold {threshold} \\
    --study_dir {full_study_path} \\
    task_type={TASK_TYPE} \\
    {params_str}
"""

    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --array=0-{N_REPLICATIONS - 1}
#SBATCH --output={LOG_DIR}/{full_study_path.split('/')[-1]}/{dataset_name}/{method_label}/out/{method_label}_S%a.out
#SBATCH --error={LOG_DIR}/{full_study_path.split('/')[-1]}/{dataset_name}/{method_label}/error/{method_label}_S%a.err
#SBATCH --time={SLURM_CONFIG['time']}
#SBATCH --mem-per-cpu={SLURM_CONFIG['mem_per_cpu']}
#SBATCH --mail-type={SLURM_CONFIG['mail_type']}
#SBATCH --mail-user={SLURM_CONFIG['mail_user']}

cd {PROJECT_ROOT}
module load Python/3.10.8-GCCcore-12.2.0
source .RAL_CL/bin/activate
export PYTHONPATH=$PYTHONPATH:.
export PYTHONDONTWRITEBYTECODE=1

echo "Running {job_name} | Seed: $SLURM_ARRAY_TASK_ID"
echo "Target Path: results/{full_study_path}"

sleep $((RANDOM % 30 + 1))
{python_command}
"""
    
    sbatch_path = sbatch_dir / f"submit_{job_name}.sbatch"
    with open(sbatch_path, 'w') as f: f.write(sbatch_content)
    os.chmod(sbatch_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)

### HELPER: GENERATE MASTER SCRIPTS ###
def generate_master_scripts(full_study_path: str, study_sbatch_dir: Path):
    """Generates the control scripts using the corrected full path."""
    
    # 1. SMART RUN
    smart_run_content = f"""#!/bin/bash
MAX_JOBS=1800
CHECK_INTERVAL=60
SCRIPT_DIR=$( cd -- "$( dirname -- "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )
DATASETS_DIR="$SCRIPT_DIR/datasets"

get_job_count() {{
    squeue -u $USER -h -r | wc -l
}}

for dataset_path in "$DATASETS_DIR"/*; do
    if [ -d "$dataset_path" ]; then
        dataset_name=$(basename "$dataset_path")
        sbatch_files=("$dataset_path"/submit_*.sbatch)
        num_scripts=${{#sbatch_files[@]}}
        total_tasks=$(( num_scripts * {N_REPLICATIONS} ))

        echo "Checking: $dataset_name"
        while true; do
            current_jobs=$(get_job_count)
            if [ $(( current_jobs + total_tasks )) -le "$MAX_JOBS" ]; then
                for f in "${{sbatch_files[@]}}"; do sbatch "$f" > /dev/null; done
                break 
            else
                sleep $CHECK_INTERVAL
            fi
        done
    fi
done
"""

    # 2. GLOBAL SMART AGGREGATE
    smart_agg_content = f"""#!/bin/bash
FULL_STUDY_PATH="{full_study_path}"
REQUIRED_COUNT={N_REPLICATIONS}

SCRIPT_DIR=$( cd -- "$( dirname -- "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )
PROJECT_ROOT="{PROJECT_ROOT}"
RESULTS_ROOT="$PROJECT_ROOT/results/$FULL_STUDY_PATH"
JOB_SCRIPTS_DIR="$SCRIPT_DIR/datasets"

echo "Global Smart Aggregate: Checking $RESULTS_ROOT"

for dataset_folder in "$RESULTS_ROOT"/*; do
    if [ -d "$dataset_folder" ]; then
        dataset_name=$(basename "$dataset_folder")
        [[ "$dataset_name" == "aggregated" ]] && continue

        # Count total pkl files across all M* directories
        total_pkls=$(find "$dataset_folder" -name "*.pkl" | wc -l)
        expected=$(( 10 * REQUIRED_COUNT )) 

        if [ "$total_pkls" -ge "$expected" ]; then
            agg_script="$JOB_SCRIPTS_DIR/$dataset_name/2_aggregate_results.sbatch"
            [ -f "$agg_script" ] && sbatch "$agg_script"
        else
            echo "  > $dataset_name: $total_pkls/$expected files. Skipping."
        fi
    fi
done
"""

    # 3. GLOBAL SMART PLOT
    smart_plot_content = f"""#!/bin/bash
FULL_STUDY_PATH="{full_study_path}"
SCRIPT_DIR=$( cd -- "$( dirname -- "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )
PROJECT_ROOT="{PROJECT_ROOT}"
RESULTS_ROOT="$PROJECT_ROOT/results/$FULL_STUDY_PATH"
JOB_SCRIPTS_DIR="$SCRIPT_DIR/datasets"

echo "Global Smart Plotter: Checking $RESULTS_ROOT"

for dataset_folder in "$RESULTS_ROOT"/*; do
    dataset_name=$(basename "$dataset_folder")
    if [ -d "$dataset_folder/aggregated" ]; then
        plot_script="$JOB_SCRIPTS_DIR/$dataset_name/3_plot_results.sbatch"
        [ -f "$plot_script" ] && sbatch "$plot_script"
    fi
done

echo "--------------------------------------------------------"
echo "Submissions complete. Generating standalone 3x3 legend..."
echo "--------------------------------------------------------"

cd "$PROJECT_ROOT"
module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null || true
source .RAL/bin/activate
export PYTHONPATH=$PYTHONPATH:.
python src/utils/plot_results_ABLATION.py --legend-only

echo "Legend generation complete!"
"""

    # 4. GLOBAL LOG CLEANUP
    global_log_clean = f"""#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )
echo "This will clear ALL .out and .err logs for this ablation study."
read -p "Continue? (y/n): " confirm
if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
    find "$SCRIPT_DIR/datasets" -name "4_cleanup_logs.sh" -exec bash -c 'echo "y" | {{}}' \\;
    echo "All logs cleared."
fi
"""

    # 5. GLOBAL RESULTS CLEANUP
    global_res_clean = f"""#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )
echo "WARNING: This will delete ALL raw .pkl seeds (M1-M10) for the ablation study."
read -p "Are you absolutely sure? (y/n): " confirm
if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
    find "$SCRIPT_DIR/datasets" -name "5_delete_raw_results.sh" -exec bash -c 'echo "y" | {{}}' \\;
    echo "All raw results cleared."
fi
"""

    # Define the dictionary
    scripts = {
        "0a_ignite.sh": "#!/bin/bash\nnohup ./1_smart_run.sh > smart_run_log.txt 2>&1 &\necho 'Launched.'",
        "0b_kill.sh": "#!/bin/bash\npkill -f '1_smart_run.sh'\nscancel -u $USER",
        "1_smart_run.sh": smart_run_content,
        "2_global_aggregate.sh": smart_agg_content,
        "3a_global_plot.sh": smart_plot_content,
        "4_global_cleanup_logs.sh": global_log_clean,
        "5_global_cleanup_results.sh": global_res_clean
    }

    # Write the files
    for name, content in scripts.items():
        path = study_sbatch_dir / name
        with open(path, 'w') as f: f.write(content)
        os.chmod(path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)

### MAIN SCRIPT ###
if __name__ == "__main__":
    target_datasets = DATASETS
    
    for study in STUDIES:
        study_name_raw = study["name"]
        
        full_study_path = f"study2_ablation_study/{study_name_raw}"
        
        predictor = study["predictor"]
        study_sbatch_dir = SBATCH_ROOT_DIR / study_name_raw
        study_sbatch_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n=== GENERATING STUDY: {study_name_raw} ===")

        for dataset in target_datasets:
            print(f"  > Dataset: {dataset}")
            dataset_sbatch_dir = study_sbatch_dir / "datasets" / dataset
            dataset_log_dir = LOG_DIR / study_name_raw / dataset
            dataset_sbatch_dir.mkdir(parents=True, exist_ok=True)

            for idx, selector_config in enumerate(BASE_SELECTORS):
                method_num = idx + 1
                label = selector_config.get("label_suffix", f"M{method_num}")

                method_log_dir = dataset_log_dir / label
                (method_log_dir / "out").mkdir(parents=True, exist_ok=True)
                (method_log_dir / "error").mkdir(parents=True, exist_ok=True)
                
                create_sbatch_file(
                    dataset_name=dataset, 
                    method_label=label, 
                    method_number=method_num,
                    threshold=selector_config["fixed_threshold"], 
                    config={**selector_config, "predictor_model": predictor}, 
                    full_study_path=full_study_path, 
                    sbatch_dir=dataset_sbatch_dir
                )

            # Local 1_run_all.sh
            with open(dataset_sbatch_dir / "1_run_all.sh", 'w') as f:
                f.write(f"#!/bin/bash\nfor f in submit_*.sbatch; do sbatch \"$f\"; done")

            agg_content = f"""#!/bin/bash
#SBATCH --job-name=Agg_{dataset}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output={dataset_log_dir}/aggregate.out
#SBATCH --error={dataset_log_dir}/aggregate.err
cd {PROJECT_ROOT}
module load Python/3.10.8-GCCcore-12.2.0
source .RAL_CL/bin/activate
export PYTHONPATH=$PYTHONPATH:.
python src/utils/aggregate_results.py --dataset "{dataset}" --study_dir "{full_study_path}"
"""
            with open(dataset_sbatch_dir / "2_aggregate_results.sbatch", 'w') as f: f.write(agg_content)

            plot_content = f"""#!/bin/bash
#SBATCH --job-name=Plot_{dataset}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --output={dataset_log_dir}/plot.out
#SBATCH --error={dataset_log_dir}/plot.err
cd {PROJECT_ROOT}
module load Python/3.10.8-GCCcore-12.2.0
source .RAL_CL/bin/activate
export PYTHONPATH=$PYTHONPATH:.
python src/utils/plot_results_ABLATION.py --dataset "{dataset}" --study_dir "{full_study_path}"
"""
            with open(dataset_sbatch_dir / "3_plot_results.sbatch", 'w') as f: f.write(plot_content)

    
            with open(dataset_sbatch_dir / "4_cleanup_logs.sh", 'w') as f:
                f.write(f'rm -f "{dataset_log_dir}"/*/error/*.err "{dataset_log_dir}"/*/out/*.out\n')
                f.write(f'rm -f "{dataset_log_dir}"/*.err "{dataset_log_dir}"/*.out\n')

            with open(dataset_sbatch_dir / "5_delete_raw_results.sh", 'w') as f:
                f.write(f'rm -f "{PROJECT_ROOT}/results/{full_study_path}/{dataset}"/M*/*.pkl\n')
                f.write(f'find "{PROJECT_ROOT}/results/{full_study_path}/{dataset}" -type d -name "M*" -empty -delete\n')
                f.write(f'echo "Raw .pkl files and empty method folders deleted for {dataset}."\n')
                
            for s in ["1_run_all.sh", "4_cleanup_logs.sh", "5_delete_raw_results.sh"]:
                os.chmod(dataset_sbatch_dir / s, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)

        generate_master_scripts(full_study_path, study_sbatch_dir)
    print("\n--- DONE ---")