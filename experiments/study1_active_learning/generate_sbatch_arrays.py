### LIBRARIES ###
import os
import stat
from pathlib import Path
from master_config import (BASE_SELECTORS, N_REPLICATIONS, SLURM_CONFIG, STUDIES)

### PATHS ###
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent 
LOG_DIR = SCRIPT_DIR / "slurm_logs"
SBATCH_ROOT_DIR = SCRIPT_DIR / "job_scripts"
DATA_DIR = PROJECT_ROOT / "src" / "data"

### HELPER: CREATE SBATCH FILE ###
def create_sbatch_file(dataset_name: str, 
                       dataset_folder_name: str, 
                       threshold: float, 
                       config: dict, 
                       method_number: int,
                       study_name: str,
                       sbatch_dir: Path): 
    
    selector_model_name = config["selector_model"]
    predictor_model_name = config["predictor_model"]
    selector_name = config["selector"]
    params = config["params"]    
    job_name = f"{dataset_name}_M{method_number}"    
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
    --study_dir {study_name} \\
    {params_str}
"""

    sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --array=0-{N_REPLICATIONS - 1}
#SBATCH --output={LOG_DIR}/{study_name}/{dataset_name}/out/M{method_number}_S%a.out
#SBATCH --error={LOG_DIR}/{study_name}/{dataset_name}/error/M{method_number}_S%a.err
#SBATCH --time={SLURM_CONFIG['time']}
#SBATCH --mem-per-cpu={SLURM_CONFIG['mem_per_cpu']}
#SBATCH --mail-type={SLURM_CONFIG['mail_type']}
#SBATCH --mail-user={SLURM_CONFIG['mail_user']}

cd {PROJECT_ROOT}

# --- ENVIRONMENT SETUP ---
module load Python/3.10.8-GCCcore-12.2.0
source .RAL_CL/bin/activate
export PYTHONPATH=$PYTHONPATH:.
# -------------------------

export PYTHONDONTWRITEBYTECODE=1

echo "Running {job_name} | Seed (Task ID): $SLURM_ARRAY_TASK_ID"
echo "Study: {study_name}"

echo "Sleeping to stagger start times..."
sleep $((RANDOM % 60 + 1))

{python_command}
"""
    
    sbatch_path = sbatch_dir / f"submit_{job_name}.sbatch"
    with open(sbatch_path, 'w') as f: f.write(sbatch_content)
    os.chmod(sbatch_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)

### HELPER: GENERATE MASTER SCRIPTS ###
def generate_master_scripts(study_name: str, study_sbatch_dir: Path):
    """
    Generates the 7 master control scripts in the study root.
    """
    
    # 0. IGNITION SCRIPT
    ignite_content = f"""#!/bin/bash
if pgrep -f "1_smart_run.sh" > /dev/null; then
    echo "Smart Run is ALREADY running!"
else
    echo "Igniting Smart Run in the background..."
    nohup ./1_smart_run.sh > smart_run_log.txt 2>&1 &
    echo "Success! Process ID: $!"
    echo "Logs: tail -f smart_run_log.txt"
fi
"""

    # 0. KILL SWITCH
    kill_content = f"""#!/bin/bash
if pgrep -f "1_smart_run.sh" > /dev/null; then
    echo "Stopping Smart Run..."
    pkill -f "1_smart_run.sh"
    echo "Smart Run stopped."
else
    echo "Smart Run is not running."
fi
read -p "Also cancel all your Slurm jobs? (y/n): " s_kill
[[ "$s_kill" == "y" ]] && scancel -u $USER
"""

    # 1. SMART RUN SCRIPT
    smart_run_content = f"""#!/bin/bash
MAX_JOBS=1800
CHECK_INTERVAL=60
STUDY_DIR="{study_name}"

SCRIPT_DIR=$( cd -- "$( dirname -- "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )
DATASETS_DIR="$SCRIPT_DIR/datasets"

echo "--------------------------------------------------------"
echo "🚀 Smart Launcher: Resilient Mode"
echo "   Target: $STUDY_DIR | Max Jobs: $MAX_JOBS"
echo "--------------------------------------------------------"

get_job_count() {{
    # -r expands job arrays so we count tasks, not just the array ID
    squeue -u $USER -h -r | wc -l
}}

check_dataset_running() {{
    local ds_name=$1
    squeue -u $USER -h -o %j | grep -q "${{ds_name}}_M"
}}

for dataset_path in "$DATASETS_DIR"/*; do
    if [ -d "$dataset_path" ]; then
        dataset_name=$(basename "$dataset_path")
        
        # Calculate Load for this Dataset
        sbatch_files=("$dataset_path"/submit_*.sbatch)
        num_scripts=${{#sbatch_files[@]}}
        total_tasks=$(( num_scripts * {N_REPLICATIONS} ))

        if [ "$num_scripts" -eq 0 ]; then continue; fi

        echo "📂 Processing: $dataset_name (Needs $total_tasks slots)"

        if check_dataset_running "$dataset_name"; then
             echo "Skipping $dataset_name (Already in Queue)"
             continue
        fi
        
        while true; do
            current_jobs=$(get_job_count)
            
            if [ $(( current_jobs + total_tasks )) -le "$MAX_JOBS" ]; then
                echo "   Launching $dataset_name (Load: $current_jobs + $total_tasks <= $MAX_JOBS)"
                for f in "${{sbatch_files[@]}}"; do sbatch "$f" > /dev/null; done
                sleep 5
                break 
            else
                echo "   ⏳ Queue Full ($current_jobs/$MAX_JOBS). Need $total_tasks. Sleeping ${{CHECK_INTERVAL}}s..."
                sleep $CHECK_INTERVAL
            fi
        done
    fi
done
echo "--------------------------------------------------------"
echo "🎉 All Datasets Submitted! Exiting."
echo "--------------------------------------------------------"
"""

    # 2. GLOBAL SMART AGGREGATE
    smart_agg_content = f"""#!/bin/bash
STUDY_DIR="study1_active_learning/{study_name}"
REQUIRED_COUNT={N_REPLICATIONS}
METHODS=("M1" "M2" "M3" "M4" "M5" "M6" "M7")

SCRIPT_DIR=$( cd -- "$( dirname -- "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")
RESULTS_ROOT="$PROJECT_ROOT/results/$STUDY_DIR"
JOB_SCRIPTS_DIR="$SCRIPT_DIR/datasets"

echo "🔍 Global Smart Aggregate: Checking $STUDY_DIR"

for dataset_path in "$RESULTS_ROOT"/*; do
    if [ -d "$dataset_path" ]; then
        dataset_name=$(basename "$dataset_path")
        [[ "$dataset_name" == "aggregated" ]] && continue

        complete=true
        for m in "${{METHODS[@]}}"; do
            count=$(ls -1 "$dataset_path/$m"/*.pkl 2>/dev/null | wc -l)
            [[ "$count" -ne "$REQUIRED_COUNT" ]] && complete=false && break
        done

        if [ "$complete" = true ]; then
            agg_script="$JOB_SCRIPTS_DIR/$dataset_name/2_aggregate_results.sbatch"
            [ -f "$agg_script" ] && sbatch "$agg_script"
        fi
    fi
done
"""

    # 3. GLOBAL SMART PLOT
    smart_plot_content = f"""#!/bin/bash
STUDY_DIR="study1_active_learning/{study_name}"
SCRIPT_DIR=$( cd -- "$( dirname -- "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")
RESULTS_ROOT="$PROJECT_ROOT/results/$STUDY_DIR"
JOB_SCRIPTS_DIR="$SCRIPT_DIR/datasets"

echo "🎨 Global Smart Plotter: Launching plots for aggregated datasets"

for dataset_path in "$RESULTS_ROOT"/*; do
    dataset_name=$(basename "$dataset_path")
    if [ -d "$dataset_path/aggregated" ]; then
        plot_script="$JOB_SCRIPTS_DIR/$dataset_name/3_plot_results.sbatch"
        [ -f "$plot_script" ] && sbatch "$plot_script"
    fi
done
"""

    # 4. GLOBAL LOG CLEANUP
    global_log_clean = f"""#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )
echo "This will clear ALL .out and .err logs for this study."
read -p "Continue? (y/n): " confirm
if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
    find "$SCRIPT_DIR/datasets" -name "4_cleanup_logs.sh" -exec bash -c 'echo "y" | {{}}' \;
    echo "All logs cleared."
fi
"""

    # 5. GLOBAL RESULTS CLEANUP
    global_res_clean = f"""#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )
echo "WARNING: This will delete ALL raw .pkl seeds (M1-M10)."
read -p "Are you absolutely sure? (y/n): " confirm
if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
    find "$SCRIPT_DIR/datasets" -name "5_delete_raw_results.sh" -exec bash -c 'echo "y" | {{}}' \;
    echo "All raw results cleared."
fi
"""
    
    # 6. COLLECT PLOTS
    collect_plots_content = f"""##!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../../../../")

echo "Generating standalone legend..."
python "$PROJECT_ROOT/src/utils/plot_results.py" --legend-only --study_dir "study1_active_learning/tree_predictor"

echo "Generating AUC Heatmaps for multiple budgets..."
for budget in 0.5 0.6 0.7 0.8 0.9 1.0; do
    echo "  > Budget Fraction: $budget"
    python "$PROJECT_ROOT/src/utils/generate_auc_heatmaps.py" --budget_fraction "$budget"
done

echo "Generating Label Efficiency Plots..."
python "$PROJECT_ROOT/src/utils/generate_label_efficiency.py"

echo "Generating Runtime Comparison Table..."
python "$PROJECT_ROOT/src/utils/generate_runtime_table.py"

echo "Collecting all plots into final directory..."
python "$PROJECT_ROOT/src/utils/collect_plots.py"

echo "Done! All figures and tables generated."
"""

    # --- Write Files ---
    scripts = {
        "0a_ignite.sh": ignite_content,
        "0b_kill.sh": kill_content,
        "1_smart_run.sh": smart_run_content,
        "2_global_aggregate.sh": smart_agg_content,
        "3a_global_plot.sh": smart_plot_content,
        "3b_collect_plots.sh": collect_plots_content,
        "4_global_cleanup_logs.sh": global_log_clean,
        "5_global_cleanup_results.sh": global_res_clean
    }

    for name, content in scripts.items():
        path = study_sbatch_dir / name
        with open(path, 'w') as f: f.write(content)
        os.chmod(path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)

### MAIN SCRIPT ###
if __name__ == "__main__":
    datasets = sorted([f.stem for f in DATA_DIR.glob("*.pkl") if not f.stem.startswith(".")])
    
    for study in STUDIES:
        study_name = study["name"]
        predictor  = study["predictor"]
        study_sbatch_dir = SBATCH_ROOT_DIR / study_name
        study_sbatch_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n=== STUDY: {study_name} ===")

        for dataset in datasets:
            print(f"🛠️  Generating: {dataset}\n", end=" ", flush=True)
            dataset_sbatch_dir = study_sbatch_dir / "datasets" / dataset
            dataset_log_dir = LOG_DIR / study_name / dataset
            dataset_sbatch_dir.mkdir(parents=True, exist_ok=True)
            (dataset_log_dir / "out").mkdir(parents=True, exist_ok=True)
            (dataset_log_dir / "error").mkdir(parents=True, exist_ok=True)
            
            # Individual Method Sbatch
            for idx, selector_config in enumerate(BASE_SELECTORS):
                full_config = selector_config.copy()
                full_config["predictor_model"] = predictor 
                create_sbatch_file(dataset, dataset, full_config["fixed_threshold"], 
                                   full_config, idx + 1, study_name, dataset_sbatch_dir)

            # Local Aggregate
            agg_content = f"""#!/bin/bash
#SBATCH --job-name=Agg_{dataset}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --output={dataset_log_dir}/out/Agg.out
#SBATCH --error={dataset_log_dir}/error/Agg.err
#SBATCH --time=00:20:00
#SBATCH --mem=2G
cd {PROJECT_ROOT}
module load Python/3.10.8-GCCcore-12.2.0
source .RAL_CL/bin/activate
export PYTHONPATH=$PYTHONPATH:.
python src/utils/aggregate_results.py --dataset "{dataset}" --study_dir "study1_active_learning/{study_name}"
"""
            with open(dataset_sbatch_dir / "2_aggregate_results.sbatch", 'w') as f: f.write(agg_content)

            # Local Plot
            plot_content = f"""#!/bin/bash
#SBATCH --job-name=Plot_{dataset}
#SBATCH --partition={SLURM_CONFIG['partition']}
#SBATCH --output={dataset_log_dir}/out/Plot.out
#SBATCH --error={dataset_log_dir}/error/Plot.err
#SBATCH --time=00:20:00
#SBATCH --mem=2G
cd {PROJECT_ROOT}
source .RAL/bin/activate
python src/utils/plot_results.py --dataset "{dataset}" --study_dir "study1_active_learning/{study_name}" --no-legend

"""
            with open(dataset_sbatch_dir / "3_plot_results.sbatch", 'w') as f: f.write(plot_content)

            # Local Cleanups
            with open(dataset_sbatch_dir / "4_cleanup_logs.sh", 'w') as f:
                f.write(f'rm -f "{dataset_log_dir}/error"/*.err "{dataset_log_dir}/out"/*.out\n')

            with open(dataset_sbatch_dir / "5_delete_raw_results.sh", 'w') as f:
                # 1. Delete the .pkl files
                f.write(f'rm -f "{PROJECT_ROOT}/results/study1_active_learning/{study_name}/{dataset}"/M*/*.pkl\n')
                # 2. Find and delete the now-empty M folders
                f.write(f'find "{PROJECT_ROOT}/results/study1_active_learning/{study_name}/{dataset}" -type d -name "M*" -empty -delete\n')
                f.write(f'echo "Raw .pkl files and empty method folders deleted for {dataset}."\n')

            # Local Run All
            run_all_content = f"""#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${{BASH_SOURCE[0]}}" )" &> /dev/null && pwd )

echo "🚀 Submitting all jobs for {dataset}..."
for sbatch_file in "$SCRIPT_DIR"/submit_*.sbatch; do
    # echo "  -> Submitting $sbatch_file"
    sbatch "$sbatch_file"
done
echo "Done."
"""
            with open(dataset_sbatch_dir / "1_run_all.sh", 'w') as f: f.write(run_all_content)

            # Make local cleanups executable
            for s in ["1_run_all.sh", "4_cleanup_logs.sh", "5_delete_raw_results.sh"]:
                os.chmod(dataset_sbatch_dir / s, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)

            # print("Done!")

        generate_master_scripts(study_name, study_sbatch_dir)
    print("\n--- All Scripts Generated Successfully ---")