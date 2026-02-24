#!/bin/bash
MAX_JOBS=1800
CHECK_INTERVAL=60
STUDY_DIR="tree_predictor"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATASETS_DIR="$SCRIPT_DIR/datasets"

echo "--------------------------------------------------------"
echo "🚀 Smart Launcher: Resilient Mode"
echo "   Target: $STUDY_DIR | Max Jobs: $MAX_JOBS"
echo "--------------------------------------------------------"

get_job_count() {
    # -r expands job arrays so we count tasks, not just the array ID
    squeue -u $USER -h -r | wc -l
}

check_dataset_running() {
    local ds_name=$1
    squeue -u $USER -h -o %j | grep -q "${ds_name}_M"
}

for dataset_path in "$DATASETS_DIR"/*; do
    if [ -d "$dataset_path" ]; then
        dataset_name=$(basename "$dataset_path")
        
        # Calculate Load for this Dataset
        sbatch_files=("$dataset_path"/submit_*.sbatch)
        num_scripts=${#sbatch_files[@]}
        total_tasks=$(( num_scripts * 25 ))

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
                for f in "${sbatch_files[@]}"; do sbatch "$f" > /dev/null; done
                sleep 5
                break 
            else
                echo "   ⏳ Queue Full ($current_jobs/$MAX_JOBS). Need $total_tasks. Sleeping ${CHECK_INTERVAL}s..."
                sleep $CHECK_INTERVAL
            fi
        done
    fi
done
echo "--------------------------------------------------------"
echo "🎉 All Datasets Submitted! Exiting."
echo "--------------------------------------------------------"
