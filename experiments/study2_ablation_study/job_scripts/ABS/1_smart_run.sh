#!/bin/bash
MAX_JOBS=1800
CHECK_INTERVAL=60
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DATASETS_DIR="$SCRIPT_DIR/datasets"

get_job_count() {
    squeue -u $USER -h -r | wc -l
}

for dataset_path in "$DATASETS_DIR"/*; do
    if [ -d "$dataset_path" ]; then
        dataset_name=$(basename "$dataset_path")
        sbatch_files=("$dataset_path"/submit_*.sbatch)
        num_scripts=${#sbatch_files[@]}
        total_tasks=$(( num_scripts * 25 ))

        echo "Checking: $dataset_name"
        while true; do
            current_jobs=$(get_job_count)
            if [ $(( current_jobs + total_tasks )) -le "$MAX_JOBS" ]; then
                for f in "${sbatch_files[@]}"; do sbatch "$f" > /dev/null; done
                break 
            else
                sleep $CHECK_INTERVAL
            fi
        done
    fi
done
