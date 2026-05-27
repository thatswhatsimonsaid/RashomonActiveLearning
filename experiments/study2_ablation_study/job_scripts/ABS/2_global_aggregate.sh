#!/bin/bash
FULL_STUDY_PATH="study2_ablation_study/ABS"
REQUIRED_COUNT=25

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT="/mnt/beegfs/homes/simondn/RashomonActiveLearning"
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
