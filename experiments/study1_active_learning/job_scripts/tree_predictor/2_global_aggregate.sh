#!/bin/bash
STUDY_DIR="study1_active_learning/tree_predictor"
REQUIRED_COUNT=25
METHODS=("M1" "M2" "M3" "M4" "M5" "M6" "M7")

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")
RESULTS_ROOT="$PROJECT_ROOT/results/$STUDY_DIR"
JOB_SCRIPTS_DIR="$SCRIPT_DIR/datasets"

echo "🔍 Global Smart Aggregate: Checking $STUDY_DIR"

for dataset_path in "$RESULTS_ROOT"/*; do
    if [ -d "$dataset_path" ]; then
        dataset_name=$(basename "$dataset_path")
        [[ "$dataset_name" == "aggregated" ]] && continue

        complete=true
        for m in "${METHODS[@]}"; do
            count=$(ls -1 "$dataset_path/$m"/*.pkl 2>/dev/null | wc -l)
            [[ "$count" -ne "$REQUIRED_COUNT" ]] && complete=false && break
        done

        if [ "$complete" = true ]; then
            agg_script="$JOB_SCRIPTS_DIR/$dataset_name/2_aggregate_results.sbatch"
            [ -f "$agg_script" ] && sbatch "$agg_script"
        fi
    fi
done
