#!/bin/bash
STUDY_DIR="study1_active_learning/tree_predictor"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
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
