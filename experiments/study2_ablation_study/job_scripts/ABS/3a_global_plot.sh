#!/bin/bash
FULL_STUDY_PATH="study2_ablation_study/ABS"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT="/mnt/beegfs/homes/simondn/RashomonActiveLearning"
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
