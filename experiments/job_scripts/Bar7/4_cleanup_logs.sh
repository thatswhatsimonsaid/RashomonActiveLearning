#!/bin/bash
# This script cleans up logs and sbatch files for the Bar7 dataset.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( cd -- "${SCRIPT_DIR}/../../../" &> /dev/null && pwd )
DATASET_NAME=$(basename "$SCRIPT_DIR")
DATASET_LOG_DIR="${PROJECT_ROOT}/experiments/slurm_logs/${DATASET_NAME}"
echo "Cleaning up logs and sbatch files for dataset: ${DATASET_NAME}"
echo "  -> Deleting .sbatch files in current directory..."
find . -maxdepth 1 -type f -name "*.sbatch" -delete
echo "  -> Deleting log directory: ${DATASET_LOG_DIR}"
if [ -d "$DATASET_LOG_DIR" ]; then rm -rf "$DATASET_LOG_DIR"; echo "     Log directory deleted."; else echo "     Log directory not found."; fi
echo "  -> Deleting helper scripts..."
rm -- 1_run_all.sh 2_aggregate_results.sh 3_plot_results.sh 5_cleanup_results.sh "$0"
echo "Log cleanup complete."
