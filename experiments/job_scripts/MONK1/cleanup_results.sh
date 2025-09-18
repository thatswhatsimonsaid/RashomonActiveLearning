#!/bin/bash
# This script cleans up ONLY the result .pkl files for the MONK1 dataset.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( cd -- "${SCRIPT_DIR}/../../" &> /dev/null && pwd )
DATASET_NAME=$(basename "$SCRIPT_DIR")
DATASET_RESULTS_DIR="${PROJECT_ROOT}/src/results/${DATASET_NAME}"
echo "Cleaning up result files for dataset: ${DATASET_NAME}"
echo "  -> Deleting results directory: ${DATASET_RESULTS_DIR}"
if [ -d "$DATASET_RESULTS_DIR" ]; then
    rm -rf "$DATASET_RESULTS_DIR"
    echo "     Result directory deleted."
else
    echo "     Result directory not found."
fi
echo "Result cleanup complete."
