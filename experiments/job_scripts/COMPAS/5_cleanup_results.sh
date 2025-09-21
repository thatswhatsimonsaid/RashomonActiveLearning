#!/bin/bash
# This script cleans up ONLY the raw result files for the COMPAS dataset, leaving the 'aggregated' folder intact.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( cd -- "${SCRIPT_DIR}/../../../" &> /dev/null && pwd )
DATASET_NAME=$(basename "$SCRIPT_DIR")
DATASET_RESULTS_DIR="${PROJECT_ROOT}/results/${DATASET_NAME}"

echo "Cleaning up raw result files for dataset: ${DATASET_NAME} (leaving 'aggregated' folder)"
echo "  -> Targeting directory: ${DATASET_RESULTS_DIR}"

# --- Safety Checks ---
if [ -z "$DATASET_RESULTS_DIR" ] || [ "$DATASET_RESULTS_DIR" == "$PROJECT_ROOT" ] || [ "$DATASET_RESULTS_DIR" == "/" ]; then
    echo "     [SAFETY] Path is empty or points to a critical directory. Aborting."
    exit 1
fi
if [[ "$DATASET_RESULTS_DIR" != *"/results/"* ]]; then
    echo "     [SAFETY] Path does not appear to be a valid results directory. Aborting."
    exit 1
fi

# --- Deletion Logic ---
if [ -d "$DATASET_RESULTS_DIR" ]; then
    echo "     Directory found. Starting multi-step cleanup..."
    
    # Step 1: Delete all .pkl files inside the method subdirectories (M1, M2, etc.)
    echo "       - Deleting raw .pkl result files..."
    find "$DATASET_RESULTS_DIR" -type f -name "*.pkl" -delete
    
    # Step 2: Delete the now-empty method subdirectories (M1, M2, etc.)
    # This will not affect the 'aggregated' folder as it is not empty.
    echo "       - Deleting empty M* subdirectories..."
    find "$DATASET_RESULTS_DIR" -mindepth 1 -type d -empty -delete

else
    echo "     Result directory not found. Nothing to do."
fi
echo "Result cleanup complete."
