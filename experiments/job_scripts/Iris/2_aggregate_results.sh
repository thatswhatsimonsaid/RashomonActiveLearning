#!/bin/bash
# This script aggregates results ONLY for the Iris dataset.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( cd -- "${SCRIPT_DIR}/../../../" &> /dev/null && pwd )
DATASET_NAME=$(basename "$SCRIPT_DIR")
PYTHON_EXEC="${PROJECT_ROOT}/.RAL/bin/python"

echo "Aggregating results for dataset: ${DATASET_NAME}"

# Change to the project root to ensure python -m works
cd "${PROJECT_ROOT}"

# Run the aggregation script with the correct python, targeting only this dataset
"${PYTHON_EXEC}" -m src.utils.aggregate_results --dataset "${DATASET_NAME}"

echo "Aggregation complete for ${DATASET_NAME}."
