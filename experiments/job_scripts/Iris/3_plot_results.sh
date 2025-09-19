#!/bin/bash
# This script generates plots for the Iris dataset.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( cd -- "${SCRIPT_DIR}/../../../" &> /dev/null && pwd )
DATASET_NAME=$(basename "$SCRIPT_DIR")
PYTHON_EXEC="${PROJECT_ROOT}/.RAL/bin/python"

echo "Generating plots for dataset: ${DATASET_NAME}"
cd "${PROJECT_ROOT}"
"${PYTHON_EXEC}" -m src.utils.plot_results --dataset "${DATASET_NAME}"
echo "Plotting complete for ${DATASET_NAME}. Images are in results/images/${DATASET_NAME}/"
