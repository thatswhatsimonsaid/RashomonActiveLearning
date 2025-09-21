#!/bin/bash
# This script aggregates results ONLY for the COMPAS dataset.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$( cd -- "${SCRIPT_DIR}/../../../" &> /dev/null && pwd )
DATASET_NAME=$(basename "$SCRIPT_DIR")
PYTHON_EXEC="${PROJECT_ROOT}/.RAL/bin/python"

# --- Pre-Aggregation Sanity Check ---
echo "Verifying result file counts for dataset: ${DATASET_NAME}"
ALL_FILES_PRESENT=true
EXPECTED_COUNT=50
METHOD_DIRS=$(find "${PROJECT_ROOT}/results/${DATASET_NAME}" -mindepth 1 -maxdepth 1 -type d -name "M*" 2>/dev/null | sort)

if [ -z "$METHOD_DIRS" ]; then
    echo "  [WARNING] No method result directories (M*) found. Nothing to aggregate."
    exit 0
fi

for method_dir in $METHOD_DIRS; do
    actual_count=$(find "$method_dir" -type f -name "*.pkl" | wc -l)
    method_name=$(basename "$method_dir")
    
    if [ "$actual_count" -ne "$EXPECTED_COUNT" ]; then
        echo "  [WARNING] Incomplete results for ${method_name}: Found ${actual_count} files, expected ${EXPECTED_COUNT}."
        ALL_FILES_PRESENT=false
    else
        echo "  [OK] Found ${actual_count}/${EXPECTED_COUNT} files for ${method_name}."
    fi
done

if [ "$ALL_FILES_PRESENT" = false ]; then
    echo ""
    echo "[CAUTION] Aggregating incomplete data. Results may be skewed. Check logs and re-run failed jobs."
else
    echo "All result files are present. Proceeding with aggregation."
fi
echo ""

# --- Run Aggregation ---
echo "Aggregating results for dataset: ${DATASET_NAME}"
cd "${PROJECT_ROOT}"
"${PYTHON_EXEC}" -m src.utils.aggregate_results --dataset "${DATASET_NAME}"
echo "Aggregation complete for ${DATASET_NAME}."
