#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Generating final summary plots for RMSE and MAE ==="

cd "$PROJECT_ROOT"
source .RAL/bin/activate

for METRIC in rmse mae; do
    echo ""
    echo "--- Generating plot for $METRIC ---"
    
    python -m experiments.study3_entropy_validation.entropy_estimation_study final_plot --metric "$METRIC"

    echo "✓ Plot saved to: $PROJECT_ROOT/results/study3_entropy_validation/images/final_summary_by_threshold_${METRIC}.png"
done

echo ""
echo "=== All plots generated successfully! ==="