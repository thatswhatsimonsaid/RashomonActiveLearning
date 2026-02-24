##!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath "$SCRIPT_DIR/../../../../")

echo "Generating standalone legend..."
python "$PROJECT_ROOT/src/utils/plot_results.py" --legend-only --study_dir "study1_active_learning/tree_predictor"

echo "Generating AUC Heatmaps for multiple budgets..."
for budget in 0.5 0.6 0.7 0.8 0.9 1.0; do
    echo "  > Budget Fraction: $budget"
    python "$PROJECT_ROOT/src/utils/generate_auc_heatmaps.py" --budget_fraction "$budget"
done

echo "Generating Label Efficiency Plots..."
python "$PROJECT_ROOT/src/utils/generate_label_efficiency.py"

echo "Generating Runtime Comparison Table..."
python "$PROJECT_ROOT/src/utils/generate_runtime_table.py"

echo "Collecting all plots into final directory..."
python "$PROJECT_ROOT/src/utils/collect_plots.py"

echo "Done! All figures and tables generated."
