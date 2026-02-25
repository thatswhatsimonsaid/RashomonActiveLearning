#!/bin/bash

cd ../../..
PROJECT_ROOT=$(pwd)
PYTHON_EXEC="${PROJECT_ROOT}/.RAL/bin/python"
SCRIPT_PATH="${PROJECT_ROOT}/experiments/study2_visual_analysis/2_continuous_heatmap/plot_boundary_sensitivity.py"

echo "Starting Boundary Sensitivity Plot Analysis..."
echo "Running script: $SCRIPT_PATH"

$PYTHON_EXEC $SCRIPT_PATH

echo "Boundary sensitivity analysis complete. Check the 'plots' directory."