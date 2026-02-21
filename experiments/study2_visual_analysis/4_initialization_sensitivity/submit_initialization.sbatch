#!/bin/bash
#SBATCH --job-name=init_sensitivity
#SBATCH --partition=short
#SBATCH --array=0-199
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=2000
#SBATCH --output=/homes/simondn/RashomonActiveLearning/results/study2_visual_analysis/4_initialization_sensitivity/logs/out/task_%a.out
#SBATCH --error=/homes/simondn/RashomonActiveLearning/results/study2_visual_analysis/4_initialization_sensitivity/logs/error/task_%a.err


### Setup Paths ###
PROJECT_ROOT="/homes/simondn/RashomonActiveLearning"
PYTHON_EXEC="${PROJECT_ROOT}/.RAL/bin/python"
SCRIPT_PATH="${PROJECT_ROOT}/experiments/study2_visual_analysis/4_initialization_sensitivity/initialization_task.py"

### Create Log Directories ###
mkdir -p $(dirname "$SCRIPT_PATH")/logs/out
mkdir -p $(dirname "$SCRIPT_PATH")/logs/error

### Execution ###
cd $PROJECT_ROOT
echo "=== Starting Initialization Sensitivity Task ==="
echo "Task ID: $SLURM_ARRAY_TASK_ID"

# 0-99:   Scenario A (Corners)
# 100-199: Scenario B (Boundary)
# 200-299: Scenario C (Margin 0.4)
# 300-399: Scenario C (Margin 0.2)
# 400-499: Scenario C (Margin 0.05)

$PYTHON_EXEC $SCRIPT_PATH $SLURM_ARRAY_TASK_ID

echo "=== Task Complete ==="
