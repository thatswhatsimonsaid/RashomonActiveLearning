#!/bin/bash
#SBATCH --job-name=heatmap_bubble
#SBATCH --partition=short
#SBATCH --array=0-119
#SBATCH --time=00:30:00  
#SBATCH --mem-per-cpu=2000
#SBATCH --output=/homes/simondn/RashomonActiveLearning/results/study2_visual_analysis/2_heat_bubble/logs/out/task_%a.out
#SBATCH --error=/homes/simondn/RashomonActiveLearning/results/study2_visual_analysis/2_heat_bubble/logs/error/task_%a.err

### Setup ###
PROJECT_ROOT="/homes/simondn/RashomonActiveLearning"
PYTHON_EXEC="${PROJECT_ROOT}/.RAL/bin/python"
SCRIPT_PATH="${PROJECT_ROOT}/experiments/study2_visual_analysis/2_heat_bubble/heatmap_task.py"
TASK_ID=$SLURM_ARRAY_TASK_ID

### Create Log Dirs ###
mkdir -p ${PROJECT_ROOT}/results/study2_visual_analysis/2_heat_bubble/logs/out
mkdir -p ${PROJECT_ROOT}/results/study2_visual_analysis/2_heat_bubble/logs/error
cd $PROJECT_ROOT

echo "Project Root set to: $PROJECT_ROOT"
echo "Running Heatmap Task $TASK_ID"

### Run the Python task ###
$PYTHON_EXEC $SCRIPT_PATH $TASK_ID
