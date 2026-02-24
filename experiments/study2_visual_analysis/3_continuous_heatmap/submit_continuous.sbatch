#!/bin/bash
#SBATCH --job-name=cont_heat
#SBATCH --partition=medium
#SBATCH --array=0-9
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=2000MB
#SBATCH --output=/homes/simondn/RashomonActiveLearning/results/study2_visual_analysis/3_continuous_heatmap/logs/out/task_%a.out
#SBATCH --error=/homes/simondn/RashomonActiveLearning/results/study2_visual_analysis/3_continuous_heatmap/logs/error/task_%a.err

# 1. Define Root
PROJECT_ROOT="/mnt/beegfs/homes/simondn/RashomonActiveLearning"
SCRIPT_PATH="${PROJECT_ROOT}/experiments/study2_visual_analysis/3_continuous_heatmap/continuous_task.py"
TASK_ID=$SLURM_ARRAY_TASK_ID

# 2. Make log directories
mkdir -p ${PROJECT_ROOT}/results/study2_visual_analysis/3_continuous_heatmap/logs/out
mkdir -p ${PROJECT_ROOT}/results/study2_visual_analysis/3_continuous_heatmap/logs/error

# 3. Change Directory
cd $PROJECT_ROOT

# --- 4. ENVIRONMENT SETUP (Your Working Config) ---
module load Python/3.10.8-GCCcore-12.2.0
source .RAL_CL/bin/activate
export PYTHONPATH=$PYTHONPATH:.
# --------------------------------------------------

echo "Project Root: $PROJECT_ROOT"
echo "Python Path: $(which python)"
echo "Running Task: $TASK_ID"

# 5. Run the script using the activated python
python $SCRIPT_PATH $TASK_ID