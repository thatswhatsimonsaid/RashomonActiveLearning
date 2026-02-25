#!/bin/bash
#SBATCH --job-name=heatmap_bubble
#SBATCH --partition=short
#SBATCH --array=0-119
#SBATCH --time=00:30:00  
#SBATCH --mem-per-cpu=2G
#SBATCH --output=/mnt/beegfs/homes/simondn/RashomonActiveLearning/results/study2_visual_analysis/1_heat_bubble/logs/out/task_%a.out
#SBATCH --error=/mnt/beegfs/homes/simondn/RashomonActiveLearning/results/study2_visual_analysis/1_heat_bubble/logs/error/task_%a.err

# 1. Define Root (Using the verified mount path)
PROJECT_ROOT="/mnt/beegfs/homes/simondn/RashomonActiveLearning"
SCRIPT_PATH="${PROJECT_ROOT}/experiments/study2_visual_analysis/1_heat_bubble/heatmap_task.py"
TASK_ID=$SLURM_ARRAY_TASK_ID

# 2. Make log directories
mkdir -p ${PROJECT_ROOT}/results/study2_visual_analysis/1_heat_bubble/logs/out
mkdir -p ${PROJECT_ROOT}/results/study2_visual_analysis/1_heat_bubble/logs/error

# 3. Change Directory
cd $PROJECT_ROOT

# --- 4. ENVIRONMENT SETUP (The Verified Working Config) ---
module load Python/3.10.8-GCCcore-12.2.0
source .RAL_CL/bin/activate
export PYTHONPATH=$PYTHONPATH:.
# --------------------------------------------------

echo "Project Root set to: $PROJECT_ROOT"
echo "Running Heatmap Task $TASK_ID with .RAL_CL environment"

# 5. Run the Python task
python $SCRIPT_PATH $TASK_ID