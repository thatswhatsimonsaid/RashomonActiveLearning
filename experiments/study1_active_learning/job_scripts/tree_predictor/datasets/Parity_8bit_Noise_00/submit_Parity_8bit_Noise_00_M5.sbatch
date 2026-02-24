#!/bin/bash
#SBATCH --job-name=Parity_8bit_Noise_00_M5
#SBATCH --partition=short
#SBATCH --array=0-24
#SBATCH --output=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study1_active_learning/slurm_logs/tree_predictor/Parity_8bit_Noise_00/out/M5_S%a.out
#SBATCH --error=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study1_active_learning/slurm_logs/tree_predictor/Parity_8bit_Noise_00/error/M5_S%a.err
#SBATCH --time=11:59:00
#SBATCH --mem-per-cpu=30G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd /mnt/beegfs/homes/simondn/RashomonActiveLearning

# --- ENVIRONMENT SETUP ---
module load Python/3.10.8-GCCcore-12.2.0
source .RAL_CL/bin/activate
export PYTHONPATH=$PYTHONPATH:.
# -------------------------

export PYTHONDONTWRITEBYTECODE=1

echo "Running Parity_8bit_Noise_00_M5 | Seed (Task ID): $SLURM_ARRAY_TASK_ID"
echo "Study: tree_predictor"

echo "Sleeping to stagger start times..."
sleep $((RANDOM % 60 + 1))


/mnt/beegfs/homes/simondn/RashomonActiveLearning/.RAL_CL/bin/python src/utils/run_experiment.py \
    --dataset Parity_8bit_Noise_00 \
    --selector_model PySORTDWrapper \
    --predictor_model PySORTDWrapper \
    --selector QBC \
    --seed $SLURM_ARRAY_TASK_ID \
    --method_number 5 \
    --rashomon_threshold 0.0 \
    --study_dir tree_predictor \
    max_depth=3 regularization=0.001 time_limit=30 max_num_trees=10000 beta=0.0

