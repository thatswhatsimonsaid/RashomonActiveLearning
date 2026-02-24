#!/bin/bash
#SBATCH --job-name=fico_M4
#SBATCH --partition=short
#SBATCH --array=0-24
#SBATCH --output=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study1_active_learning/slurm_logs/tree_predictor/fico/out/M4_S%a.out
#SBATCH --error=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study1_active_learning/slurm_logs/tree_predictor/fico/error/M4_S%a.err
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

echo "Running fico_M4 | Seed (Task ID): $SLURM_ARRAY_TASK_ID"
echo "Study: tree_predictor"

echo "Sleeping to stagger start times..."
sleep $((RANDOM % 60 + 1))


/mnt/beegfs/homes/simondn/RashomonActiveLearning/.RAL_CL/bin/python src/utils/run_experiment.py \
    --dataset fico \
    --selector_model RandomForest \
    --predictor_model PySORTDWrapper \
    --selector QBC \
    --seed $SLURM_ARRAY_TASK_ID \
    --method_number 4 \
    --rashomon_threshold 0.0 \
    --study_dir tree_predictor \
    n_estimators=100 use_unique_trees=False time_limit=30 max_features=1.0

