#!/bin/bash
#SBATCH --job-name=monk2_U0.05
#SBATCH --partition=short
#SBATCH --array=0,1,7,13,14,20
#SBATCH --output=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study2_ablation_study/slurm_logs/ABS/monk2/U0.05/out/U0.05_S%a.out
#SBATCH --error=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study2_ablation_study/slurm_logs/ABS/monk2/U0.05/error/U0.05_S%a.err
#SBATCH --time=7:59:00
#SBATCH --mem-per-cpu=30G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd /mnt/beegfs/homes/simondn/RashomonActiveLearning
module load Python/3.10.8-GCCcore-12.2.0
source .RAL_CL/bin/activate
export PYTHONPATH=$PYTHONPATH:.
export PYTHONDONTWRITEBYTECODE=1

echo "Running monk2_U0.05 | Seed: $SLURM_ARRAY_TASK_ID"
echo "Target Path: results/study2_ablation_study/ABS"

sleep $((RANDOM % 30 + 1))

/mnt/beegfs/homes/simondn/RashomonActiveLearning/.RAL_CL/bin/python src/utils/run_experiment.py \
    --dataset monk2 \
    --selector_model PySORTDWrapper \
    --predictor_model PySORTDWrapper \
    --selector QBC \
    --seed $SLURM_ARRAY_TASK_ID \
    --method_number 3 \
    --rashomon_threshold 0.05 \
    --study_dir study2_ablation_study/ABS \
    task_type=classification \
    max_depth=3 regularization=0.1 time_limit=60 max_num_trees=100000 beta=0.0

