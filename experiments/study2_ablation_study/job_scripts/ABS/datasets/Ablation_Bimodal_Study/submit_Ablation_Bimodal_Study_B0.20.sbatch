#!/bin/bash
#SBATCH --job-name=Ablation_Bimodal_Study_B0.20
#SBATCH --partition=short
#SBATCH --array=9,11,19
#SBATCH --output=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study2_ablation_study/slurm_logs/ABS/Ablation_Bimodal_Study/B0.20/out/B0.20_S%a.out
#SBATCH --error=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study2_ablation_study/slurm_logs/ABS/Ablation_Bimodal_Study/B0.20/error/B0.20_S%a.err
#SBATCH --time=7:59:00
#SBATCH --mem-per-cpu=30G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd /mnt/beegfs/homes/simondn/RashomonActiveLearning
module load Python/3.10.8-GCCcore-12.2.0
source .RAL_CL/bin/activate
export PYTHONPATH=$PYTHONPATH:.
export PYTHONDONTWRITEBYTECODE=1

echo "Running Ablation_Bimodal_Study_B0.20 | Seed: $SLURM_ARRAY_TASK_ID"
echo "Target Path: results/study2_ablation_study/ABS"

sleep $((RANDOM % 30 + 1))

/mnt/beegfs/homes/simondn/RashomonActiveLearning/.RAL_CL/bin/python src/utils/run_experiment.py \
    --dataset Ablation_Bimodal_Study \
    --selector_model PySORTDWrapper \
    --predictor_model PySORTDWrapper \
    --selector QBC \
    --seed $SLURM_ARRAY_TASK_ID \
    --method_number 8 \
    --rashomon_threshold 0.2 \
    --study_dir study2_ablation_study/ABS \
    task_type=classification \
    max_depth=3 regularization=0.1 time_limit=60 max_num_trees=100000 beta=100.0

