#!/bin/bash
#SBATCH --job-name=Parity_8bit_Noise_16_M1
#SBATCH --partition=short
#SBATCH --array=0-24
#SBATCH --output=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study2_ablation_study/slurm_logs/ABS/Parity_8bit_Noise_16/M1/out/M1_S%a.out
#SBATCH --error=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study2_ablation_study/slurm_logs/ABS/Parity_8bit_Noise_16/M1/error/M1_S%a.err
#SBATCH --time=2:59:00
#SBATCH --mem-per-cpu=30G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

cd /mnt/beegfs/homes/simondn/RashomonActiveLearning
module load Python/3.10.8-GCCcore-12.2.0
source .RAL_CL/bin/activate
export PYTHONPATH=$PYTHONPATH:.
export PYTHONDONTWRITEBYTECODE=1

echo "Running Parity_8bit_Noise_16_M1 | Seed: $SLURM_ARRAY_TASK_ID"
echo "Target Path: results/study2_ablation_study/ABS"

sleep $((RANDOM % 30 + 1))

/mnt/beegfs/homes/simondn/RashomonActiveLearning/.RAL_CL/bin/python src/utils/run_experiment.py \
    --dataset Parity_8bit_Noise_16 \
    --selector_model RandomForest \
    --predictor_model PySORTDWrapper \
    --selector Random \
    --seed $SLURM_ARRAY_TASK_ID \
    --method_number 1 \
    --rashomon_threshold 0.0 \
    --study_dir study2_ablation_study/ABS \
    task_type=classification \
    max_depth=3 regularization=0.1 time_limit=60 max_num_trees=100000 beta=100.0

