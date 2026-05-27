#!/bin/bash
#SBATCH --job-name=Agg_Parity_8bit_Noise_06
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --output=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study2_ablation_study/slurm_logs/ABS/Parity_8bit_Noise_06/aggregate.out
#SBATCH --error=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study2_ablation_study/slurm_logs/ABS/Parity_8bit_Noise_06/aggregate.err
cd /mnt/beegfs/homes/simondn/RashomonActiveLearning
module load Python/3.10.8-GCCcore-12.2.0
source .RAL_CL/bin/activate
export PYTHONPATH=$PYTHONPATH:.
python src/utils/aggregate_results.py --dataset "Parity_8bit_Noise_06" --study_dir "study2_ablation_study/ABS"
