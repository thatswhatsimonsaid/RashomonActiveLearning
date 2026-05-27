#!/bin/bash
#SBATCH --job-name=Plot_Parity_8bit_Noise_16
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --output=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study2_ablation_study/slurm_logs/ABS/Parity_8bit_Noise_16/plot.out
#SBATCH --error=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study2_ablation_study/slurm_logs/ABS/Parity_8bit_Noise_16/plot.err
cd /mnt/beegfs/homes/simondn/RashomonActiveLearning
module load Python/3.10.8-GCCcore-12.2.0
source .RAL_CL/bin/activate
export PYTHONPATH=$PYTHONPATH:.
python src/utils/plot_results_ABLATION.py --dataset "Parity_8bit_Noise_16" --study_dir "study2_ablation_study/ABS"
