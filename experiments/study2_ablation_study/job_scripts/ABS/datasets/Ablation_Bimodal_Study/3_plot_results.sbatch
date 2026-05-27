#!/bin/bash
#SBATCH --job-name=Plot_Ablation_Bimodal_Study
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --output=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study2_ablation_study/slurm_logs/ABS/Ablation_Bimodal_Study/plot.out
#SBATCH --error=/mnt/beegfs/homes/simondn/RashomonActiveLearning/experiments/study2_ablation_study/slurm_logs/ABS/Ablation_Bimodal_Study/plot.err
cd /mnt/beegfs/homes/simondn/RashomonActiveLearning
module load Python/3.10.8-GCCcore-12.2.0
source .RAL_CL/bin/activate
export PYTHONPATH=$PYTHONPATH:.
python src/utils/plot_results_ABLATION.py --dataset "Ablation_Bimodal_Study" --study_dir "study2_ablation_study/ABS"
