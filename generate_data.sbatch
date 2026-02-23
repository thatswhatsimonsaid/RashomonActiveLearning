#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=gen_dataset_table
#SBATCH --output=logs/gen_ds_table_%j.out
#SBATCH --error=logs/gen_ds_table_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=1-11:00:00
#SBATCH --partition=largemem 

# --- Environment Setup ---
cd /mnt/beegfs/homes/simondn/RashomonActiveLearning

# Match the environment setup from your working sbatch files
module load Python/3.10.8-GCCcore-12.2.0
source .RAL_CL/bin/activate
export PYTHONPATH=$PYTHONPATH:.
mkdir -p "logs"
echo "📅 Job started at: $(date)"
echo "🚀 Running generate_dataset_table.py"

# --- Execute Script ---
python -u src/utils/generate_dataset_table.py

echo "✅ Job finished at: $(date)"