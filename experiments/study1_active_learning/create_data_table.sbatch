#!/bin/bash
# --- Slurm Configuration ---
#SBATCH --job-name=AL_TableGen
#SBATCH --output=slurm_logs/table_%A_%a.out
#SBATCH --error=slurm_logs/table_%A_%a.err
#SBATCH --array=0-32 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=02:00:00
#SBATCH --partition=short

# --- Environment Setup ---
cd /mnt/beegfs/homes/simondn/RashomonActiveLearning
mkdir -p "logs"
module load Python/3.10.8-GCCcore-12.2.0
source .RAL_CL/bin/activate
export PYTHONPATH=$PYTHONPATH:.

# --- Dataset Mapping ---
# Exact keys from your src/data folder and generate_dataset_table.py
DATASETS=(
    "anneal" "bank_marketing" "banknote_authentication" "bar-7" "biodeg" 
    "breast_cancer_wisconsin" "cheap_restaurant" "coffee_house" "expensive_restaurant" 
    "haberman" "hepatitis" "hypothyroid" "lymph" "monk1" "monk2" "monk3" 
    "primary-tumor" "tic-tac-toe" "vote" "yeast" 
    "Synthetic_XOR_Baseline" "Synthetic_XOR_Alpha_25" "Synthetic_XOR_Alpha_50" 
    "Synthetic_XOR_Alpha_75" "Synthetic_XOR_Alpha_100" "Synthetic_XOR_Phi_05" 
    "Synthetic_XOR_Phi_10" "Synthetic_XOR_Phi_25" "Synthetic_XOR_Phi_45"
    "Parity_8bit_Noise_00" "Parity_8bit_Noise_06" "Parity_8bit_Noise_16" "Parity_8bit_Noise_26"
)

NUM_DATASETS=${#DATASETS[@]}

if [ $SLURM_ARRAY_TASK_ID -ge $NUM_DATASETS ]; then
    exit 0
fi

CURRENT_DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "📅 Started: $(date) | Dataset: $CURRENT_DS"

# --- Execute Worker ---
python src/utils/generate_dataset_table.py --dataset "$CURRENT_DS"

# --- Final Merge Trigger ---
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo "⌛ Task 0: Sleeping to allow workers to finish..."
    sleep 300
    echo "📝 Merging fragments into DatasetTable.tex..."
    python src/utils/generate_dataset_table.py --merge
fi

echo "✅ Finished processing $CURRENT_DS at: $(date)"