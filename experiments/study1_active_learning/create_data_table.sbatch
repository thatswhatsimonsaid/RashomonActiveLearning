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
DATASETS=(
    "Synthetic_XOR_Baseline",
    "Synthetic_XOR_Phi_05",
    "Synthetic_XOR_Phi_10",
    "Synthetic_XOR_Phi_25",
    "Synthetic_XOR_Phi_45",
    "Synthetic_XOR_Alpha_50",
    "Synthetic_XOR_Alpha_25",
    "Synthetic_XOR_Alpha_75",
    "Synthetic_XOR_Alpha_100",
    "anneal",                               # Dataset 1
    "bank_marketing",                       # Dataset 2
    "bar-7",                                # Dataset 3
    "breast_cancer_wisconsin",              # Dataset 4
    "car_evaluation",                       # Dataset 5
    "cheap_restaurant",                     # Dataset 6
    "coffee_house",                         # Dataset 7
    "expensive_restaurant",                 # Dataset 8
    "haberman",                             # Dataset 9
    "hepatitis",                            # Dataset 10
    "hypothyroid",                          # Dataset 11
    "kr-vs-kp",                             # Dataset 12
    "lymph",                                # Dataset 13
    "monk1",                                # Dataset 14 (replace with FICO if finish on time)
    "monk2",                                # Dataset 15 
    "monk3",                                # Dataset 16 (replace with COMPAS if finish on time)
    "primary-tumor",                        # Dataset 17
    "tic-tac-toe",                          # Dataset 18
    "vote",                                 # Dataset 19
    "yeast"                                 # Dataset 20
)

NUM_DATASETS=${#DATASETS[@]}

if [ $SLURM_ARRAY_TASK_ID -ge $NUM_DATASETS ]; then
    exit 0
fi

CURRENT_DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}

echo "Started: $(date) | Dataset: $CURRENT_DS"

python src/utils/generate_dataset_table.py --dataset "$CURRENT_DS"

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    echo "Task 0: Sleeping to allow workers to finish..."
    sleep 600
    echo "Merging fragments into DatasetTable.tex..."
    python src/utils/generate_dataset_table.py --merge
fi

echo "Finished processing $CURRENT_DS at: $(date)"