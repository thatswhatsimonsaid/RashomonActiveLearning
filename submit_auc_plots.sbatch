#!/bin/bash

# --- SLURM Configuration ---
#SBATCH --job-name=generate_all_auc_plots
#SBATCH --partition=largemem             # Using a partition from your master_config.py
#SBATCH --time=11:59:00               # 2 hours should be sufficient for several datasets
#SBATCH --mem-per-cpu=100GB          # 16GB memory, as TreeFarms can be memory-intensive    
#SBATCH --mail-type=ALL               # Email notifications for start, end, and failure
#SBATCH --mail-user=simondn@uw.edu    # Using the email from your config
#SBATCH --output=logs/auc_plots_%j.out # Standard output log file
#SBATCH --error=logs/auc_plots_%j.err  # Standard error log file

# --- Job Steps ---

# Create a logs directory if it doesn't exist
mkdir -p logs

echo "--- Setting up job environment ---"
# Navigate to the directory where the job was submitted from
# This ensures all relative paths in the python script work correctly
cd $SLURM_SUBMIT_DIR

# Activate your Python virtual environment
source .RAL/bin/activate

echo "--- Starting AUC Plot Generation for All Datasets ---"

# Define the directory where your processed datasets are located
DATA_DIR="src/data/processed"

# Loop through each .pkl file in the data directory
for pkl_file in "$DATA_DIR"/*.pkl; do
    # Extract the dataset name from the file path (e.g., "MONK1" from ".../MONK1.pkl")
    DATASET_NAME=$(basename "$pkl_file" .pkl)
    
    echo "" # Add a blank line for readability in the log
    echo "Processing dataset: $DATASET_NAME"
    
    # Define the output file path for the plot, specific to this dataset
    OUTPUT_FILE="results/images/${DATASET_NAME}_auc_comparison.png"
    
    # Run the Python script as a module, passing the current dataset name and output path
    # Using python -m is the recommended way to handle the 'src' module path issue
    python -m experiments.generate_auc_plot \
        --dataset "$DATASET_NAME" \
        --output_file "$OUTPUT_FILE" \
        --seed 42
        
    echo "Plot for $DATASET_NAME saved to $OUTPUT_FILE"
done

echo ""
echo "--- All datasets processed successfully! ---"
