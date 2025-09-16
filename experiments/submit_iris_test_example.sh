#!/bin/bash
# This script generates and submits sbatch jobs for all experiments.

# --- Configuration ---
# TODO: Adjust these SLURM settings for your cluster's requirements
# I've removed the --account line as it wasn't in your last config, add it back if needed.
PARTITION="short"
MEMORY="8G"
TIME="01:00:00" # 1 hour

# Create a directory for SLURM log files
LOG_DIR="slurm_logs"
mkdir -p $LOG_DIR

# --- Define Experiments ---
declare -a experiments=(
    "--dataset Iris --model RandomForest --selector Passive n_estimators=100"
    "--dataset Iris --model GPC --selector Passive"
    "--dataset Iris --model BNN --selector Passive epochs=50"
    "--dataset Iris --model BNN --selector BALD epochs=50 n_ensemble_samples=100"
    "--dataset Iris --model GPC --selector BALD n_ensemble_samples=100"
    "--dataset Iris --model TreeFarms --selector QBC regularization=0.01 rashomon_threshold=0.05 use_unique_trees=True"
    "--dataset Iris --model TreeFarms --selector QBC regularization=0.01 rashomon_threshold=0.05 use_unique_trees=False"
    "--dataset Iris --model RandomForest --selector QBC n_estimators=100 use_unique_trees=False"
)

# --- Loop and Submit ---
for args in "${experiments[@]}"; do
    job_name=$(echo $args | tr -d ' ' | tr -d '-' | tr '=' '_')
    echo "Submitting job: ${job_name}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME}
#SBATCH --mem-per-cpu=${MEMORY}
#SBATCH --ntasks=1
#SBATCH --output=${LOG_DIR}/${job_name}_%j.out
#SBATCH --error=${LOG_DIR}/${job_name}_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

echo "----------------------------------------"
echo "Job:         \${SLURM_JOB_NAME}"
echo "Running on:  \$(hostname)"
echo "Submit dir:  \${SLURM_SUBMIT_DIR}"
echo "----------------------------------------"

# --- THIS IS THE CORRECTED PART ---
# Instead of 'source activate', use the full path to the Python executable
# in your virtual environment. \$SLURM_SUBMIT_DIR is the directory
# where you ran this submission script.

\${SLURM_SUBMIT_DIR}/RAL/bin/python -m src.utils.run_experiment ${args}

EOF
done

echo ""
echo "✅ All 8 jobs submitted."