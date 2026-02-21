#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOWS_DIR="$SCRIPT_DIR/job_workflows"

echo "=== Submitting SLURM jobs for all thresholds ==="

for thresh_dir in "$WORKFLOWS_DIR"/thresh_*/ ; do
    if [ -d "$thresh_dir" ]; then
        thresh_name=$(basename "$thresh_dir")
        echo "Processing: $thresh_name"
        
        cd "$thresh_dir"
        ./1_run_all.sh
        cd - > /dev/null
        
        echo "  ✓ Jobs submitted for $thresh_name"
    fi
done

echo ""
echo "=== All job submissions complete ==="
echo "Monitor jobs with: squeue -u \$USER"