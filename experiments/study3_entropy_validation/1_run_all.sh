#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOWS_DIR="$SCRIPT_DIR/job_workflows"

echo "=== Submitting SLURM jobs for all Rashomon Multipliers ==="

for mult_dir in "$WORKFLOWS_DIR"/mult_*/ ; do
    if [ -d "$mult_dir" ]; then
        mult_name=$(basename "$mult_dir")
        echo "Processing: $mult_name"
        
        cd "$mult_dir"
        ./1_run_all.sh
        cd - > /dev/null
        
        echo "  ✓ Jobs submitted for $mult_name"
    fi
done

echo ""
echo "=== All job submissions complete ==="
echo "Monitor jobs with: squeue -u \$USER"