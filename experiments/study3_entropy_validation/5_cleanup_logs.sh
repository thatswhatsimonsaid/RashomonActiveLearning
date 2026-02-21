#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOWS_DIR="$SCRIPT_DIR/job_workflows"

echo "=== Cleaning up job workflows ==="
echo "This will delete the entire directory: $WORKFLOWS_DIR"
echo "(This includes all .sbatch files, helper scripts, and SLURM logs)"
echo ""
read -p "Are you sure you want to continue? (y/n): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Cleanup cancelled."
    exit 0
fi

if [ -d "$WORKFLOWS_DIR" ]; then
    rm -rf "$WORKFLOWS_DIR"
    echo "✓ Job workflows directory deleted"
else
    echo "Job workflows directory not found: $WORKFLOWS_DIR"
fi

echo ""
echo "=== Job workflows cleanup complete ==="