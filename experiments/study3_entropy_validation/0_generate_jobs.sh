#!/bin/bash
# Generate all workflow scripts for the threshold sweep

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Generating workflow scripts for all thresholds ==="

cd "$SCRIPT_DIR"
python generate_sbatch_arrays.py

echo ""
echo "=== Workflow generation complete ==="
echo "Ready to run ./1_run_all.sh"
