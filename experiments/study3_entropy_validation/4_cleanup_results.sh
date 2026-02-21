#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RAW_RESULTS_DIR="$PROJECT_ROOT/results/study3_entropy_validation/raw"

echo "=== Cleaning up raw .pkl files ==="
echo "This will delete all files in: $RAW_RESULTS_DIR"
echo ""
read -p "Are you sure you want to continue? (y/n): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Cleanup cancelled."
    exit 0
fi

if [ -d "$RAW_RESULTS_DIR" ]; then
    rm -rf "$RAW_RESULTS_DIR"
    echo "✓ All raw results deleted"
else
    echo "Raw results directory not found: $RAW_RESULTS_DIR"
fi

echo ""
echo "=== Raw results cleanup complete ==="