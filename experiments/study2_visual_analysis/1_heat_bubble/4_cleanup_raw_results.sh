#!/bin/bash

echo "Setting up paths..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

### Define the specific target directory ###
RAW_DIR="$PROJECT_ROOT/results/study2_visual_analysis/1_heat_bubble/raw"

### Confirmation ###
echo "=== Raw .pkl File Cleanup ==="
echo ""
echo "This script will delete:"
echo "  1. All .pkl files inside:"
echo "     $RAW_DIR"
echo "  2. The (now empty) 'raw' directory."
echo ""
read -p "Are you sure you want to permanently delete these? (y/n): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Cleanup cancelled."
    exit 0
fi

### Execution ###
echo ""
echo "--- Starting Cleanup ---"

if [ -d "$RAW_DIR" ]; then
    echo "Deleting .pkl files..."
    rm -f "$RAW_DIR"/*.pkl
    echo "Removing 'raw' directory..."
    rmdir "$RAW_DIR"
    
    echo "✓ Cleanup complete."
else
    echo "i Info: Raw results directory not found (already clean): $RAW_DIR"
fi

echo ""
echo "=== Cleanup Complete ==="