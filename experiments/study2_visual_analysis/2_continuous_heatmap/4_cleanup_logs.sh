#!/bin/bash
echo "Setting up paths..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

### Define the specific target directory ###
LOGS_DIR="$PROJECT_ROOT/results/study2_visual_analysis/2_continuous_heatmap/logs"

### Confirmation ###
echo "=== Log File Cleanup ==="
echo ""
echo "This script will delete:"
echo "  1. All .out and .err files inside:"
echo "     $LOGS_DIR/error"
echo "     $LOGS_DIR/out"
echo "  2. The (now empty) 'error', 'out', and 'logs' directories."
echo ""
read -p "Are you sure you want to permanently delete these? (y/n): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Cleanup cancelled."
    exit 0
fi

### Execution ###
echo ""
echo "--- Starting Cleanup ---"

if [ -d "$LOGS_DIR" ]; then
    echo "Deleting .err files..."
    rm -f "$LOGS_DIR/error"/*.err
    echo "Deleting .out files..."
    rm -f "$LOGS_DIR/out"/*.out
    echo "Removing directories..."
    rmdir "$LOGS_DIR/error"
    rmdir "$LOGS_DIR/out"
    rmdir "$LOGS_DIR"
    
    echo "✓ Cleanup complete."
else
    echo "i Info: Log directory not found (already clean): $LOGS_DIR"
fi

echo ""
echo "=== Cleanup Complete ==="