#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOWS_DIR="$SCRIPT_DIR/job_workflows"

echo "=== Aggregating results for all thresholds ==="

for thresh_dir in "$WORKFLOWS_DIR"/thresh_*/ ; do
    if [ -d "$thresh_dir" ]; then
        thresh_name=$(basename "$thresh_dir")
        echo "Processing: $thresh_name"
        
        cd "$thresh_dir"
        ./2_aggregate_results.sh
        cd - > /dev/null
        
        echo "  ✓ Results aggregated for $thresh_name"
        echo ""
    fi
done

echo "=== All aggregations complete ==="