#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "This will clear ALL .out and .err logs for this ablation study."
read -p "Continue? (y/n): " confirm
if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
    find "$SCRIPT_DIR/datasets" -name "4_cleanup_logs.sh" -exec bash -c 'echo "y" | {}' \;
    echo "All logs cleared."
fi
