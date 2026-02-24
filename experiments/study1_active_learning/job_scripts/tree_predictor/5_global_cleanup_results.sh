#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "WARNING: This will delete ALL raw .pkl seeds (M1-M10)."
read -p "Are you absolutely sure? (y/n): " confirm
if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
    find "$SCRIPT_DIR/datasets" -name "5_delete_raw_results.sh" -exec bash -c 'echo "y" | {}' \;
    echo "All raw results cleared."
fi
