#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "🚀 Submitting all jobs for expensive_restaurant..."
for sbatch_file in "$SCRIPT_DIR"/submit_*.sbatch; do
    # echo "  -> Submitting $sbatch_file"
    sbatch "$sbatch_file"
done
echo "Done."
