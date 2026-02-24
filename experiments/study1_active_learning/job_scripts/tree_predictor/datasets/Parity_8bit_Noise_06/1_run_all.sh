#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo "🚀 Submitting all jobs for Parity_8bit_Noise_06..."
for sbatch_file in "$SCRIPT_DIR"/submit_*.sbatch; do
    # echo "  -> Submitting $sbatch_file"
    sbatch "$sbatch_file"
done
echo "Done."
