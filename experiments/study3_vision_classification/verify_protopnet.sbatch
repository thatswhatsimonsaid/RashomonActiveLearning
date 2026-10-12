#!/bin/bash
#SBATCH --job-name=protopnet_verify
#SBATCH --partition=short
#SBATCH --output=experiments/study3_vision_classification/protopnet_verify_%j.out
#SBATCH --error=experiments/study3_vision_classification/protopnet_verify_%j.err
#SBATCH --time=11:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=simondn@uw.edu

# Cluster check that a cosine ProtoPNet trains through warm, joint, project, and last-layer.
# Submit from the repo root: sbatch experiments/study3_vision_classification/verify_protopnet.sbatch

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
module load Python/3.10.8-GCCcore-12.2.0

bash experiments/study3_vision_classification/install_protopnet.sh
# shellcheck disable=SC1091
source .RAL_VISION/bin/activate

bash experiments/study3_vision_classification/run_protopnet_verify.sh
