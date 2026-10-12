#!/bin/bash
# Proto-rset's own ProtoPNet check, from test/test_end_to_end.py:
#   python -u -m protopnet train-vanilla-cos --verify --dataset=cifar10 --backbone=squeezenet1_0
# --verify runs one warm epoch, one joint epoch, one prototype-projection epoch,
# and one last-layer epoch. CIFAR-10 is downloaded on first use.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d "${REPO_ROOT}/third_party/proto-rset/protopnet" ]]; then
  echo "proto-rset is missing. From the repo root, run: git submodule update --init third_party/proto-rset" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/third_party/proto-rset${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE="${WANDB_MODE:-dryrun}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export CIFAR10_DIR="${CIFAR10_DIR:-${REPO_ROOT}/src/data/CIFAR10}"
mkdir -p "$CIFAR10_DIR" "${REPO_ROOT}/experiments/study3_vision_classification/logs"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "ProtoPNet verify"
echo "  python: $(command -v "$PYTHON_BIN")"
echo "  CIFAR10_DIR: ${CIFAR10_DIR}"
echo "  WANDB_MODE: ${WANDB_MODE}"

# The proto-rset CLI default for --dataset-dir is the string "None", which
# writes CIFAR-10 into a directory with that name. Pass an explicit directory.
"$PYTHON_BIN" -u -m protopnet train-vanilla-cos \
  --verify \
  --dataset=cifar10 \
  --dataset-dir="$CIFAR10_DIR" \
  --backbone=squeezenet1_0
