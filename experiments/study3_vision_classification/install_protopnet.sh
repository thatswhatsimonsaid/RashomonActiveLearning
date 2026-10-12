#!/bin/bash
# Install proto-rset into .RAL_VISION and register it on that environment's path.
# Safe to rerun. The tree environment .RAL_CL is left alone.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

git submodule update --init third_party/proto-rset

if [[ ! -d .RAL_VISION ]]; then
  python3 -m venv .RAL_VISION
fi
# shellcheck disable=SC1091
source .RAL_VISION/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-vision.txt \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple

python - << 'PY'
import pathlib
import sysconfig

repo = pathlib.Path.cwd().resolve()
site = pathlib.Path(sysconfig.get_paths()["purelib"])
site.mkdir(parents=True, exist_ok=True)
(site / "proto-rset.pth").write_text(
    f"{repo / 'third_party' / 'proto-rset'}\n{repo}\n"
)
print(f"Registered proto-rset at {site / 'proto-rset.pth'}")
PY

python -m src.vision.check_protopnet
