#!/usr/bin/env bash
# Clone the pinned RashomonDR and coMDS checkouts and install them in place.
# Does not modify either upstream tree.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR="${ROOT}/vendor"
LOCK="${ROOT}/VENDOR.lock"
PY="${PYTHON:-python3}"

mkdir -p "${VENDOR}"

while read -r name url sha; do
  [[ -z "${name}" || "${name}" == \#* ]] && continue
  dest="${VENDOR}/${name}"
  if [[ ! -d "${dest}/.git" ]]; then
    git clone "${url}" "${dest}"
  fi
  git -C "${dest}" fetch --depth 1 origin "${sha}"
  git -C "${dest}" checkout --detach "${sha}"
  if [[ -n "$(git -C "${dest}" status --porcelain)" ]]; then
    echo "Refusing to continue: ${dest} has local modifications." >&2
    exit 1
  fi
done < "${LOCK}"

"${PY}" -m pip install --no-deps -e "${VENDOR}/RashomonDR/code/contrastive-ne"
"${PY}" -m pip install --no-deps -e "${VENDOR}/RashomonDR/code/paramrepulsor"

R_LIB="${R_LIBS_USER:-${HOME}/R/x86_64-pc-linux-gnu-library/4.3}"
mkdir -p "${R_LIB}"
Rscript -e 'if (!requireNamespace("smacof", quietly=TRUE)) install.packages("smacof", repos="https://cloud.r-project.org", lib=Sys.getenv("R_LIBS_USER"))'
R CMD INSTALL -l "${R_LIB}" "${VENDOR}/coMDS/coMDS"
echo "Vendor checkouts installed."
