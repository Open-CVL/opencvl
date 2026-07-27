#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAST3R_DIR="${MAST3R_ROOT:-${REPOSITORY_ROOT}/third_party/mast3r}"
MAST3R_COMMIT="f5209afc300cec36239a7ac992263f36847bbba0"

if [[ ! -d "${MAST3R_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${MAST3R_DIR}")"
  git clone --recursive https://github.com/naver/mast3r.git "${MAST3R_DIR}"
fi

git -C "${MAST3R_DIR}" fetch origin "${MAST3R_COMMIT}"
git -C "${MAST3R_DIR}" checkout --detach "${MAST3R_COMMIT}"
git -C "${MAST3R_DIR}" submodule update --init --recursive

python -m pip install -r "${MAST3R_DIR}/requirements.txt"
python -m pip install -r "${MAST3R_DIR}/dust3r/requirements.txt"

echo "MASt3R installed at ${MAST3R_DIR}"
echo "Pinned revision: ${MAST3R_COMMIT}"
