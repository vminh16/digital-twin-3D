#!/usr/bin/env bash
set -euo pipefail

SCRIPT_SOURCE="${BASH_SOURCE[0]}"
SCRIPT_BASE="."
if [[ "${SCRIPT_SOURCE}" == */* ]]; then
    SCRIPT_BASE="${SCRIPT_SOURCE%/*}"
fi
SCRIPT_DIR="$(cd -- "${SCRIPT_BASE}" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCENES_ROOT="${BTS_SCENES_ROOT:-${REPO_ROOT}/data/bts_scenes}"
MANIFESTS_ROOT="${BTS_MANIFESTS_ROOT:-${REPO_ROOT}/runs/manifests}"

if [[ ! -d "${SCENES_ROOT}" ]]; then
    echo "ERROR: BTS scene root does not exist: ${SCENES_ROOT}" >&2
    exit 1
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" src/bts_nvs/data/prepare_phase4_artifacts.py \
    --scenes_root "${SCENES_ROOT}" \
    --manifests_root "${MANIFESTS_ROOT}" \
    --expected_scenes 18 \
    --require_expected
