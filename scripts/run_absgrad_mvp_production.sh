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
BACKEND_ROOT="${BTS_BACKEND_ROOT:-${REPO_ROOT}/runs/phase4/backend_qualification}"
EXPERIMENT_ROOT="${BTS_EXPERIMENT_ROOT:-${REPO_ROOT}/runs/scene_opt_v1}"
OUTPUT_ROOT="${BTS_MVP_PRODUCTION_ROOT:-${REPO_ROOT}/runs/scene_opt_v1/production_mvp}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m bts_nvs.experiments.run_mvp_production \
    --repo_root "${REPO_ROOT}" \
    --scenes_root "${SCENES_ROOT}" \
    --manifests_root "${MANIFESTS_ROOT}" \
    --backend_root "${BACKEND_ROOT}" \
    --experiment_root "${EXPERIMENT_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --python_bin "${PYTHON_BIN}" \
    "$@"
