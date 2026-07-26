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
SCENES_ROOT="${AUX_SCENES_ROOT:-${REPO_ROOT}/data/auxiliary}"
MANIFESTS_ROOT="${AUX_MANIFESTS_ROOT:-${REPO_ROOT}/runs/manifests_auxiliary}"
BACKEND_ROOT="${BTS_BACKEND_ROOT:-${REPO_ROOT}/runs/phase4/backend_qualification}"
REFERENCE_ROOT="${BTS_B0_EXPERIMENT_ROOT:-${REPO_ROOT}/runs/scene_opt_v1}"
EXPERIMENT_ROOT="${BTS_MVP_EXPERIMENT_ROOT:-${REPO_ROOT}/runs/scene_opt_v2}"
OUTPUT_ROOT="${AUX_MVP_PRODUCTION_ROOT:-${REPO_ROOT}/runs/scene_opt_v2/production_mvp}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "$#" -eq 0 ]]; then
    set -- chair bonsai
fi

"${PYTHON_BIN}" -m bts_nvs.experiments.run_mvp_production \
    --repo_root "${REPO_ROOT}" \
    --scenes_root "${SCENES_ROOT}" \
    --manifests_root "${MANIFESTS_ROOT}" \
    --backend_root "${BACKEND_ROOT}" \
    --reference_root "${REFERENCE_ROOT}" \
    --experiment_root "${EXPERIMENT_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --python_bin "${PYTHON_BIN}" \
    "$@"
