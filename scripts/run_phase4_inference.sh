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
FULL_ROOT="${BTS_FULL_ROOT:-${REPO_ROOT}/runs/phase4/full_training}"
OUTPUT_ROOT="${BTS_OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
REPORT_PATH="${BTS_INFERENCE_REPORT:-${REPO_ROOT}/runs/phase4/inference_report.json}"

SKIP_PREPARE=0
FORWARDED_ARGS=()
for argument in "$@"; do
    if [[ "${argument}" == "--skip_prepare" ]]; then
        SKIP_PREPARE=1
    else
        FORWARDED_ARGS+=("${argument}")
    fi
done

cd "${REPO_ROOT}"
if [[ "${SKIP_PREPARE}" -eq 0 ]]; then
    BTS_SCENES_ROOT="${SCENES_ROOT}" \
    BTS_MANIFESTS_ROOT="${MANIFESTS_ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
        bash "${REPO_ROOT}/scripts/prepare_phase4_artifacts.sh"
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m bts_nvs.rendering.run_inference \
    --scenes_root "${SCENES_ROOT}" \
    --manifests_root "${MANIFESTS_ROOT}" \
    --backend_root "${BACKEND_ROOT}" \
    --full_root "${FULL_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --report_path "${REPORT_PATH}" \
    "${FORWARDED_ARGS[@]}"
