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

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m bts_nvs.submission.prepare_jpeg \
    --source_root "${BTS_RENDER_ROOT:-${REPO_ROOT}/outputs}" \
    --output_root "${BTS_SUBMISSION_ROOT:-${REPO_ROOT}/submission_outputs}" \
    --scenes_root "${BTS_SCENES_ROOT:-${REPO_ROOT}/data/bts_scenes}" \
    --manifests_root "${BTS_MANIFESTS_ROOT:-${REPO_ROOT}/runs/manifests}" \
    --report_path "${BTS_JPEG_REPORT:-${REPO_ROOT}/runs/submission/jpeg_report.json}" \
    --quality "${BTS_JPEG_QUALITY:-99}" \
    --max_bytes "${BTS_SUBMISSION_MAX_BYTES:-350000000}" \
    "$@"
