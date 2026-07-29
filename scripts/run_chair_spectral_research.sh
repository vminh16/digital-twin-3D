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
EXPERIMENT_ROOT="${BTS_RESEARCH_EXPERIMENT_ROOT:-${REPO_ROOT}/runs/scene_opt_v3}"
CANDIDATE_ID="E8-chair-observation-scale-spectral-split-v1"
OUTPUT_DIR="${EXPERIMENT_ROOT}/research/chair/${CANDIDATE_ID}"

if [[ "${E8_STOP_STEP:-15000}" != "15000" ]]; then
    echo "ERROR: E8 is authorized only for the fresh 15000-step gate" >&2
    exit 2
fi
for artifact in manifest.json arrays.npz; do
    path="${MANIFESTS_ROOT}/chair/${artifact}"
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: required paired artifact is missing: ${path}" >&2
        exit 2
    fi
done

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m bts_nvs.data.prepare_research_artifacts \
    --scene-root "${SCENES_ROOT}/chair" \
    --manifest-dir "${MANIFESTS_ROOT}/chair"

ACTION="run"
if [[ -f "${OUTPUT_DIR}/metrics.jsonl" ]] \
    && [[ "$(wc -l < "${OUTPUT_DIR}/metrics.jsonl")" == "15000" ]] \
    && [[ -f "${OUTPUT_DIR}/experiment_report.json" ]]; then
    ACTION="validate"
elif [[ -d "${OUTPUT_DIR}" ]] \
    && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "ERROR: E8 requires a fresh or complete output directory: ${OUTPUT_DIR}" >&2
    exit 2
fi

"${PYTHON_BIN}" -m bts_nvs.experiments.run_experiment \
    "${ACTION}" \
    --repo-root "${REPO_ROOT}" \
    --scenes-root "${SCENES_ROOT}" \
    --manifests-root "${MANIFESTS_ROOT}" \
    --backend-root "${BACKEND_ROOT}" \
    --experiment-root "${EXPERIMENT_ROOT}" \
    --stage research \
    --scene-id chair \
    --candidate-id "${CANDIDATE_ID}" \
    --stop-step 15000

echo "Spectral split research evidence ready at step 15000: ${OUTPUT_DIR}"
