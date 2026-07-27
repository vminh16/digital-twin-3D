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

if [[ "$#" -ne 2 ]]; then
    echo "Usage: $0 <chair|bonsai> <registered-candidate-id>" >&2
    exit 2
fi

SCENE_ID="$1"
CANDIDATE_ID="$2"
case "${SCENE_ID}" in
    chair|bonsai)
        ;;
    *)
        echo "ERROR: active research is limited to chair and bonsai." >&2
        exit 2
        ;;
esac

for artifact in manifest.json arrays.npz; do
    path="${MANIFESTS_ROOT}/${SCENE_ID}/${artifact}"
    if [[ ! -f "${path}" ]]; then
        echo "ERROR: required paired artifact is missing: ${path}" >&2
        exit 2
    fi
done

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m bts_nvs.data.prepare_research_artifacts \
    --scene-root "${SCENES_ROOT}/${SCENE_ID}" \
    --manifest-dir "${MANIFESTS_ROOT}/${SCENE_ID}"
if [[ ! -f "${MANIFESTS_ROOT}/${SCENE_ID}/holdout_research_v3.json" ]]; then
    echo "ERROR: targeted research holdout was not created." >&2
    exit 2
fi
mkdir -p "${EXPERIMENT_ROOT}"
COMMAND_LOG="${EXPERIMENT_ROOT}/deployment_commands.log"
printf '# %s commit=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$(git rev-parse HEAD)" >> "${COMMAND_LOG}"

OUTPUT_DIR="${EXPERIMENT_ROOT}/research/${SCENE_ID}/${CANDIDATE_ID}"
ACTION="run"
if [[ -d "${OUTPUT_DIR}" ]] \
    && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    ACTION="validate"
fi

COMMAND=(
    "${PYTHON_BIN}"
    -m bts_nvs.experiments.run_experiment
    "${ACTION}"
    --repo-root "${REPO_ROOT}"
    --scenes-root "${SCENES_ROOT}"
    --manifests-root "${MANIFESTS_ROOT}"
    --backend-root "${BACKEND_ROOT}"
    --experiment-root "${EXPERIMENT_ROOT}"
    --stage research
    --scene-id "${SCENE_ID}"
    --candidate-id "${CANDIDATE_ID}"
    --stop-step 15000
)

printf '%q ' "${COMMAND[@]}" >> "${COMMAND_LOG}"
printf '\n' >> "${COMMAND_LOG}"
"${COMMAND[@]}"

echo "Research evidence ready: ${OUTPUT_DIR}"
echo "Deployment command log: ${COMMAND_LOG}"
