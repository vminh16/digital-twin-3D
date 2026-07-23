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
BACKEND_ROOT="${BTS_BACKEND_ROOT:-${REPO_ROOT}/runs/phase4/backend_qualification}"
EXPERIMENT_ROOT="${BTS_EXPERIMENT_ROOT:-${REPO_ROOT}/runs/scene_opt_v1}"
BTS_SCENES_ROOT="${BTS_SCENES_ROOT:-${REPO_ROOT}/data/bts_scenes}"
BTS_MANIFESTS_ROOT="${BTS_MANIFESTS_ROOT:-${REPO_ROOT}/runs/manifests}"
AUX_SCENES_ROOT="${AUX_SCENES_ROOT:-${REPO_ROOT}/data/auxiliary}"
AUX_MANIFESTS_ROOT="${AUX_MANIFESTS_ROOT:-${REPO_ROOT}/runs/manifests_auxiliary}"
DEFAULT_SCENES=(HCM0539 HCM0421 HCM0644 chair bonsai HCM0674 HCM0540)

if [[ "$#" -gt 0 ]]; then
    SCENE_IDS=("$@")
else
    SCENE_IDS=("${DEFAULT_SCENES[@]}")
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${EXPERIMENT_ROOT}"
COMMAND_LOG="${EXPERIMENT_ROOT}/deployment_commands.log"
GIT_COMMIT="$(git rev-parse HEAD)"
printf '# %s commit=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${GIT_COMMIT}" \
    >> "${COMMAND_LOG}"

invoke_recorded() {
    printf '%q ' "$@" >> "${COMMAND_LOG}"
    printf '\n' >> "${COMMAND_LOG}"
    "$@"
}

needs_bts=0
needs_aux=0
for scene_id in "${SCENE_IDS[@]}"; do
    case "${scene_id}" in
        HCM0539|HCM0421|HCM0644|HCM0674|HCM0540)
            needs_bts=1
            ;;
        chair|bonsai)
            needs_aux=1
            ;;
        *)
            echo "ERROR: unsupported Stage A scene: ${scene_id}" >&2
            exit 2
            ;;
    esac
done

if [[ "${needs_bts}" -eq 1 ]]; then
    invoke_recorded "${PYTHON_BIN}" \
        src/bts_nvs/data/prepare_phase4_artifacts.py \
        --scenes_root "${BTS_SCENES_ROOT}" \
        --manifests_root "${BTS_MANIFESTS_ROOT}" \
        --expected_scenes 18 \
        --require_expected
fi
if [[ "${needs_aux}" -eq 1 ]]; then
    invoke_recorded "${PYTHON_BIN}" \
        src/bts_nvs/data/prepare_phase4_artifacts.py \
        --scenes_root "${AUX_SCENES_ROOT}" \
        --manifests_root "${AUX_MANIFESTS_ROOT}" \
        --expected_scenes 2 \
        --require_expected
fi

for scene_id in "${SCENE_IDS[@]}"; do
    case "${scene_id}" in
        chair|bonsai)
            scenes_root="${AUX_SCENES_ROOT}"
            manifests_root="${AUX_MANIFESTS_ROOT}"
            ;;
        *)
            scenes_root="${BTS_SCENES_ROOT}"
            manifests_root="${BTS_MANIFESTS_ROOT}"
            ;;
    esac

    output_dir="${EXPERIMENT_ROOT}/reference/${scene_id}"
    action="run"
    if [[ -d "${output_dir}" ]] \
        && [[ -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        action="validate"
        echo "Validating existing Stage A reference: ${scene_id}"
    else
        echo "Running fresh Stage A reference: ${scene_id}"
    fi

    invoke_recorded "${PYTHON_BIN}" \
        -m bts_nvs.experiments.run_experiment "${action}" \
        --repo-root "${REPO_ROOT}" \
        --scenes-root "${scenes_root}" \
        --manifests-root "${manifests_root}" \
        --backend-root "${BACKEND_ROOT}" \
        --experiment-root "${EXPERIMENT_ROOT}" \
        --stage reference \
        --scene-id "${scene_id}" \
        --candidate-id B0-reference \
        --stop-step 7000
done

echo "Stage A references completed or validated: ${SCENE_IDS[*]}"
echo "Deployment command log: ${COMMAND_LOG}"
