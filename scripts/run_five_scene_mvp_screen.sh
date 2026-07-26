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
B0_ROOT="${BTS_B0_EXPERIMENT_ROOT:-${REPO_ROOT}/runs/scene_opt_v1}"
MVP_ROOT="${BTS_MVP_EXPERIMENT_ROOT:-${REPO_ROOT}/runs/scene_opt_v2}"
BTS_SCENES_ROOT="${BTS_SCENES_ROOT:-${REPO_ROOT}/data/bts_scenes}"
BTS_MANIFESTS_ROOT="${BTS_MANIFESTS_ROOT:-${REPO_ROOT}/runs/manifests}"
AUX_SCENES_ROOT="${AUX_SCENES_ROOT:-${REPO_ROOT}/data/auxiliary}"
AUX_MANIFESTS_ROOT="${AUX_MANIFESTS_ROOT:-${REPO_ROOT}/runs/manifests_auxiliary}"
DEFAULT_SCENES=(chair bonsai HCM0674 HCM0540 HCM0644)

if [[ "$#" -gt 0 ]]; then
    SCENE_IDS=("$@")
else
    SCENE_IDS=("${DEFAULT_SCENES[@]}")
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${MVP_ROOT}"
COMMAND_LOG="${MVP_ROOT}/deployment_commands.log"
printf '# %s commit=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$(git rev-parse HEAD)" >> "${COMMAND_LOG}"

invoke_recorded() {
    printf '%q ' "$@" >> "${COMMAND_LOG}"
    printf '\n' >> "${COMMAND_LOG}"
    "$@"
}

for scene_id in "${SCENE_IDS[@]}"; do
    case "${scene_id}" in
        HCM0644|HCM0674|HCM0540)
            ;;
        chair|bonsai)
            ;;
        *)
            echo "ERROR: unsupported five-scene MVP scene: ${scene_id}" >&2
            exit 2
            ;;
    esac
done

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
    for required_artifact in manifest.json arrays.npz holdout.json; do
        artifact_path="${manifests_root}/${scene_id}/${required_artifact}"
        if [[ ! -f "${artifact_path}" ]]; then
            echo "ERROR: paired Stage A artifact is missing: ${artifact_path}" >&2
            echo "Restore the original manifest artifacts used by B0; do not regenerate them." >&2
            exit 2
        fi
    done

    b0_report="${B0_ROOT}/reference/${scene_id}/experiment_report.json"
    if ! invoke_recorded "${PYTHON_BIN}" \
            -m bts_nvs.experiments.run_experiment validate \
            --repo-root "${REPO_ROOT}" \
            --scenes-root "${scenes_root}" \
            --manifests-root "${manifests_root}" \
            --backend-root "${BACKEND_ROOT}" \
            --experiment-root "${B0_ROOT}" \
            --stage reference \
            --scene-id "${scene_id}" \
            --candidate-id B0-reference \
            --stop-step 7000; then
        echo "ERROR: B0 validation failed for ${scene_id}; skipping scene." >&2
        continue
    fi

    decision_path="${MVP_ROOT}/decisions/screen/${scene_id}.json"
    if [[ "${scene_id}" == "HCM0644" ]]; then
        invoke_recorded "${PYTHON_BIN}" \
            -m bts_nvs.experiments.run_experiment retain-b0 \
            --b0-report "${b0_report}" \
            --output "${decision_path}"
        continue
    fi

    case "${scene_id}" in
        HCM0674|HCM0540)
            candidates=(E2-raster-aa-v1)
            ;;
        chair)
            candidates=(E2-loss-local-laplacian-v1)
            ;;
        bonsai)
            candidates=(
                E2-loss-local-laplacian-v1
                E2-appearance-sh4-v1
            )
            ;;
    esac

    candidate_reports=()
    for candidate_id in "${candidates[@]}"; do
        output_dir="${MVP_ROOT}/screen/${scene_id}/${candidate_id}"
        action="run"
        if [[ -d "${output_dir}" ]] \
            && [[ -n "$(find "${output_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
            action="validate"
        fi
        if invoke_recorded "${PYTHON_BIN}" \
                -m bts_nvs.experiments.run_experiment "${action}" \
                --repo-root "${REPO_ROOT}" \
                --scenes-root "${scenes_root}" \
                --manifests-root "${manifests_root}" \
                --backend-root "${BACKEND_ROOT}" \
                --experiment-root "${MVP_ROOT}" \
                --stage screen \
                --scene-id "${scene_id}" \
                --candidate-id "${candidate_id}" \
                --stop-step 7000 \
                --b0-report "${b0_report}"; then
            candidate_reports+=(
                "${output_dir}/experiment_report.json"
            )
        else
            echo "ERROR: ${scene_id}/${candidate_id} failed; continuing." >&2
        fi
    done

    if [[ "${#candidate_reports[@]}" -eq 0 ]]; then
        invoke_recorded "${PYTHON_BIN}" \
            -m bts_nvs.experiments.run_experiment retain-b0 \
            --b0-report "${b0_report}" \
            --output "${decision_path}"
        continue
    fi

    decision_command=(
        "${PYTHON_BIN}"
        -m bts_nvs.experiments.run_experiment decide-screen
        --b0-report "${b0_report}"
        --output "${decision_path}"
    )
    for candidate_report in "${candidate_reports[@]}"; do
        decision_command+=(--candidate-report "${candidate_report}")
    done
    invoke_recorded "${decision_command[@]}"
done

echo "Five-scene MVP screens completed or validated: ${SCENE_IDS[*]}"
echo "Decisions: ${MVP_ROOT}/decisions/screen"
echo "Deployment command log: ${COMMAND_LOG}"
