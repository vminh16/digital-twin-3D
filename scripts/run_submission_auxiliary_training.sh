#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCENES_ROOT="${AUX_SCENES_ROOT:-${REPO_ROOT}/data/auxiliary}"
MANIFESTS_ROOT="${AUX_MANIFESTS_ROOT:-${REPO_ROOT}/runs/manifests_auxiliary}"
BACKEND_ROOT="${BTS_BACKEND_ROOT:-${REPO_ROOT}/runs/phase4/backend_qualification}"
OUTPUT_ROOT="${AUX_FULL_ROOT:-${REPO_ROOT}/runs/phase4/auxiliary_training}"
if [[ "$#" -gt 0 ]]; then
    SCENE_IDS=("$@")
else
    SCENE_IDS=(chair bonsai)
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" src/bts_nvs/data/prepare_phase4_artifacts.py \
    --scenes_root "${SCENES_ROOT}" \
    --manifests_root "${MANIFESTS_ROOT}" \
    --expected_scenes 2 \
    --require_expected

read -r OPTIMIZER_BACKEND PRECISION < <(
    "${PYTHON_BIN}" -c \
        "from pathlib import Path; from bts_nvs.training.full_training import load_or_create_backend_decision; d=load_or_create_backend_decision(Path(r'${BACKEND_ROOT}')); print(d.optimizer_backend, d.precision)"
)

for SCENE_ID in "${SCENE_IDS[@]}"; do
    RUN_DIR="${OUTPUT_ROOT}/scenes/${SCENE_ID}"
    RESUME_ARGS=()
    if [[ -f "${RUN_DIR}/checkpoints/recovery.pt" ]]; then
        RESUME_ARGS=(--resume "${RUN_DIR}/checkpoints/recovery.pt")
    elif [[ -d "${RUN_DIR}" ]] && [[ -n "$(find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "ERROR: non-empty run has no recovery checkpoint: ${RUN_DIR}" >&2
        exit 1
    fi

    "${PYTHON_BIN}" src/bts_nvs/training/run_training.py \
        --scene_dir "${SCENES_ROOT}/${SCENE_ID}" \
        --manifest_dir "${MANIFESTS_ROOT}/${SCENE_ID}" \
        --output_dir "${RUN_DIR}" \
        --resize_factor 1 \
        --max_steps 30000 \
        --checkpoint_every 3000 \
        --seed 0 \
        --cache_images \
        --pinned_transfer \
        --optimizer_backend "${OPTIMIZER_BACKEND}" \
        --precision "${PRECISION}" \
        --rolling_checkpoint \
        "${RESUME_ARGS[@]}"
done
