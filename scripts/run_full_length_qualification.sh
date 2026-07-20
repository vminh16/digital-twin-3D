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
DRY_RUN_ROOT="${BTS_DRY_RUN_ROOT:-${REPO_ROOT}/runs/phase4/dry_run_30k}"
RUN_DIR="${DRY_RUN_ROOT}/HCM0181"
RECOVERY="${RUN_DIR}/checkpoints/recovery.pt"
REPORT="${RUN_DIR}/full_length_report.json"
RESUME="${BTS_RESUME:-0}"

cd "${REPO_ROOT}"
BTS_SCENES_ROOT="${SCENES_ROOT}" \
BTS_MANIFESTS_ROOT="${MANIFESTS_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
"${BASH}" "${SCRIPT_DIR}/prepare_scene_manifests.sh"

if [[ -f "${REPORT}" ]]; then
    echo "30k dry run already complete: ${REPORT}"
    exit 0
fi

if [[ "${RESUME}" == "1" ]]; then
    if [[ ! -f "${RECOVERY}" ]]; then
        echo "ERROR: BTS_RESUME=1 requires ${RECOVERY}" >&2
        exit 1
    fi
elif [[ "${RESUME}" != "0" ]]; then
    echo "ERROR: BTS_RESUME must be 0 or 1" >&2
    exit 1
elif [[ -e "${RUN_DIR}" ]]; then
    shopt -s nullglob dotglob
    existing=("${RUN_DIR}"/*)
    shopt -u nullglob dotglob
    if (( ${#existing[@]} > 0 )); then
        echo "ERROR: incomplete run exists; inspect it and use BTS_RESUME=1" >&2
        exit 1
    fi
fi

echo "Checking the pretrained LPIPS backend before the 30k run..."
BTS_RUN_LPIPS_SMOKE=1 "${PYTHON_BIN}" -m pytest \
    tests/integration/test_lpips_smoke.py -q

args=(
    --scene_dir "${SCENES_ROOT}/HCM0181"
    --manifest_dir "${MANIFESTS_ROOT}/HCM0181"
    --output_dir "${RUN_DIR}"
    --resize_factor 1
    --max_steps 30000
    --checkpoint_every 3000
    --seed 0
    --cache_images
    --pinned_transfer
    --full_length_qualification
)
if [[ "${RESUME}" == "1" ]]; then
    args+=(--resume "${RECOVERY}")
fi

"${PYTHON_BIN}" src/bts_nvs/training/run_training.py "${args[@]}"

if [[ ! -f "${REPORT}" ]]; then
    echo "ERROR: training ended without ${REPORT}" >&2
    exit 1
fi
echo "30k dry run complete: ${REPORT}"
