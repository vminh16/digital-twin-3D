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
OUTPUT_ROOT="${BTS_BACKEND_ROOT:-${REPO_ROOT}/runs/phase4/backend_qualification}"

variants=(
    "adam:fp32"
    "adam-fused:fp32"
    "adam-fused:amp-fp16"
)
run_names=("reference" "fused" "amp")

cd "${REPO_ROOT}"
BTS_SCENES_ROOT="${SCENES_ROOT}" \
BTS_MANIFESTS_ROOT="${MANIFESTS_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
"${BASH}" "${SCRIPT_DIR}/prepare_phase4_artifacts.sh"

for index in "${!variants[@]}"; do
    variant="${variants[index]}"
    backend="${variant%%:*}"
    precision="${variant#*:}"
    run_dir="${OUTPUT_ROOT}/${run_names[index]}"
    profile="${run_dir}/backend_profile.json"

    if [[ -f "${profile}" ]]; then
        echo "Backend profile already complete: ${profile}"
        continue
    fi
    if [[ -e "${run_dir}" ]]; then
        shopt -s nullglob dotglob
        existing=("${run_dir}"/*)
        shopt -u nullglob dotglob
        if (( ${#existing[@]} > 0 )); then
            echo "ERROR: ${run_dir} exists but has no backend profile" >&2
            exit 1
        fi
    fi

    "${PYTHON_BIN}" src/bts_nvs/training/run_training.py \
        --scene_dir "${SCENES_ROOT}/HCM0181" \
        --manifest_dir "${MANIFESTS_ROOT}/HCM0181" \
        --output_dir "${run_dir}" \
        --resize_factor 1 \
        --max_steps 1000 \
        --checkpoint_every 1000 \
        --seed 0 \
        --cache_images \
        --pinned_transfer \
        --optimizer_backend "${backend}" \
        --precision "${precision}" \
        --backend_qualification
done

"${PYTHON_BIN}" -m bts_nvs.training.compare_backend_qualification \
    --reference "${OUTPUT_ROOT}/reference/backend_profile.json" \
    --fused "${OUTPUT_ROOT}/fused/backend_profile.json" \
    --amp "${OUTPUT_ROOT}/amp/backend_profile.json" \
    --output "${OUTPUT_ROOT}/backend_qualification.json"

echo "Backend qualification complete: ${OUTPUT_ROOT}/backend_qualification.json"

