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
QUALIFICATION_ROOT="${BTS_QUALIFICATION_ROOT:-${REPO_ROOT}/runs/phase4/qualification}"

SCENES=(hcm0031 HCM0181 HCM0421 HCM1439 HNI0131 HNI0265)
CANDIDATES=(B0-reference B0-compact)

cd "${REPO_ROOT}"
BTS_SCENES_ROOT="${SCENES_ROOT}" \
BTS_MANIFESTS_ROOT="${MANIFESTS_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash "${SCRIPT_DIR}/prepare_phase4_artifacts.sh"

echo "Checking the pretrained LPIPS backend before GPU qualification..."
BTS_RUN_LPIPS_SMOKE=1 "${PYTHON_BIN}" -m pytest \
    tests/integration/test_lpips_smoke.py -q

for scene_id in "${SCENES[@]}"; do
    for candidate_id in "${CANDIDATES[@]}"; do
        run_dir="${QUALIFICATION_ROOT}/${scene_id}/${candidate_id}"
        report_path="${run_dir}/qualification_report.json"

        if [[ -f "${report_path}" ]]; then
            echo "${scene_id}/${candidate_id} already complete; skipping"
            continue
        fi
        if [[ -e "${run_dir}" ]]; then
            echo "ERROR: ${run_dir} exists but has no qualification report" >&2
            echo "Inspect it, then remove it or choose a new BTS_QUALIFICATION_ROOT." >&2
            exit 1
        fi

        echo "Running ${scene_id}/${candidate_id}..."
        "${PYTHON_BIN}" src/bts_nvs/training/run_training.py \
            --scene_dir "${SCENES_ROOT}/${scene_id}" \
            --manifest_dir "${MANIFESTS_ROOT}/${scene_id}" \
            --output_dir "${run_dir}" \
            --resize_factor 1 \
            --max_steps 7000 \
            --seed 0 \
            --cache_images \
            --pinned_transfer \
            --qualification_candidate "${candidate_id}"
    done
done

"${PYTHON_BIN}" src/bts_nvs/training/decide_qualification.py \
    --reports_root "${QUALIFICATION_ROOT}" \
    --output "${QUALIFICATION_ROOT}/qualification_decision.json"

echo "Qualification complete: ${QUALIFICATION_ROOT}/qualification_decision.json"
