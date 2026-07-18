from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PREPARE_SCRIPT = REPO_ROOT / "scripts" / "prepare_phase4_artifacts.sh"
QUALIFICATION_SCRIPT = REPO_ROOT / "scripts" / "run_phase4_qualification.sh"
DRY_RUN_SCRIPT = REPO_ROOT / "scripts" / "run_phase4_30k_dry_run.sh"
BACKEND_SCRIPT = REPO_ROOT / "scripts" / "run_phase4_backend_qualification.sh"
FULL_TRAINING_SCRIPT = REPO_ROOT / "scripts" / "run_phase4_full_training.sh"


def _read_script(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_artifact_script_locks_the_canonical_scene_pool():
    script = _read_script(PREPARE_SCRIPT)

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert 'SCENES_ROOT="${BTS_SCENES_ROOT:-${REPO_ROOT}/data/bts_scenes}"' in script
    assert 'MANIFESTS_ROOT="${BTS_MANIFESTS_ROOT:-${REPO_ROOT}/runs/manifests}"' in script
    assert "--expected_scenes 18" in script
    assert "--require_expected" in script


def test_qualification_script_runs_the_locked_matrix_and_decision():
    script = _read_script(QUALIFICATION_SCRIPT)

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    for scene_id in (
        "hcm0031",
        "HCM0181",
        "HCM0421",
        "HCM1439",
        "HNI0131",
        "HNI0265",
    ):
        assert script.count(scene_id) == 1
    for candidate_id in ("B0-reference", "B0-compact"):
        assert script.count(candidate_id) == 1
    assert "prepare_phase4_artifacts.sh" in script
    assert "BTS_RUN_LPIPS_SMOKE=1" in script
    assert "tests/integration/test_lpips_smoke.py" in script
    assert "--qualification_candidate" in script
    assert "--resize_factor 1" in script
    assert "--max_steps 7000" in script
    assert "--seed 0" in script
    assert "--cache_images" in script
    assert "--pinned_transfer" in script
    assert "qualification_report.json" in script
    assert "decide_qualification.py" in script


def test_qualification_script_never_deletes_or_overwrites_partial_runs():
    script = _read_script(QUALIFICATION_SCRIPT)

    assert "rm " not in script
    assert "rm\n" not in script
    assert "already complete; skipping" in script
    assert "exists but has no qualification report" in script


def test_scripts_do_not_require_external_dirname_command():
    assert "dirname" not in _read_script(PREPARE_SCRIPT)
    assert "dirname" not in _read_script(QUALIFICATION_SCRIPT)


def test_30k_dry_run_script_locks_science_and_bounded_artifacts():
    script = _read_script(DRY_RUN_SCRIPT)

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "prepare_phase4_artifacts.sh" in script
    assert "BTS_RUN_LPIPS_SMOKE=1" in script
    assert "HCM0181" in script
    assert "--resize_factor 1" in script
    assert "--max_steps 30000" in script
    assert "--checkpoint_every 3000" in script
    assert "--seed 0" in script
    assert "--cache_images" in script
    assert "--pinned_transfer" in script
    assert "--full_length_qualification" in script
    assert "checkpoints/recovery.pt" in script
    assert "BTS_RESUME" in script
    assert "full_length_report.json" in script
    assert "rm " not in script


def test_backend_qualification_script_runs_three_fresh_1000_step_jobs():
    script = _read_script(BACKEND_SCRIPT)

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert script.count("--backend_qualification") == 1
    assert script.count("--max_steps 1000") == 1
    assert "--resize_factor 1" in script
    assert "--seed 0" in script
    assert "--cache_images" in script
    assert "--pinned_transfer" in script
    assert '"adam:fp32"' in script
    assert '"adam-fused:fp32"' in script
    assert '"adam-fused:amp-fp16"' in script
    assert "compare_backend_qualification" in script
    assert "backend_qualification.json" in script
    assert "rm " not in script


def test_full_training_script_is_a_path_only_wrapper():
    script = _read_script(FULL_TRAINING_SCRIPT)

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "prepare_phase4_artifacts.sh" in script
    assert 'BACKEND_ROOT="${BTS_BACKEND_ROOT:-' in script
    assert 'FULL_ROOT="${BTS_FULL_ROOT:-' in script
    assert 'PYTHONPATH="${REPO_ROOT}/src' in script
    assert "-m bts_nvs.training.run_full_training" in script
    assert '"$@"' in script
    for flag in (
        "--repo_root",
        "--scenes_root",
        "--manifests_root",
        "--backend_root",
        "--output_root",
        "--python_bin",
    ):
        assert flag in script
    for forbidden in (
        "rm ",
        "--max_steps",
        "--optimizer_backend",
        "--precision",
        "for scene",
    ):
        assert forbidden not in script
    for scene_id in ("HCM0644", "HCM0674", "HCM0540", "HCM0539", "HCM0421"):
        assert scene_id not in script
