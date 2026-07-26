from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_five_scene_mvp_screen.sh"


def test_five_scene_mvp_script_locks_scope_and_stage_contract() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "runs/scene_opt_v2" in text
    assert "runs/scene_opt_v1" in text
    assert "E2-raster-aa-v1" in text
    assert "E2-loss-local-laplacian-v1" in text
    assert "E2-appearance-sh4-v1" in text
    assert "retain-b0" in text
    assert "decide-screen" in text
    assert "--stop-step 7000" in text
    assert "DEFAULT_SCENES=(chair bonsai HCM0674 HCM0540 HCM0644)" in text
    assert "Restore the original manifest artifacts used by B0" in text
    assert 'if [[ "${#candidate_reports[@]}" -eq 0 ]]' in text
    assert "failed; continuing." in text
    for forbidden in (
        "--stop-step 30000",
        "--resume",
        "rm ",
        "prepare_phase4_artifacts.py",
    ):
        assert forbidden not in text
