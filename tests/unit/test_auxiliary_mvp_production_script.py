from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_auxiliary_mvp_production.sh"


def test_auxiliary_mvp_script_locks_deadline_production_contract() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "-m bts_nvs.experiments.run_mvp_production" in text
    assert "data/auxiliary" in text
    assert "runs/manifests_auxiliary" in text
    assert "runs/scene_opt_v1" in text
    assert "runs/scene_opt_v2" in text
    assert "set -- chair bonsai" in text
    for flag in (
        "--reference_root",
        "--experiment_root",
        "--output_root",
        "--python_bin",
    ):
        assert flag in text
    for forbidden in ("run_training.py", "--internal_holdout", "--max_steps", "rm "):
        assert forbidden not in text
