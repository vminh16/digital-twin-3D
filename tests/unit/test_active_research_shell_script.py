from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_chair_bonsai_research.sh"


def test_active_research_script_is_a_thin_generic_runner_wrapper() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "-m bts_nvs.experiments.run_experiment" in script
    assert "--stage research" in script
    assert "--stop-step 15000" in script
    assert "BTS_RESEARCH_EXPERIMENT_ROOT" in script
    assert "runs/scene_opt_v3" in script
    assert "chair|bonsai" in script
    assert "-m bts_nvs.data.prepare_research_artifacts" in script
    assert "holdout_research_v3.json" in script
    for forbidden in (
        "run_training.py",
        "--max_steps",
        "--internal_holdout",
        "--rolling_checkpoint",
        "rm ",
    ):
        assert forbidden not in script
