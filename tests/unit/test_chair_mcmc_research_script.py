from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_chair_mcmc_research.sh"


def test_chair_mcmc_script_is_locked_and_recoverable() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "E5-chair-observation-scale-mcmc-v1" in script
    assert "-m bts_nvs.experiments.run_experiment" in script
    assert "--stage research" in script
    assert "--scene-id chair" in script
    assert "--stop-step 30000" in script
    assert "checkpoints/recovery.pt" in script
    assert "EXTRA_ARGS+=(--resume)" in script
    assert "-m bts_nvs.data.prepare_research_artifacts" in script
    assert "holdout_research_v3.json" not in script
    for forbidden in ("run_training.py", "--max_steps", "--internal_holdout", "rm "):
        assert forbidden not in script
