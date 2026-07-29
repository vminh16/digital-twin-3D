from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_chair_perceptual_adc_research.sh"


def test_chair_perceptual_adc_script_is_fresh_staged_and_recoverable() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert script.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "E7-chair-perceptual-adc-corrected-v1" in script
    assert 'STOP_STEP="${E7_STOP_STEP:-15000}"' in script
    assert 'EXTRA_ARGS+=(--resume)' in script
    assert "checkpoints/recovery.pt" in script
    assert "-m bts_nvs.experiments.run_experiment" in script
    assert "--stage research" in script
    assert "--scene-id chair" in script
    for forbidden in ("run_training.py", "--max_steps", "rm "):
        assert forbidden not in script
