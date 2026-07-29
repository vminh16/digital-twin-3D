from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_chair_spectral_research.sh"


def test_chair_spectral_script_is_fresh_15k_and_candidate_locked() -> None:
    script = SCRIPT.read_text(encoding="utf-8")

    assert "E8-chair-observation-scale-spectral-split-v1" in script
    assert '--stop-step 15000' in script
    assert "E8 is authorized only for the fresh 15000-step gate" in script
    assert "--scene-id chair" in script
    assert "--stage research" in script
