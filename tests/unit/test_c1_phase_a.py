import json

import pytest

from bts_nvs.training.c1_candidates import C1_CANDIDATES
from bts_nvs.training.c1_phase_a import (
    PHASE_A_SCENES,
    build_phase_a_decision,
    save_phase_a_decision,
    score50,
)


def _report(scene: str, candidate: str, psnr: float) -> dict:
    return {
        "schema_version": 1,
        "scene_id": scene,
        "candidate_id": candidate,
        "step": 7000,
        "image_count": 8,
        "psnr_db_mean": psnr,
        "ssim_mean": 0.8,
        "lpips_mean": 0.2,
        "peak_gaussians": 1_000_000,
        "max_vram_mb": 8_000.0,
        "total_time_seconds": 900.0,
    }


def _diagnostic(
    scene: str,
    candidate: str,
    *,
    missing: float = 0.1,
    spurious: float = 0.1,
    hf_l1: float = 0.1,
) -> dict:
    return {
        "schema_version": 1,
        "scene_id": scene,
        "candidate_id": candidate,
        "image_count": 8,
        "missing_edge_mean": missing,
        "spurious_edge_mean": spurious,
        "hf_l1_mean": hf_l1,
    }


def _matrix():
    baselines = [_report(scene, "B0-reference", 20.0) for scene in PHASE_A_SCENES]
    candidates = []
    diagnostics = []
    for scene in PHASE_A_SCENES:
        diagnostics.append(_diagnostic(scene, "B0-reference"))
        candidates.append(_report(scene, C1_CANDIDATES[0], 21.0))
        candidates.append(_report(scene, C1_CANDIDATES[1], 20.5))
        diagnostics.append(
            _diagnostic(scene, C1_CANDIDATES[0], missing=0.08, spurious=0.09)
        )
        diagnostics.append(
            _diagnostic(scene, C1_CANDIDATES[1], missing=0.09, spurious=0.09)
        )
    return baselines, candidates, diagnostics


def test_score50_matches_locked_formula() -> None:
    report = {"psnr_db_mean": 25.0, "ssim_mean": 0.8, "lpips_mean": 0.2}

    assert score50(report) == pytest.approx(71.0)


def test_candidate_must_win_both_scenes_and_not_worsen_both_edge_errors() -> None:
    baselines, candidates, diagnostics = _matrix()

    decision = build_phase_a_decision(baselines, candidates, diagnostics)

    assert decision["phase_a_passed"] is True
    assert decision["selected_candidate"] == C1_CANDIDATES[0]
    selected = decision["candidates"][C1_CANDIDATES[0]]
    assert selected["eligible"] is True
    assert all(item["delta_score50"] > 0.0 for item in selected["scenes"].values())


def test_candidate_is_rejected_when_both_edge_errors_worsen() -> None:
    baselines, candidates, diagnostics = _matrix()
    for item in diagnostics:
        if item["candidate_id"] == C1_CANDIDATES[0] and item["scene_id"] == "HCM0421":
            item["missing_edge_mean"] = 0.2
            item["spurious_edge_mean"] = 0.2
        if item["candidate_id"] == C1_CANDIDATES[1]:
            item["missing_edge_mean"] = 0.2
            item["spurious_edge_mean"] = 0.2

    decision = build_phase_a_decision(baselines, candidates, diagnostics)

    assert decision["phase_a_passed"] is False
    assert decision["selected_candidate"] is None


def test_incomplete_duplicate_and_nonfinite_matrices_are_rejected() -> None:
    baselines, candidates, diagnostics = _matrix()
    with pytest.raises(ValueError, match="exactly"):
        build_phase_a_decision(baselines[:-1], candidates, diagnostics)
    with pytest.raises(ValueError, match="duplicate"):
        build_phase_a_decision(baselines + [baselines[0]], candidates, diagnostics)
    candidates[0]["psnr_db_mean"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_phase_a_decision(baselines, candidates, diagnostics)


def test_phase_a_decision_json_is_deterministic(tmp_path) -> None:
    decision = build_phase_a_decision(*_matrix())
    path = tmp_path / "phase_a_decision.json"

    save_phase_a_decision(decision, path)
    first = path.read_bytes()
    save_phase_a_decision(decision, path)

    assert path.read_bytes() == first
    assert json.loads(first)["selected_candidate"] == C1_CANDIDATES[0]
    assert b"NaN" not in first and b"Infinity" not in first
