from __future__ import annotations

import json
from pathlib import Path

import pytest

from bts_nvs.training.c1_phase_b import (
    LOCKED_CANDIDATE,
    SCREENING_SCENES,
    build_phase_b_decision,
    exact_sign_test,
    save_phase_b_decision,
)
from bts_nvs.training.c1_screening import BASELINE_CANDIDATE, MAX_VRAM_MB


def _report(
    scene: str,
    candidate: str,
    psnr: float,
    *,
    lpips: float = 0.2,
    vram: float = 8_000.0,
) -> dict:
    return {
        "schema_version": 1,
        "scene_id": scene,
        "candidate_id": candidate,
        "step": 7000,
        "image_count": 8,
        "psnr_db_mean": psnr,
        "ssim_mean": 0.8,
        "lpips_mean": lpips,
        "peak_gaussians": 1_000_000,
        "max_vram_mb": vram,
        "total_time_seconds": 900.0,
        "holdout_sha256": f"holdout-{scene}",
    }


def _diagnostic(scene: str, candidate: str, value: float) -> dict:
    return {
        "schema_version": 1,
        "scene_id": scene,
        "candidate_id": candidate,
        "image_count": 8,
        "missing_edge_mean": value,
        "spurious_edge_mean": value,
        "hf_l1_mean": value,
    }


def _matrix(positive_scenes: int):
    baselines = []
    candidates = []
    diagnostics = []
    for index, scene in enumerate(SCREENING_SCENES):
        baselines.append(_report(scene, BASELINE_CANDIDATE, 20.0))
        candidate_psnr = 21.0 if index < positive_scenes else 19.0
        candidates.append(_report(scene, LOCKED_CANDIDATE, candidate_psnr))
        diagnostics.append(_diagnostic(scene, BASELINE_CANDIDATE, 0.10))
        diagnostics.append(_diagnostic(scene, LOCKED_CANDIDATE, 0.09))
    return baselines, candidates, diagnostics


def test_exact_sign_test_matches_locked_six_scene_values() -> None:
    assert exact_sign_test(4, 6) == pytest.approx(0.6875)
    assert exact_sign_test(5, 6) == pytest.approx(0.21875)
    assert exact_sign_test(6, 6) == pytest.approx(0.03125)


def test_phase_b_passes_six_positive_scenes() -> None:
    decision = build_phase_b_decision(*_matrix(positive_scenes=6))

    assert decision["phase_b_passed"] is True
    assert decision["positive_scene_count"] == 6
    assert decision["sign_test_p_value"] == pytest.approx(0.03125)
    assert decision["selected_candidate"] == LOCKED_CANDIDATE
    assert decision["requires_negative_scene_review"] is False


def test_phase_b_rejects_three_positive_scenes() -> None:
    decision = build_phase_b_decision(*_matrix(positive_scenes=3))

    assert decision["phase_b_passed"] is False
    assert decision["gates"]["at_least_four_positive"] is False
    assert decision["gates"]["mean_delta_positive"] is False


def test_phase_b_four_of_six_is_conditional_pass() -> None:
    decision = build_phase_b_decision(*_matrix(positive_scenes=4))

    assert decision["phase_b_passed"] is True
    assert decision["positive_scene_count"] == 4
    assert decision["requires_negative_scene_review"] is True
    assert decision["sign_test_p_value"] == pytest.approx(0.6875)


def test_phase_b_rejects_aggregate_lpips_regression() -> None:
    baselines, candidates, diagnostics = _matrix(positive_scenes=6)
    for report in candidates:
        report["lpips_mean"] = 0.21

    decision = build_phase_b_decision(baselines, candidates, diagnostics)

    assert decision["mean_delta_score50"] > 0.0
    assert decision["gates"]["aggregate_lpips_not_worse"] is False
    assert decision["phase_b_passed"] is False


def test_phase_b_rejects_when_both_edge_errors_worsen_in_aggregate() -> None:
    baselines, candidates, diagnostics = _matrix(positive_scenes=6)
    for diagnostic in diagnostics:
        if diagnostic["candidate_id"] == LOCKED_CANDIDATE:
            diagnostic["missing_edge_mean"] = 0.11
            diagnostic["spurious_edge_mean"] = 0.12

    decision = build_phase_b_decision(baselines, candidates, diagnostics)

    assert decision["gates"]["edge_errors_not_both_worse"] is False
    assert decision["phase_b_passed"] is False


def test_phase_b_rejects_candidate_at_vram_limit() -> None:
    baselines, candidates, diagnostics = _matrix(positive_scenes=6)
    candidates[0]["max_vram_mb"] = MAX_VRAM_MB

    decision = build_phase_b_decision(baselines, candidates, diagnostics)

    assert decision["gates"]["resources_valid"] is False
    assert decision["phase_b_passed"] is False


def test_phase_b_rejects_mismatched_holdout_and_diagnostic_count() -> None:
    baselines, candidates, diagnostics = _matrix(positive_scenes=6)
    candidates[0]["holdout_sha256"] = "different"
    with pytest.raises(ValueError, match="holdout"):
        build_phase_b_decision(baselines, candidates, diagnostics)

    baselines, candidates, diagnostics = _matrix(positive_scenes=6)
    diagnostics[1]["image_count"] = 7
    with pytest.raises(ValueError, match="image counts"):
        build_phase_b_decision(baselines, candidates, diagnostics)


def test_phase_b_rejects_incomplete_duplicate_and_nonfinite_records() -> None:
    baselines, candidates, diagnostics = _matrix(positive_scenes=6)
    with pytest.raises(ValueError, match="exactly"):
        build_phase_b_decision(baselines[:-1], candidates, diagnostics)
    with pytest.raises(ValueError, match="duplicate"):
        build_phase_b_decision(baselines + [baselines[0]], candidates, diagnostics)
    candidates[0]["psnr_db_mean"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_phase_b_decision(baselines, candidates, diagnostics)


def test_phase_b_decision_json_is_deterministic(tmp_path: Path) -> None:
    decision = build_phase_b_decision(*_matrix(positive_scenes=6))
    path = tmp_path / "phase_b_decision.json"

    save_phase_b_decision(decision, path)
    first = path.read_bytes()
    save_phase_b_decision(decision, path)

    assert path.read_bytes() == first
    assert json.loads(first)["phase_b_passed"] is True
    assert b"NaN" not in first and b"Infinity" not in first
