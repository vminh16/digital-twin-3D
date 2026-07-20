from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from bts_nvs.training.c1_screening import (
    BASELINE_CANDIDATE,
    MAX_VRAM_MB,
    atomic_json,
    score50,
)


PHASE_B_SCENES = ("hcm0031", "HCM0181", "HNI0131", "HNI0265")
SCREENING_SCENES = (
    "hcm0031",
    "HCM0181",
    "HCM0421",
    "HCM1439",
    "HNI0131",
    "HNI0265",
)
LOCKED_CANDIDATE = "C1-absgrad-t08-revopacity-v1"


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def exact_sign_test(positive_count: int, total: int) -> float:
    if (
        isinstance(positive_count, bool)
        or isinstance(total, bool)
        or not isinstance(positive_count, int)
        or not isinstance(total, int)
        or total <= 0
        or not 0 <= positive_count <= total
    ):
        raise ValueError("sign-test counts must satisfy 0 <= positive <= total")
    tail = min(positive_count, total - positive_count)
    probability = 2.0 * sum(math.comb(total, k) for k in range(tail + 1)) / (
        2**total
    )
    return min(1.0, probability)


def _validate_report(report: Mapping[str, object], candidate_id: str) -> None:
    if report.get("schema_version") != 1 or report.get("step") != 7000:
        raise ValueError("Phase B report must use schema 1 at step 7000")
    if report.get("scene_id") not in SCREENING_SCENES:
        raise ValueError("Phase B report contains an unexpected scene")
    if report.get("candidate_id") != candidate_id:
        raise ValueError("Phase B report contains an unexpected candidate")
    image_count = report.get("image_count")
    peak_gaussians = report.get("peak_gaussians")
    if isinstance(image_count, bool) or not isinstance(image_count, int) or image_count <= 0:
        raise ValueError("Phase B image_count must be positive")
    if (
        isinstance(peak_gaussians, bool)
        or not isinstance(peak_gaussians, int)
        or peak_gaussians <= 0
    ):
        raise ValueError("Phase B peak_gaussians must be positive")
    holdout_hash = report.get("holdout_sha256")
    if not isinstance(holdout_hash, str) or not holdout_hash:
        raise ValueError("Phase B report requires a holdout hash")
    score50(report)
    for key in ("max_vram_mb", "total_time_seconds"):
        _finite_number(report.get(key), key)


def _validate_diagnostic(record: Mapping[str, object], candidate_id: str) -> None:
    if record.get("schema_version") != 1:
        raise ValueError("Phase B diagnostic must use schema 1")
    if record.get("scene_id") not in SCREENING_SCENES:
        raise ValueError("Phase B diagnostic contains an unexpected scene")
    if record.get("candidate_id") != candidate_id:
        raise ValueError("Phase B diagnostic contains an unexpected candidate")
    image_count = record.get("image_count")
    if isinstance(image_count, bool) or not isinstance(image_count, int) or image_count <= 0:
        raise ValueError("Phase B diagnostic image_count must be positive")
    for key in ("missing_edge_mean", "spurious_edge_mean", "hf_l1_mean"):
        if _finite_number(record.get(key), key) < 0.0:
            raise ValueError(f"{key} must be non-negative")


def _index_records(
    records: Sequence[Mapping[str, object]],
    *,
    candidate_id: str,
    diagnostic: bool,
) -> dict[str, Mapping[str, object]]:
    indexed: dict[str, Mapping[str, object]] = {}
    validator = _validate_diagnostic if diagnostic else _validate_report
    for record in records:
        validator(record, candidate_id)
        scene_id = str(record["scene_id"])
        if scene_id in indexed:
            raise ValueError("duplicate Phase B record")
        indexed[scene_id] = record
    if set(indexed) != set(SCREENING_SCENES):
        raise ValueError(f"Phase B requires exactly {len(SCREENING_SCENES)} records")
    return indexed


def build_phase_b_decision(
    baseline_reports: Sequence[Mapping[str, object]],
    candidate_reports: Sequence[Mapping[str, object]],
    diagnostics: Sequence[Mapping[str, object]],
) -> dict:
    baseline_index = _index_records(
        baseline_reports,
        candidate_id=BASELINE_CANDIDATE,
        diagnostic=False,
    )
    candidate_index = _index_records(
        candidate_reports,
        candidate_id=LOCKED_CANDIDATE,
        diagnostic=False,
    )
    baseline_diagnostics = _index_records(
        [
            record
            for record in diagnostics
            if record.get("candidate_id") == BASELINE_CANDIDATE
        ],
        candidate_id=BASELINE_CANDIDATE,
        diagnostic=True,
    )
    candidate_diagnostics = _index_records(
        [
            record
            for record in diagnostics
            if record.get("candidate_id") == LOCKED_CANDIDATE
        ],
        candidate_id=LOCKED_CANDIDATE,
        diagnostic=True,
    )
    if len(diagnostics) != 2 * len(SCREENING_SCENES):
        raise ValueError("Phase B requires exactly twelve diagnostic records")

    scenes: dict[str, dict] = {}
    for scene_id in SCREENING_SCENES:
        baseline = baseline_index[scene_id]
        candidate = candidate_index[scene_id]
        baseline_diagnostic = baseline_diagnostics[scene_id]
        candidate_diagnostic = candidate_diagnostics[scene_id]
        if baseline["holdout_sha256"] != candidate["holdout_sha256"]:
            raise ValueError(f"Phase B holdout hash mismatch for {scene_id}")
        if (
            baseline["image_count"] != baseline_diagnostic["image_count"]
            or candidate["image_count"] != candidate_diagnostic["image_count"]
        ):
            raise ValueError(f"Phase B report and diagnostic image counts differ for {scene_id}")
        baseline_score = score50(baseline)
        candidate_score = score50(candidate)
        scenes[scene_id] = {
            "baseline_score50": baseline_score,
            "candidate_score50": candidate_score,
            "delta_score50": candidate_score - baseline_score,
            "baseline_psnr_db": float(baseline["psnr_db_mean"]),
            "candidate_psnr_db": float(candidate["psnr_db_mean"]),
            "baseline_ssim": float(baseline["ssim_mean"]),
            "candidate_ssim": float(candidate["ssim_mean"]),
            "baseline_lpips": float(baseline["lpips_mean"]),
            "candidate_lpips": float(candidate["lpips_mean"]),
            "missing_edge_delta": float(
                candidate_diagnostic["missing_edge_mean"]
                - baseline_diagnostic["missing_edge_mean"]
            ),
            "spurious_edge_delta": float(
                candidate_diagnostic["spurious_edge_mean"]
                - baseline_diagnostic["spurious_edge_mean"]
            ),
            "candidate_hf_l1": float(candidate_diagnostic["hf_l1_mean"]),
            "peak_gaussians": int(candidate["peak_gaussians"]),
            "max_vram_mb": float(candidate["max_vram_mb"]),
            "total_time_seconds": float(candidate["total_time_seconds"]),
        }

    scene_values = list(scenes.values())
    deltas = [item["delta_score50"] for item in scene_values]
    positive_count = sum(delta > 0.0 for delta in deltas)
    baseline_lpips = float(np.mean([item["baseline_lpips"] for item in scene_values]))
    candidate_lpips = float(np.mean([item["candidate_lpips"] for item in scene_values]))
    missing_edge_delta = float(
        np.mean([item["missing_edge_delta"] for item in scene_values])
    )
    spurious_edge_delta = float(
        np.mean([item["spurious_edge_delta"] for item in scene_values])
    )
    mean_delta = float(np.mean(deltas))
    gates = {
        "mean_delta_positive": mean_delta > 0.0,
        "at_least_four_positive": positive_count >= 4,
        "aggregate_lpips_not_worse": candidate_lpips <= baseline_lpips,
        "edge_errors_not_both_worse": not (
            missing_edge_delta > 0.0 and spurious_edge_delta > 0.0
        ),
        "resources_valid": all(
            item["max_vram_mb"] < MAX_VRAM_MB for item in scene_values
        ),
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "phase": "C1-phase-B",
        "scenes": list(SCREENING_SCENES),
        "candidate_id": LOCKED_CANDIDATE,
        "phase_b_passed": bool(passed),
        "selected_candidate": LOCKED_CANDIDATE if passed else None,
        "positive_scene_count": int(positive_count),
        "sign_test_p_value": exact_sign_test(positive_count, len(SCREENING_SCENES)),
        "requires_negative_scene_review": bool(passed and positive_count == 4),
        "mean_delta_score50": mean_delta,
        "median_delta_score50": float(np.median(deltas)),
        "baseline_lpips_mean": baseline_lpips,
        "candidate_lpips_mean": candidate_lpips,
        "missing_edge_delta_mean": missing_edge_delta,
        "spurious_edge_delta_mean": spurious_edge_delta,
        "gates": gates,
        "scenes_detail": scenes,
    }


def save_phase_b_decision(decision: dict, path: Path) -> None:
    atomic_json(path, decision)
