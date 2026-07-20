from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from bts_nvs.training.c1_candidates import C1_CANDIDATES
from bts_nvs.training.c1_screening import (
    BASELINE_CANDIDATE,
    MAX_VRAM_MB,
    score50,
)


PHASE_A_SCENES = ("HCM0421", "HCM1439")


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_report(report: Mapping[str, object], candidates: set[str]) -> None:
    if report.get("schema_version") != 1 or report.get("step") != 7000:
        raise ValueError("Phase A report must use schema 1 at step 7000")
    if report.get("scene_id") not in PHASE_A_SCENES:
        raise ValueError("Phase A report contains an unexpected scene")
    if report.get("candidate_id") not in candidates:
        raise ValueError("Phase A report contains an unexpected candidate")
    image_count = report.get("image_count")
    peak_gaussians = report.get("peak_gaussians")
    if isinstance(image_count, bool) or not isinstance(image_count, int) or image_count <= 0:
        raise ValueError("Phase A image_count must be positive")
    if (
        isinstance(peak_gaussians, bool)
        or not isinstance(peak_gaussians, int)
        or peak_gaussians <= 0
    ):
        raise ValueError("Phase A peak_gaussians must be positive")
    for key in (
        "psnr_db_mean",
        "ssim_mean",
        "lpips_mean",
        "max_vram_mb",
        "total_time_seconds",
    ):
        _finite_number(report.get(key), key)


def _validate_diagnostic(record: Mapping[str, object], candidates: set[str]) -> None:
    if record.get("schema_version") != 1:
        raise ValueError("Phase A diagnostic must use schema 1")
    if record.get("scene_id") not in PHASE_A_SCENES:
        raise ValueError("Phase A diagnostic contains an unexpected scene")
    if record.get("candidate_id") not in candidates:
        raise ValueError("Phase A diagnostic contains an unexpected candidate")
    image_count = record.get("image_count")
    if isinstance(image_count, bool) or not isinstance(image_count, int) or image_count <= 0:
        raise ValueError("Phase A diagnostic image_count must be positive")
    for key in ("missing_edge_mean", "spurious_edge_mean", "hf_l1_mean"):
        if _finite_number(record.get(key), key) < 0.0:
            raise ValueError(f"{key} must be non-negative")


def _index_records(
    records: Sequence[Mapping[str, object]],
    *,
    candidates: set[str],
    diagnostic: bool,
) -> dict[tuple[str, str], Mapping[str, object]]:
    indexed: dict[tuple[str, str], Mapping[str, object]] = {}
    validator = _validate_diagnostic if diagnostic else _validate_report
    for record in records:
        validator(record, candidates)
        key = (str(record["scene_id"]), str(record["candidate_id"]))
        if key in indexed:
            raise ValueError("duplicate Phase A record")
        indexed[key] = record
    required = {
        (scene_id, candidate_id)
        for scene_id in PHASE_A_SCENES
        for candidate_id in candidates
    }
    if set(indexed) != required:
        raise ValueError(f"Phase A requires exactly {len(required)} records")
    return indexed


def build_phase_a_decision(
    baseline_reports: Sequence[Mapping[str, object]],
    candidate_reports: Sequence[Mapping[str, object]],
    diagnostics: Sequence[Mapping[str, object]],
) -> dict:
    baseline_index = _index_records(
        baseline_reports,
        candidates={BASELINE_CANDIDATE},
        diagnostic=False,
    )
    candidate_index = _index_records(
        candidate_reports,
        candidates=set(C1_CANDIDATES),
        diagnostic=False,
    )
    diagnostic_index = _index_records(
        diagnostics,
        candidates={BASELINE_CANDIDATE, *C1_CANDIDATES},
        diagnostic=True,
    )

    summaries: dict[str, dict] = {}
    for candidate_id in C1_CANDIDATES:
        scenes: dict[str, dict] = {}
        for scene_id in PHASE_A_SCENES:
            baseline = baseline_index[(scene_id, BASELINE_CANDIDATE)]
            candidate = candidate_index[(scene_id, candidate_id)]
            baseline_diagnostic = diagnostic_index[(scene_id, BASELINE_CANDIDATE)]
            candidate_diagnostic = diagnostic_index[(scene_id, candidate_id)]
            if candidate["image_count"] != candidate_diagnostic["image_count"]:
                raise ValueError("candidate report and diagnostic image counts differ")
            if baseline["image_count"] != baseline_diagnostic["image_count"]:
                raise ValueError("baseline report and diagnostic image counts differ")
            missing_worsened = (
                candidate_diagnostic["missing_edge_mean"]
                > baseline_diagnostic["missing_edge_mean"]
            )
            spurious_worsened = (
                candidate_diagnostic["spurious_edge_mean"]
                > baseline_diagnostic["spurious_edge_mean"]
            )
            scenes[scene_id] = {
                "baseline_score50": score50(baseline),
                "candidate_score50": score50(candidate),
                "delta_score50": score50(candidate) - score50(baseline),
                "missing_edge_delta": float(
                    candidate_diagnostic["missing_edge_mean"]
                    - baseline_diagnostic["missing_edge_mean"]
                ),
                "spurious_edge_delta": float(
                    candidate_diagnostic["spurious_edge_mean"]
                    - baseline_diagnostic["spurious_edge_mean"]
                ),
                "both_edge_errors_worsened": bool(
                    missing_worsened and spurious_worsened
                ),
                "hf_l1": float(candidate_diagnostic["hf_l1_mean"]),
                "peak_gaussians": int(candidate["peak_gaussians"]),
                "max_vram_mb": float(candidate["max_vram_mb"]),
            }
        eligible = all(
            item["delta_score50"] > 0.0
            and not item["both_edge_errors_worsened"]
            and item["max_vram_mb"] < MAX_VRAM_MB
            for item in scenes.values()
        )
        summaries[candidate_id] = {
            "eligible": bool(eligible),
            "mean_delta_score50": float(
                np.mean([item["delta_score50"] for item in scenes.values()])
            ),
            "mean_hf_l1": float(
                np.mean([item["hf_l1"] for item in scenes.values()])
            ),
            "mean_peak_gaussians": float(
                np.mean([item["peak_gaussians"] for item in scenes.values()])
            ),
            "scenes": scenes,
        }

    eligible_candidates = [
        candidate_id
        for candidate_id in C1_CANDIDATES
        if summaries[candidate_id]["eligible"]
    ]
    selected = (
        min(
            eligible_candidates,
            key=lambda candidate_id: (
                -summaries[candidate_id]["mean_delta_score50"],
                summaries[candidate_id]["mean_hf_l1"],
                summaries[candidate_id]["mean_peak_gaussians"],
                C1_CANDIDATES.index(candidate_id),
            ),
        )
        if eligible_candidates
        else None
    )
    return {
        "schema_version": 1,
        "phase": "C1-phase-A",
        "scenes": list(PHASE_A_SCENES),
        "phase_a_passed": selected is not None,
        "selected_candidate": selected,
        "candidates": summaries,
    }


def save_phase_a_decision(decision: dict, path: Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, output)
