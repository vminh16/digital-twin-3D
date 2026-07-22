from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from bts_nvs.experiments.candidates import candidate_settings
from bts_nvs.experiments.experiment import (
    COHORT_SCENE_IDS,
    validate_paired_wall_time_ratio,
    validate_peak_vram_mb,
)
from bts_nvs.experiments.provenance import (
    canonical_json_sha256,
    save_json_artifact,
)


_PAIR_FIELDS = ("scene_id", "step", "manifest_sha256", "holdout_sha256")
_QUALITY_GATES = (
    "score50_improved",
    "lpips_not_worse",
    "hard_score_not_worse",
    "detail_not_jointly_worse",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def evaluate_candidate(
    b0_report: Mapping[str, object],
    candidate_report: Mapping[str, object],
    *,
    integrity_passed: bool = True,
    primitive_growth_controlled: bool = True,
    result_15k: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate one already-validated candidate against its paired B0 report."""
    b0 = _report(b0_report, "b0_report")
    candidate = _report(candidate_report, "candidate_report")
    _validate_pair(b0, candidate)
    if b0.get("candidate_id") != "B0-reference":
        raise ValueError("b0_report candidate_id must be B0-reference")
    candidate_id = candidate.get("candidate_id")
    candidate_settings(_string(candidate_id, "candidate_id"))
    if candidate_id == "B0-reference":
        raise ValueError("candidate_report must be a non-B0 candidate")
    step = _positive_int(candidate.get("step"), "step")
    if step not in (7_000, 15_000, 30_000):
        raise ValueError("decision step must be 7000, 15000, or 30000")
    if not isinstance(integrity_passed, bool) or not isinstance(
        primitive_growth_controlled, bool
    ):
        raise ValueError("integrity and primitive growth flags must be boolean")

    b0_overall = _mapping(b0.get("overall"), "b0 overall")
    candidate_overall = _mapping(candidate.get("overall"), "candidate overall")
    b0_hard = _mapping(_mapping(b0.get("strata"), "b0 strata").get("hard"), "b0 hard")
    candidate_hard = _mapping(
        _mapping(candidate.get("strata"), "candidate strata").get("hard"),
        "candidate hard",
    )
    b0_resources = _mapping(b0.get("resources"), "b0 resources")
    candidate_resources = _mapping(candidate.get("resources"), "candidate resources")

    deltas = {
        "score50": _finite(candidate_overall.get("score50"), "candidate score50")
        - _finite(b0_overall.get("score50"), "B0 score50"),
        "lpips": _finite(candidate_overall.get("lpips"), "candidate lpips")
        - _finite(b0_overall.get("lpips"), "B0 lpips"),
        "hard_score50": _finite(candidate_hard.get("score50"), "candidate hard score50")
        - _finite(b0_hard.get("score50"), "B0 hard score50"),
        "missing_edge": _finite(
            candidate_overall.get("missing_edge"), "candidate missing_edge"
        )
        - _finite(b0_overall.get("missing_edge"), "B0 missing_edge"),
        "spurious_edge": _finite(
            candidate_overall.get("spurious_edge"), "candidate spurious_edge"
        )
        - _finite(b0_overall.get("spurious_edge"), "B0 spurious_edge"),
    }
    b0_time = _finite_nonnegative(
        b0_resources.get("total_time_seconds"), "B0 total_time_seconds"
    )
    if b0_time <= 0.0:
        raise ValueError("B0 total_time_seconds must be positive")
    time_ratio = _finite_nonnegative(
        candidate_resources.get("total_time_seconds"),
        "candidate total_time_seconds",
    ) / b0_time
    peak_vram = _finite_nonnegative(
        candidate_resources.get("max_vram_mb"), "candidate max_vram_mb"
    )
    peak_gaussians = _positive_int(
        candidate_resources.get("peak_gaussians"), "candidate peak_gaussians"
    )
    gates = {
        "score50_improved": deltas["score50"] > 0.0,
        "lpips_not_worse": deltas["lpips"] <= 0.0,
        "hard_score_not_worse": deltas["hard_score50"] >= 0.0,
        "detail_not_jointly_worse": not (
            deltas["missing_edge"] > 0.0 and deltas["spurious_edge"] > 0.0
        ),
        "time_budget": time_ratio <= 1.25,
        "vram_budget": peak_vram < 23 * 1024,
        "integrity_passed": integrity_passed,
        "primitive_growth_controlled": primitive_growth_controlled,
    }
    # Validate numeric boundaries through the shared locked contract as well.
    try:
        validate_paired_wall_time_ratio(time_ratio)
    except ValueError:
        gates["time_budget"] = False
    try:
        validate_peak_vram_mb(peak_vram)
    except ValueError:
        gates["vram_budget"] = False

    reversed_after_15k = False
    if result_15k is not None:
        prior = _mapping(result_15k, "result_15k")
        if prior.get("candidate_id") != candidate_id or prior.get("step") != 15_000:
            raise ValueError("result_15k identity does not match candidate")
        for field in ("scene_id", "manifest_sha256", "holdout_sha256"):
            if prior.get(field) != candidate.get(field):
                raise ValueError(f"result_15k {field} does not match candidate")
        prior_gates = _mapping(prior.get("gates"), "result_15k gates")
        prior_quality_passed = all(prior_gates.get(name) is True for name in _QUALITY_GATES)
        reversed_after_15k = bool(
            step == 30_000
            and prior_quality_passed
            and not all(gates[name] for name in _QUALITY_GATES)
        )

    all_gates = all(gates.values())
    screen_qualified = bool(step == 7_000 and all_gates)
    eligible = bool(step == 30_000 and all_gates and not reversed_after_15k)
    status = (
        "accepted"
        if eligible
        else "screen_passed"
        if screen_qualified
        else "pending_confirmation"
        if step == 15_000 and all_gates
        else "rejected"
    )
    return {
        "schema_version": 1,
        "scene_id": candidate["scene_id"],
        "candidate_id": candidate_id,
        "step": step,
        "manifest_sha256": candidate["manifest_sha256"],
        "holdout_sha256": candidate["holdout_sha256"],
        "config_sha256": candidate.get("config_sha256"),
        "deltas": deltas,
        "gates": gates,
        "paired_wall_time_ratio": time_ratio,
        "candidate_lpips": _finite(candidate_overall.get("lpips"), "candidate lpips"),
        "candidate_symmetric_edge_distance": _finite_nonnegative(
            candidate_overall.get("symmetric_edge_distance"),
            "candidate symmetric_edge_distance",
        ),
        "candidate_peak_gaussians": peak_gaussians,
        "reversed_after_15k": reversed_after_15k,
        "screen_qualified": screen_qualified,
        "eligible": eligible,
        "status": status,
    }


def select_scene_candidate(
    b0_report: Mapping[str, object],
    candidate_reports: Sequence[Mapping[str, object]],
    *,
    validation_flags: Mapping[str, tuple[bool, bool]] | None = None,
    results_15k: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, object]:
    if isinstance(candidate_reports, (str, bytes)) or not candidate_reports:
        raise ValueError("candidate_reports must be a non-empty sequence")
    evaluations = []
    seen: set[str] = set()
    for report in candidate_reports:
        candidate_id = _string(_report(report, "candidate_report").get("candidate_id"), "candidate_id")
        if candidate_id in seen:
            raise ValueError("candidate reports contain duplicate candidate_id")
        seen.add(candidate_id)
        flags = (validation_flags or {}).get(candidate_id, (True, True))
        evaluations.append(
            evaluate_candidate(
                b0_report,
                report,
                integrity_passed=flags[0],
                primitive_growth_controlled=flags[1],
                result_15k=(results_15k or {}).get(candidate_id),
            )
        )
    b0 = _report(b0_report, "b0_report")
    step = int(b0["step"])
    if step == 15_000:
        raise ValueError("15000-step evidence cannot create a scene selection")
    selection_field = "screen_qualified" if step == 7_000 else "eligible"
    eligible = [result for result in evaluations if result[selection_field] is True]
    selected = (
        min(
            eligible,
            key=lambda result: (
                -float(_mapping(result["deltas"], "deltas")["score50"]),
                float(result["candidate_lpips"]),
                float(result["candidate_symmetric_edge_distance"]),
                int(result["candidate_peak_gaussians"]),
                str(result["candidate_id"]),
            ),
        )
        if eligible
        else None
    )
    body: dict[str, object] = {
        "schema_version": 1,
        "scene_id": b0["scene_id"],
        "step": b0["step"],
        "decision_stage": "screen" if step == 7_000 else "confirmation",
        "manifest_sha256": b0["manifest_sha256"],
        "holdout_sha256": b0["holdout_sha256"],
        "selected_candidate_id": (
            selected["candidate_id"] if selected is not None else "B0-reference"
        ),
        "fallback_to_b0": selected is None,
        "evaluations": sorted(evaluations, key=lambda item: str(item["candidate_id"])),
    }
    body["decision_sha256"] = canonical_json_sha256(body)
    return body


def build_cohort_decision(
    scene_decisions: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if isinstance(scene_decisions, (str, bytes)) or len(scene_decisions) != len(
        COHORT_SCENE_IDS
    ):
        raise ValueError("cohort decision requires exactly seven scene decisions")
    by_scene: dict[str, Mapping[str, object]] = {}
    for raw in scene_decisions:
        decision = _mapping(raw, "scene decision")
        scene_id = _string(decision.get("scene_id"), "scene_id")
        if scene_id not in COHORT_SCENE_IDS or scene_id in by_scene:
            raise ValueError("scene decisions do not match the locked cohort")
        selected = _string(decision.get("selected_candidate_id"), "selected_candidate_id")
        candidate_settings(selected)
        if decision.get("decision_stage") != "confirmation" or decision.get("step") != 30_000:
            raise ValueError("cohort requires 30000-step confirmation decisions")
        digest = _string(decision.get("decision_sha256"), "decision_sha256")
        if _SHA256.fullmatch(digest) is None:
            raise ValueError("decision_sha256 must be a lowercase SHA-256 digest")
        unhashed = dict(decision)
        del unhashed["decision_sha256"]
        if canonical_json_sha256(unhashed) != digest:
            raise ValueError("scene decision SHA-256 does not match its contents")
        by_scene[scene_id] = decision
    if set(by_scene) != set(COHORT_SCENE_IDS):
        raise ValueError("scene decisions do not match the locked cohort")
    scenes = {
        scene_id: {
            "candidate_id": by_scene[scene_id]["selected_candidate_id"],
            "source_decision_sha256": by_scene[scene_id]["decision_sha256"],
        }
        for scene_id in COHORT_SCENE_IDS
    }
    body: dict[str, object] = {"schema_version": 1, "scenes": scenes}
    body["cohort_sha256"] = canonical_json_sha256(body)
    return body


def save_scene_decision(decision: Mapping[str, object], path: Path) -> str:
    return save_json_artifact(decision, path)


def save_cohort_decision(decision: Mapping[str, object], path: Path) -> str:
    return save_json_artifact(decision, path)


def _validate_pair(b0: Mapping[str, object], candidate: Mapping[str, object]) -> None:
    for field in _PAIR_FIELDS:
        if b0.get(field) != candidate.get(field):
            raise ValueError(f"paired report {field} does not match")


def _report(value: object, field: str) -> Mapping[str, object]:
    report = _mapping(value, field)
    for name in ("scene_id", "candidate_id", "manifest_sha256", "holdout_sha256"):
        _string(report.get(name), f"{field} {name}")
    _positive_int(report.get("step"), f"{field} step")
    return report


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be finite")
    return float(value)


def _finite_nonnegative(value: object, field: str) -> float:
    number = _finite(value, field)
    if number < 0.0:
        raise ValueError(f"{field} must be nonnegative")
    return number
