import copy

import pytest

from bts_nvs.experiments.decisions import (
    build_cohort_decision,
    evaluate_candidate,
    select_scene_candidate,
)
from bts_nvs.experiments.experiment import COHORT_SCENE_IDS
from bts_nvs.experiments.provenance import canonical_json_sha256


HASH = "a" * 64
HOLDOUT = "b" * 64


def _report(candidate_id: str, *, step: int = 30_000) -> dict:
    return {
        "schema_version": 1,
        "scene_id": "HCM0644",
        "candidate_id": candidate_id,
        "step": step,
        "manifest_sha256": HASH,
        "config_sha256": ("c" if candidate_id == "B0-reference" else "d") * 64,
        "holdout_sha256": HOLDOUT,
        "overall": {
            "score50": 70.0 if candidate_id == "B0-reference" else 71.0,
            "lpips": 0.20 if candidate_id == "B0-reference" else 0.18,
            "missing_edge": 0.20 if candidate_id == "B0-reference" else 0.18,
            "spurious_edge": 0.10 if candidate_id == "B0-reference" else 0.11,
            "symmetric_edge_distance": 0.20 if candidate_id == "B0-reference" else 0.18,
        },
        "strata": {
            "hard": {
                "score50": 65.0 if candidate_id == "B0-reference" else 65.5,
            }
        },
        "resources": {
            "total_time_seconds": 100.0 if candidate_id == "B0-reference" else 120.0,
            "max_vram_mb": 10_000.0,
            "peak_gaussians": 100 if candidate_id == "B0-reference" else 110,
            "final_num_gaussians": 90 if candidate_id == "B0-reference" else 100,
        },
    }


def test_eligible_candidate_passes_every_locked_gate() -> None:
    result = evaluate_candidate(_report("B0-reference"), _report("E1-density-absgrad-t04-v1"))

    assert result["eligible"] is True
    assert result["status"] == "accepted"
    assert all(result["gates"].values())
    assert result["deltas"]["score50"] == pytest.approx(1.0)
    assert result["paired_wall_time_ratio"] == pytest.approx(1.2)


@pytest.mark.parametrize(
    ("mutation", "gate"),
    [
        (lambda r: r["overall"].update(score50=70.0), "score50_improved"),
        (lambda r: r["overall"].update(lpips=0.21), "lpips_not_worse"),
        (lambda r: r["strata"]["hard"].update(score50=64.9), "hard_score_not_worse"),
        (
            lambda r: r["overall"].update(missing_edge=0.21, spurious_edge=0.11),
            "detail_not_jointly_worse",
        ),
        (lambda r: r["resources"].update(total_time_seconds=126.0), "time_budget"),
        (lambda r: r["resources"].update(max_vram_mb=23 * 1024), "vram_budget"),
    ],
)
def test_each_locked_gate_rejects_candidate(mutation, gate: str) -> None:
    candidate = _report("E1-density-absgrad-t04-v1")
    mutation(candidate)

    result = evaluate_candidate(_report("B0-reference"), candidate)

    assert result["eligible"] is False
    assert result["gates"][gate] is False


def test_integrity_growth_and_15k_cannot_accept() -> None:
    b0 = _report("B0-reference", step=15_000)
    candidate = _report("E1-density-absgrad-t04-v1", step=15_000)
    pending = evaluate_candidate(b0, candidate)
    invalid = evaluate_candidate(
        _report("B0-reference"),
        _report("E1-density-absgrad-t04-v1"),
        integrity_passed=False,
        primitive_growth_controlled=False,
    )

    assert pending["eligible"] is False
    assert pending["status"] == "pending_confirmation"
    assert invalid["gates"]["integrity_passed"] is False
    assert invalid["gates"]["primitive_growth_controlled"] is False


def test_pair_identity_and_nonfinite_values_are_rejected() -> None:
    candidate = _report("E1-density-absgrad-t04-v1")
    candidate["holdout_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="holdout"):
        evaluate_candidate(_report("B0-reference"), candidate)

    candidate = _report("E1-density-absgrad-t04-v1")
    candidate["overall"]["score50"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        evaluate_candidate(_report("B0-reference"), candidate)


def test_30k_records_reversal_from_a_previously_passing_15k_result() -> None:
    at_15k = evaluate_candidate(
        _report("B0-reference", step=15_000),
        _report("E1-density-absgrad-t04-v1", step=15_000),
    )
    final = _report("E1-density-absgrad-t04-v1")
    final["overall"]["score50"] = 69.9

    result = evaluate_candidate(_report("B0-reference"), final, result_15k=at_15k)

    assert result["eligible"] is False
    assert result["reversed_after_15k"] is True


def test_15k_result_must_match_the_same_scene_and_provenance() -> None:
    at_15k = evaluate_candidate(
        _report("B0-reference", step=15_000),
        _report("E1-density-absgrad-t04-v1", step=15_000),
    )
    at_15k["holdout_sha256"] = "e" * 64

    with pytest.raises(ValueError, match="result_15k holdout"):
        evaluate_candidate(
            _report("B0-reference"),
            _report("E1-density-absgrad-t04-v1"),
            result_15k=at_15k,
        )


def test_scene_selection_uses_exact_tie_break_order_and_b0_fallback() -> None:
    b0 = _report("B0-reference")
    first = _report("E1-density-absgrad-t04-v1")
    second = _report("E1-density-scale005-v1")
    second["overall"].update(score50=71.0, lpips=0.17, symmetric_edge_distance=0.19)

    selected = select_scene_candidate(b0, [first, second])
    rejected = copy.deepcopy(first)
    rejected["overall"]["score50"] = 69.0
    fallback = select_scene_candidate(b0, [rejected])

    assert selected["selected_candidate_id"] == "E1-density-scale005-v1"
    assert fallback["selected_candidate_id"] == "B0-reference"
    assert fallback["fallback_to_b0"] is True


def test_cohort_decision_requires_exact_locked_scenes_and_isolated_choices() -> None:
    decisions = []
    for index, scene_id in enumerate(COHORT_SCENE_IDS):
        decision = {
            "schema_version": 1,
            "scene_id": scene_id,
            "selected_candidate_id": (
                "E1-density-absgrad-t04-v1" if index == 0 else "B0-reference"
            ),
        }
        decision["decision_sha256"] = canonical_json_sha256(decision)
        decisions.append(decision)

    cohort = build_cohort_decision(decisions)

    assert tuple(cohort["scenes"]) == COHORT_SCENE_IDS
    assert cohort["scenes"]["HCM0644"]["candidate_id"] == "E1-density-absgrad-t04-v1"
    assert cohort["scenes"]["chair"]["candidate_id"] == "B0-reference"

    with pytest.raises(ValueError, match="seven|cohort"):
        build_cohort_decision(decisions[:-1])

    decisions[0]["selected_candidate_id"] = "B0-reference"
    with pytest.raises(ValueError, match="SHA-256"):
        build_cohort_decision(decisions)
