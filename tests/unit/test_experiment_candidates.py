from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from bts_nvs.experiments.candidates import (
    CANDIDATE_IDS,
    candidate_settings,
    candidate_training_overrides,
)
from bts_nvs.experiments.contracts import CandidateSettings


def _settings(**overrides) -> CandidateSettings:
    values = {
        "candidate_id": "candidate-v1",
        "absgrad": False,
        "grow_grad2d": 0.0002,
        "grow_scale3d": 0.01,
        "prune_opa": 0.005,
        "refine_stop_step": 15_000,
        "rasterize_mode": "classic",
        "appearance_mode": "baseline",
        "sampling_mode": "uniform",
    }
    values.update(overrides)
    return CandidateSettings(**values)


def test_registry_locks_first_executable_candidates() -> None:
    assert CANDIDATE_IDS == (
        "B0-reference",
        "E1-density-absgrad-t04-v1",
        "E1-density-scale005-v1",
    )
    baseline = candidate_settings("B0-reference")
    absgrad = candidate_settings("E1-density-absgrad-t04-v1")
    scale = candidate_settings("E1-density-scale005-v1")

    assert absgrad == replace(
        baseline,
        candidate_id="E1-density-absgrad-t04-v1",
        absgrad=True,
        grow_grad2d=0.0004,
    )
    assert scale == replace(
        baseline,
        candidate_id="E1-density-scale005-v1",
        grow_scale3d=0.005,
    )


def test_candidate_settings_are_immutable_and_unknown_ids_fail() -> None:
    settings = candidate_settings("B0-reference")
    with pytest.raises(FrozenInstanceError):
        settings.absgrad = True
    with pytest.raises(ValueError, match="unknown candidate"):
        candidate_settings("C1-absgrad-t08-revopacity-v1")


@pytest.mark.parametrize("candidate_id", ["", "   ", None, True])
def test_candidate_lookup_rejects_invalid_ids(candidate_id) -> None:
    with pytest.raises(ValueError, match="candidate_id"):
        candidate_settings(candidate_id)


def test_training_overrides_are_complete_fresh_plain_values() -> None:
    overrides = candidate_training_overrides("E1-density-absgrad-t04-v1")
    assert overrides == {
        "candidate_id": "E1-density-absgrad-t04-v1",
        "absgrad": True,
        "grow_grad2d": 0.0004,
        "grow_scale3d": 0.01,
        "prune_opa": 0.005,
        "refine_stop_step": 15_000,
        "rasterize_mode": "classic",
        "appearance_mode": "baseline",
        "sampling_mode": "uniform",
    }

    overrides["grow_grad2d"] = 1.0
    assert candidate_training_overrides(
        "E1-density-absgrad-t04-v1"
    )["grow_grad2d"] == pytest.approx(0.0004)


@pytest.mark.parametrize(
    "field,value",
    [
        ("grow_grad2d", True),
        ("grow_scale3d", False),
        ("prune_opa", True),
        ("refine_stop_step", True),
        ("grow_grad2d", 0.0),
        ("grow_grad2d", -0.1),
        ("grow_grad2d", float("nan")),
        ("grow_scale3d", float("inf")),
        ("prune_opa", 0.0),
        ("prune_opa", 1.0),
        ("refine_stop_step", 0),
        ("refine_stop_step", 1.5),
    ],
)
def test_candidate_contract_rejects_invalid_numeric_fields(field, value) -> None:
    with pytest.raises(ValueError, match=field):
        _settings(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_id", ""),
        ("candidate_id", "   "),
        ("absgrad", 1),
        ("rasterize_mode", "ewa"),
        ("appearance_mode", "affine"),
        ("sampling_mode", "quality"),
    ],
)
def test_candidate_contract_rejects_invalid_identity_and_modes(field, value) -> None:
    with pytest.raises(ValueError, match=field):
        _settings(**{field: value})


@pytest.mark.parametrize("rasterize_mode", ["classic", "antialiased"])
def test_candidate_contract_accepts_known_rasterize_modes(rasterize_mode) -> None:
    assert _settings(rasterize_mode=rasterize_mode).rasterize_mode == rasterize_mode
