from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from bts_nvs.experiments.candidates import (
    CANDIDATE_IDS,
    candidate_settings,
    candidate_training_overrides,
)
from bts_nvs.experiments.contracts import CandidateSettings
from bts_nvs.experiments.density_policies import MCMC_CANDIDATE_ID


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
        "max_sh_degree": 3,
        "pixel_weight_mode": "uniform",
        "pixel_weight_floor": 0.5,
        "pixel_weight_patch_size": 31,
        "observation_mapping_mode": "legacy-ceil",
    }
    values.update(overrides)
    return CandidateSettings(**values)


def test_registry_locks_first_executable_candidates() -> None:
    assert CANDIDATE_IDS == (
        "B0-reference",
        "E1-density-absgrad-t04-v1",
        "E1-density-scale005-v1",
        "E2-raster-aa-v1",
        "E2-loss-local-laplacian-v1",
        "E2-appearance-sh4-v1",
        "E3-chair-observation-scale-v1",
        "E4-chair-observation-scale-absgrad-v1",
        MCMC_CANDIDATE_ID,
    )
    baseline = candidate_settings("B0-reference")
    absgrad = candidate_settings("E1-density-absgrad-t04-v1")
    scale = candidate_settings("E1-density-scale005-v1")
    antialiased = candidate_settings("E2-raster-aa-v1")
    weighted = candidate_settings("E2-loss-local-laplacian-v1")
    sh4 = candidate_settings("E2-appearance-sh4-v1")
    chair_mapping = candidate_settings("E3-chair-observation-scale-v1")
    chair_absgrad = candidate_settings(
        "E4-chair-observation-scale-absgrad-v1"
    )
    chair_mcmc = candidate_settings(MCMC_CANDIDATE_ID)

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
    assert antialiased == replace(
        baseline,
        candidate_id="E2-raster-aa-v1",
        rasterize_mode="antialiased",
    )
    assert weighted == replace(
        baseline,
        candidate_id="E2-loss-local-laplacian-v1",
        pixel_weight_mode="local-laplacian",
    )
    assert sh4 == replace(
        baseline,
        candidate_id="E2-appearance-sh4-v1",
        appearance_mode="sh4",
        max_sh_degree=4,
    )
    assert chair_mapping == replace(
        weighted,
        candidate_id="E3-chair-observation-scale-v1",
        observation_mapping_mode="continuous-reprojection",
    )
    assert chair_absgrad == replace(
        chair_mapping,
        candidate_id="E4-chair-observation-scale-absgrad-v1",
        absgrad=True,
        grow_grad2d=0.0004,
    )
    assert chair_mcmc == replace(
        chair_mapping,
        candidate_id=MCMC_CANDIDATE_ID,
        refine_stop_step=25_000,
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
        "max_sh_degree": 3,
        "pixel_weight_mode": "uniform",
        "pixel_weight_floor": 0.5,
        "pixel_weight_patch_size": 31,
        "observation_mapping_mode": "legacy-ceil",
    }

    overrides["grow_grad2d"] = 1.0
    assert candidate_training_overrides(
        "E1-density-absgrad-t04-v1"
    )["grow_grad2d"] == pytest.approx(0.0004)


def test_mcmc_candidate_adds_a_faithful_bounded_density_policy() -> None:
    overrides = candidate_training_overrides(MCMC_CANDIDATE_ID)

    assert overrides["density_strategy"] == "mcmc"
    assert overrides["mcmc_cap_max"] == 2_000_000
    assert overrides["mcmc_noise_lr"] == pytest.approx(500_000.0)
    assert overrides["mcmc_opacity_reg"] == pytest.approx(0.001)
    assert overrides["mcmc_scale_reg"] == pytest.approx(0.01)
    assert overrides["refine_stop_step"] == 25_000
    assert overrides["pixel_weight_mode"] == "local-laplacian"
    assert overrides["observation_mapping_mode"] == "continuous-reprojection"


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
        ("max_sh_degree", True),
        ("max_sh_degree", 2),
        ("max_sh_degree", 5),
        ("pixel_weight_floor", 0.0),
        ("pixel_weight_floor", 1.1),
        ("pixel_weight_patch_size", 2),
        ("pixel_weight_patch_size", 32),
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
        ("pixel_weight_mode", "gradient"),
        ("observation_mapping_mode", "affine"),
    ],
)
def test_candidate_contract_rejects_invalid_identity_and_modes(field, value) -> None:
    with pytest.raises(ValueError, match=field):
        _settings(**{field: value})


@pytest.mark.parametrize("rasterize_mode", ["classic", "antialiased"])
def test_candidate_contract_accepts_known_rasterize_modes(rasterize_mode) -> None:
    assert _settings(rasterize_mode=rasterize_mode).rasterize_mode == rasterize_mode


def test_candidate_contract_requires_consistent_sh_mode() -> None:
    with pytest.raises(ValueError, match="appearance_mode"):
        _settings(appearance_mode="sh4")
    with pytest.raises(ValueError, match="max_sh_degree"):
        _settings(max_sh_degree=4)

    settings = _settings(appearance_mode="sh4", max_sh_degree=4)
    assert settings.max_sh_degree == 4
