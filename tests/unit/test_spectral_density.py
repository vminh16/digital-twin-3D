from __future__ import annotations

import pytest
import torch

from bts_nvs.experiments.candidates import candidate_training_overrides
from bts_nvs.experiments.spectral_policy import SPECTRAL_CANDIDATE_ID
from bts_nvs.models.gaussian_parameters import GaussianParameters
from bts_nvs.models.optimizer import setup_optimizers
from bts_nvs.rendering.spectral_density_strategy import GsplatSpectralStrategy
from bts_nvs.rendering.density_strategy_factory import build_density_strategy
from bts_nvs.rendering.spectral_math import spectral_entropy_and_condition


def _gaussians(scales: torch.Tensor) -> GaussianParameters:
    count = len(scales)
    return GaussianParameters(
        means=torch.zeros((count, 3)),
        scales=torch.log(scales),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count),
        opacities=torch.zeros(count),
        sh0=torch.zeros((count, 1, 3)),
        shN=torch.zeros((count, 15, 3)),
    )


def _prime_optimizers(optimizers) -> None:
    for optimizer in optimizers.values():
        parameter = optimizer.param_groups[0]["params"][0]
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def test_spectral_candidate_keeps_e3_base_and_locks_policy() -> None:
    overrides = candidate_training_overrides(SPECTRAL_CANDIDATE_ID)

    assert overrides["density_strategy"] == "spectral"
    assert overrides["spectral_cap_max"] == 2_100_000
    assert overrides["spectral_entropy_threshold"] == pytest.approx(0.5)
    assert overrides["spectral_split_k"] == pytest.approx(0.6)
    assert overrides["spectral_split_k0"] == pytest.approx(1.0)
    assert overrides["pixel_weight_mode"] == "local-laplacian"
    assert overrides["observation_mapping_mode"] == "continuous-reprojection"


def test_density_factory_builds_spectral_strategy() -> None:
    gaussians = _gaussians(torch.tensor([[0.01, 0.01, 0.01]]))
    strategy = build_density_strategy(
        gaussians,
        setup_optimizers(gaussians),
        candidate_training_overrides(SPECTRAL_CANDIDATE_ID),
    )

    assert isinstance(strategy, GsplatSpectralStrategy)


def test_spectral_strategy_preserves_adc_and_splits_low_entropy_gaussian() -> None:
    gaussians = _gaussians(
        torch.tensor(
            [
                [0.005, 0.005, 0.005],
                [0.02, 0.02, 0.02],
                [0.20, 0.01, 0.01],
            ]
        )
    )
    optimizers = setup_optimizers(gaussians)
    _prime_optimizers(optimizers)
    strategy = GsplatSpectralStrategy(
        gaussians,
        optimizers,
        cap_max=16,
        entropy_threshold=0.5,
        split_k=0.6,
        split_k0=1.0,
    )
    state = strategy.initialize_state()
    state["grad2d"] = torch.tensor([0.001, 0.001, 0.0])
    state["count"] = torch.ones(3)

    cloned, split_count = strategy.backend._grow_gs(
        strategy.params,
        optimizers,
        state,
        step=500,
    )

    assert (cloned, split_count) == (1, 2)
    assert state["adc_clone_count"] == 1
    assert state["adc_split_count"] == 1
    assert state["spectral_split_count"] == 1
    assert state["spectral_child_condition_violations"] == 0
    assert gaussians.num_gaussians == 6
    assert all(parameter.shape[0] == 6 for parameter in strategy.params.values())
    spectral_children = gaussians.get_scales()[[-3, -1]]
    _, child_condition = spectral_entropy_and_condition(spectral_children)
    assert torch.all(child_condition < 400.0)


def test_spectral_strategy_enforces_population_cap() -> None:
    gaussians = _gaussians(
        torch.tensor([[0.20, 0.01, 0.01], [0.15, 0.01, 0.01]])
    )
    optimizers = setup_optimizers(gaussians)
    _prime_optimizers(optimizers)
    strategy = GsplatSpectralStrategy(
        gaussians,
        optimizers,
        cap_max=3,
        entropy_threshold=0.5,
        split_k=0.6,
        split_k0=1.0,
    )
    state = strategy.initialize_state()
    state["grad2d"] = torch.zeros(2)
    state["count"] = torch.ones(2)

    strategy.backend._grow_gs(strategy.params, optimizers, state, step=500)

    assert gaussians.num_gaussians == 3
    assert state["spectral_split_count"] == 1
    assert state["spectral_cap_hit_step"] == 500
