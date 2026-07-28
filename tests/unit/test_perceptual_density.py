from __future__ import annotations

import pytest
import torch

from bts_nvs.experiments.perceptual_policy import (
    PERCEPTUAL_CANDIDATE_ID,
    PerceptualDensityPolicy,
    perceptual_policy_overrides,
)
from bts_nvs.models.gaussian_parameters import GaussianParameters
from bts_nvs.models.optimizer import setup_optimizers
from bts_nvs.rendering.perceptual_density_strategy import (
    GsplatPerceptualStrategy,
)


def _gaussians(count: int = 3) -> GaussianParameters:
    gaussians = GaussianParameters(
        means=torch.zeros((count, 3)),
        scales=torch.log(torch.full((count, 3), 0.005)),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count),
        opacities=torch.logit(torch.full((count,), 0.5)),
        sh0=torch.zeros((count, 1, 3)),
        shN=torch.zeros((count, 15, 3)),
    )
    gaussians.enable_perceptual_sensitivity()
    return gaussians


def _prime(optimizers) -> None:
    for optimizer in optimizers.values():
        parameter = optimizer.param_groups[0]["params"][0]
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def _strategy(gaussians, optimizers, *, cap_max=10):
    return GsplatPerceptualStrategy(
        gaussians,
        optimizers,
        cap_max=cap_max,
        high_threshold=0.9,
        low_threshold=0.3,
        high_interval=1_000,
        medium_interval=1_500,
        high_contribution=25.0,
        medium_contribution=10.0,
        opacity_exponent=1.2,
        scene_sensitivity=0.5,
    )


def test_policy_matches_locked_e6_values() -> None:
    overrides = perceptual_policy_overrides(PERCEPTUAL_CANDIDATE_ID)

    assert overrides == PerceptualDensityPolicy().training_overrides()
    assert overrides["perceptual_cap_max"] == 2_100_000
    assert overrides["perceptual_loss_weight"] == pytest.approx(0.1)


def test_optional_sensitivity_is_part_of_topology_and_optimizer_contract() -> None:
    gaussians = _gaussians()
    optimizers = setup_optimizers(gaussians)

    assert tuple(gaussians.parameter_map())[-1] == "sensitivity_logits"
    assert optimizers["sensitivity_logits"].param_groups[0]["lr"] == pytest.approx(
        0.05
    )
    assert torch.all(gaussians.get_sensitivities() == 0.5)


def test_perceptual_selector_clones_only_eligible_medium_gaussian() -> None:
    gaussians = _gaussians()
    gaussians.sensitivity_logits.data = torch.logit(
        torch.tensor([0.5, 0.2, 0.5])
    )
    optimizers = setup_optimizers(gaussians)
    _prime(optimizers)
    strategy = _strategy(gaussians, optimizers)
    state = strategy.initialize_state()
    state.update(
        {
            "grad2d": torch.full((3,), 0.001),
            "count": torch.ones(3),
            "high_contribution": torch.zeros(3),
            "medium_contribution": torch.tensor([12.0, 30.0, 5.0]),
        }
    )

    cloned, split_count = strategy.backend._grow_gs(
        strategy.params,
        optimizers,
        state,
        step=1_500,
    )

    assert (cloned, split_count) == (1, 0)
    assert gaussians.num_gaussians == 4
    assert gaussians.sensitivity_logits.shape == (4,)
    original_alpha = torch.sigmoid(gaussians.opacities[[0, 3]])
    expected = 1.0 - torch.sqrt(1.0 - torch.tensor(0.5).pow(1.2))
    torch.testing.assert_close(original_alpha, expected.expand(2))


def test_perceptual_selector_enforces_hard_cap_by_contribution() -> None:
    gaussians = _gaussians()
    gaussians.sensitivity_logits.data.fill_(torch.logit(torch.tensor(0.95)))
    optimizers = setup_optimizers(gaussians)
    _prime(optimizers)
    strategy = _strategy(gaussians, optimizers, cap_max=4)
    state = strategy.initialize_state()
    state.update(
        {
            "grad2d": torch.full((3,), 0.001),
            "count": torch.ones(3),
            "high_contribution": torch.tensor([30.0, 50.0, 40.0]),
            "medium_contribution": torch.zeros(3),
        }
    )

    strategy.backend._grow_gs(
        strategy.params,
        optimizers,
        state,
        step=1_000,
    )

    assert gaussians.num_gaussians == 4
    assert state["perceptual_cap_hit_step"] == 1_000
