from __future__ import annotations

import pytest
import torch

from bts_nvs.models.density_regularization import DensityRegularizer
from bts_nvs.models.gaussian_parameters import GaussianParameters
from bts_nvs.models.optimizer import setup_optimizers
from bts_nvs.rendering import mcmc_density_strategy
from bts_nvs.rendering.density_strategy_factory import build_density_strategy
from bts_nvs.rendering.mcmc_density_strategy import GsplatMCMCStrategy


def _gaussians(count: int = 2) -> GaussianParameters:
    return GaussianParameters(
        means=torch.zeros((count, 3)),
        scales=torch.zeros((count, 3)),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count),
        opacities=torch.zeros(count),
        sh0=torch.zeros((count, 1, 3)),
        shN=torch.zeros((count, 15, 3)),
    )


class _FakeMCMCStrategy:
    def __init__(self, **kwargs) -> None:
        self.config = kwargs
        self.call = None

    def check_sanity(self, params, optimizers) -> None:
        assert params.keys() == optimizers.keys()

    def initialize_state(self):
        return {"binoms": torch.zeros((1, 1))}

    def step_post_backward(self, **kwargs) -> None:
        self.call = kwargs


def test_mcmc_adapter_forwards_locked_policy_and_current_means_lr(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        mcmc_density_strategy,
        "MCMCStrategy",
        _FakeMCMCStrategy,
    )
    gaussians = _gaussians()
    optimizers = setup_optimizers(gaussians)
    strategy = GsplatMCMCStrategy(
        gaussians,
        optimizers,
        cap_max=2_000_000,
    )

    assert strategy.backend.config == {
        "cap_max": 2_000_000,
        "noise_lr": 500_000.0,
        "refine_start_iter": 500,
        "refine_stop_iter": 25_000,
        "refine_every": 100,
        "min_opacity": 0.005,
    }
    state = strategy.initialize_state()
    means2d = torch.zeros((2, 2), requires_grad=True)
    strategy.step_pre_backward(state, 1, {"means2d": means2d})
    assert means2d.is_leaf or means2d.retains_grad
    strategy.step_post_backward(state, 1, {"means2d": means2d})

    assert strategy.backend.call["params"] is strategy.params
    assert strategy.backend.call["optimizers"] is optimizers
    assert strategy.backend.call["state"] is state
    assert strategy.backend.call["lr"] == pytest.approx(
        optimizers["means"].param_groups[0]["lr"]
    )


def test_density_strategy_factory_keeps_default_and_mcmc_separate(
    monkeypatch,
) -> None:
    observed = {}

    class _Built:
        def __init__(self, *args, **kwargs):
            observed.update(kwargs)

    monkeypatch.setattr(
        "bts_nvs.rendering.density_strategy_factory.GsplatMCMCStrategy",
        _Built,
    )
    gaussians = _gaussians()
    strategy = build_density_strategy(
        gaussians,
        setup_optimizers(gaussians),
        {
            "density_strategy": "mcmc",
            "mcmc_cap_max": 2_000_000,
            "mcmc_noise_lr": 500_000.0,
            "refine_stop_step": 25_000,
        },
    )

    assert isinstance(strategy, _Built)
    assert observed["cap_max"] == 2_000_000
    assert observed["refine_stop_step"] == 25_000


def test_mcmc_regularizer_matches_published_deep_blending_weights() -> None:
    regularizer = DensityRegularizer(
        {
            "density_strategy": "mcmc",
            "mcmc_opacity_reg": 0.001,
            "mcmc_scale_reg": 0.01,
        }
    )

    loss = regularizer(_gaussians())

    assert loss.item() == pytest.approx(0.0105)


def test_default_density_has_exact_zero_regularization() -> None:
    gaussians = _gaussians()
    loss = DensityRegularizer({})(gaussians)

    assert loss.shape == ()
    assert loss.item() == 0.0
    assert loss.device == gaussians.means.device


def test_perceptual_density_has_exact_zero_mcmc_regularization() -> None:
    gaussians = _gaussians()
    loss = DensityRegularizer({"density_strategy": "perceptual"})(gaussians)

    assert loss.shape == ()
    assert loss.item() == 0.0

    with pytest.raises(ValueError, match="requires density_strategy=mcmc"):
        DensityRegularizer(
            {
                "density_strategy": "perceptual",
                "mcmc_opacity_reg": 0.001,
            }
        )
