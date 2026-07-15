from __future__ import annotations

import math

import pytest
import torch

from bts_nvs.models.gaussian_parameters import GaussianParameters
from bts_nvs.models.optimizer import setup_optimizers
from bts_nvs.rendering import density_strategy
from bts_nvs.rendering.density_strategy import GsplatStrategy


def _gaussians(
    *,
    scales: torch.Tensor | None = None,
    opacities: torch.Tensor | None = None,
) -> GaussianParameters:
    count = 2
    return GaussianParameters(
        means=torch.zeros((count, 3)),
        scales=scales if scales is not None else torch.zeros((count, 3)),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * count),
        opacities=opacities if opacities is not None else torch.zeros(count),
        sh0=torch.zeros((count, 1, 3)),
        shN=torch.zeros((count, 15, 3)),
    )


class _FakeDefaultStrategy:
    def __init__(self, **kwargs) -> None:
        self.config = kwargs

    def check_sanity(self, params, optimizers) -> None:
        assert params.keys() == optimizers.keys()

    def initialize_state(self, scene_scale: float = 1.0):
        return {"grad2d": None, "count": None, "scene_scale": scene_scale}

    def step_pre_backward(self, params, optimizers, state, step, info) -> None:
        info["means2d"].retain_grad()

    def step_post_backward(
        self, params, optimizers, state, step, info, packed=False
    ) -> None:
        if step == 500:
            for name in params:
                old = params[name]
                new = torch.nn.Parameter(torch.cat((old, old[:1])))
                optimizers[name].param_groups[0]["params"] = [new]
                params[name] = new


def test_parameter_mapping_replacement_updates_module_and_state_dict() -> None:
    gaussians = _gaussians()
    params = gaussians.parameter_map()
    replacement = torch.nn.Parameter(torch.zeros((3, 3)))

    params["means"] = replacement

    assert gaussians.means is replacement
    assert gaussians.get_means() is replacement
    assert gaussians.num_gaussians == 3
    assert gaussians.state_dict()["means"].shape == (3, 3)


def test_strategy_fails_fast_without_gsplat(monkeypatch) -> None:
    monkeypatch.setattr(density_strategy, "DefaultStrategy", None)
    gaussians = _gaussians()

    with pytest.raises(ImportError, match="gsplat==1.4.0"):
        GsplatStrategy(gaussians, setup_optimizers(gaussians))


def test_strategy_owns_canonical_parameters_and_explicit_config(monkeypatch) -> None:
    monkeypatch.setattr(density_strategy, "DefaultStrategy", _FakeDefaultStrategy)
    gaussians = _gaussians()
    optimizers = setup_optimizers(gaussians)
    strategy = GsplatStrategy(gaussians, optimizers)

    assert strategy.params.keys() == optimizers.keys()
    assert strategy.backend.config == {
        "prune_opa": 0.005,
        "grow_grad2d": 0.0002,
        "grow_scale3d": 0.01,
        "prune_scale3d": math.inf,
        "refine_scale2d_stop_iter": 0,
        "refine_start_iter": 499,
        "refine_stop_iter": 15_000,
        "refine_every": 100,
        "reset_every": 3_000,
        "absgrad": False,
    }

    state = strategy.initialize_state(scene_scale=1.0)
    means2d = torch.zeros((2, 2), requires_grad=True)
    info = {"means2d": means2d}
    strategy.step_pre_backward(state, step=500, info=info)
    means2d.sum().backward()
    strategy.step_post_backward(state, step=500, info=info, packed=True)

    assert gaussians.num_gaussians == 3
    assert strategy.params["means"] is gaussians.means
    assert optimizers["means"].param_groups[0]["params"] == [gaussians.means]
    assert gaussians.state_dict()["means"].shape == (3, 3)


@pytest.mark.parametrize("step", [0, -1, True])
def test_strategy_rejects_non_positive_one_based_step(monkeypatch, step) -> None:
    monkeypatch.setattr(density_strategy, "DefaultStrategy", _FakeDefaultStrategy)
    gaussians = _gaussians()
    strategy = GsplatStrategy(gaussians, setup_optimizers(gaussians))
    state = strategy.initialize_state()

    with pytest.raises(ValueError, match="positive integer"):
        strategy.step_pre_backward(
            state,
            step=step,
            info={"means2d": torch.zeros((2, 2), requires_grad=True)},
        )


@pytest.mark.parametrize("scene_scale", [0.0, -1.0, math.inf, math.nan])
def test_strategy_rejects_invalid_scene_scale(monkeypatch, scene_scale) -> None:
    monkeypatch.setattr(density_strategy, "DefaultStrategy", _FakeDefaultStrategy)
    gaussians = _gaussians()
    strategy = GsplatStrategy(gaussians, setup_optimizers(gaussians))

    with pytest.raises(ValueError, match="scene_scale"):
        strategy.initialize_state(scene_scale=scene_scale)


@pytest.mark.skipif(
    getattr(density_strategy, "DefaultStrategy", None) is None,
    reason="requires gsplat==1.4.0",
)
def test_real_strategy_duplicates_and_splits_with_canonical_mapping() -> None:
    scales = torch.log(torch.tensor([[0.005, 0.005, 0.005], [0.02, 0.02, 0.02]]))
    gaussians = _gaussians(scales=scales)
    optimizers = setup_optimizers(gaussians)
    for optimizer in optimizers.values():
        parameter = optimizer.param_groups[0]["params"][0]
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    strategy = GsplatStrategy(gaussians, optimizers)
    state = {
        "grad2d": torch.full((2,), 0.001),
        "count": torch.ones(2),
        "scene_scale": 1.0,
    }

    duplicated, split = strategy.backend._grow_gs(
        strategy.params, optimizers, state, step=500
    )

    assert (duplicated, split) == (1, 1)
    assert gaussians.num_gaussians == 4
    assert all(parameter.shape[0] == 4 for parameter in strategy.params.values())


@pytest.mark.skipif(
    getattr(density_strategy, "DefaultStrategy", None) is None,
    reason="requires gsplat==1.4.0",
)
def test_real_strategy_prunes_only_low_opacity() -> None:
    gaussians = _gaussians(
        scales=torch.log(torch.full((2, 3), 0.2)),
        opacities=torch.logit(torch.tensor([0.001, 0.5])),
    )
    optimizers = setup_optimizers(gaussians)
    for optimizer in optimizers.values():
        parameter = optimizer.param_groups[0]["params"][0]
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    strategy = GsplatStrategy(gaussians, optimizers)
    state = {
        "grad2d": torch.zeros(2),
        "count": torch.ones(2),
        "scene_scale": 1.0,
    }

    pruned = strategy.backend._prune_gs(strategy.params, optimizers, state, step=3_100)

    assert pruned == 1
    assert gaussians.num_gaussians == 1
