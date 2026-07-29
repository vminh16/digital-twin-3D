from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from bts_nvs.evaluation.perceptual_diagnostics import (
    build_perceptual_diagnostics,
)
from bts_nvs.models.gaussian_parameters import GaussianParameters
from bts_nvs.models.optimizer import setup_optimizers
from bts_nvs.rendering.perceptual_adc_strategy import (
    GsplatPerceptualADCStrategy,
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
    return GsplatPerceptualADCStrategy(
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


def _state(strategy, gradients, contribution=None):
    state = strategy.initialize_state()
    state["grad2d"] = torch.tensor(gradients)
    state["count"] = torch.ones(len(gradients))
    state["event_contribution"] = (
        None if contribution is None else torch.tensor(contribution)
    )
    return state


def test_adc_still_grows_on_non_perceptual_refine_step() -> None:
    gaussians = _gaussians()
    optimizers = setup_optimizers(gaussians)
    _prime(optimizers)
    strategy = _strategy(gaussians, optimizers)
    state = _state(strategy, [0.001, 0.0, 0.0])

    cloned, split_count = strategy.backend._grow_gs(
        strategy.params,
        optimizers,
        state,
        step=700,
    )

    assert (cloned, split_count) == (1, 0)
    assert gaussians.num_gaussians == 4
    assert state["adc_clone_count"] == 1
    assert state["perceptual_clone_count"] == 0


def test_adc_still_splits_large_gaussian_on_non_perceptual_refine_step() -> None:
    gaussians = _gaussians()
    gaussians.scales.data[0] = torch.log(torch.full((3,), 0.02))
    optimizers = setup_optimizers(gaussians)
    _prime(optimizers)
    strategy = _strategy(gaussians, optimizers)
    state = _state(strategy, [0.001, 0.0, 0.0])

    cloned, split_count = strategy.backend._grow_gs(
        strategy.params,
        optimizers,
        state,
        step=700,
    )

    assert (cloned, split_count) == (0, 1)
    assert gaussians.num_gaussians == 4
    assert state["adc_split_count"] == 1
    assert state["perceptual_split_count"] == 0


def test_high_sensitivity_adds_low_gradient_clone_at_event() -> None:
    gaussians = _gaussians()
    gaussians.sensitivity_logits.data = torch.logit(
        torch.tensor([0.95, 0.2, 0.2])
    )
    optimizers = setup_optimizers(gaussians)
    _prime(optimizers)
    strategy = _strategy(gaussians, optimizers)
    state = _state(
        strategy,
        [0.0, 0.0, 0.0],
        contribution=[30.0, 30.0, 30.0],
    )

    cloned, split_count = strategy.backend._grow_gs(
        strategy.params,
        optimizers,
        state,
        step=1_000,
    )

    assert (cloned, split_count) == (1, 0)
    assert state["adc_clone_count"] == 0
    assert state["perceptual_clone_count"] == 1
    assert state["event_contribution"] is None


def test_medium_sensitivity_adds_low_gradient_split_at_event() -> None:
    gaussians = _gaussians()
    gaussians.sensitivity_logits.data = torch.logit(
        torch.tensor([0.5, 0.2, 0.2])
    )
    optimizers = setup_optimizers(gaussians)
    _prime(optimizers)
    strategy = _strategy(gaussians, optimizers)
    state = _state(
        strategy,
        [0.0, 0.0, 0.0],
        contribution=[12.0, 12.0, 12.0],
    )

    cloned, split_count = strategy.backend._grow_gs(
        strategy.params,
        optimizers,
        state,
        step=1_500,
    )

    assert (cloned, split_count) == (0, 1)
    assert gaussians.num_gaussians == 4
    assert state["perceptual_split_count"] == 1


def test_perceptual_event_requires_full_view_contribution() -> None:
    gaussians = _gaussians()
    optimizers = setup_optimizers(gaussians)
    strategy = _strategy(gaussians, optimizers)
    state = _state(strategy, [0.0, 0.0, 0.0])

    with pytest.raises(RuntimeError, match="full-view contribution"):
        strategy.backend._grow_gs(
            strategy.params,
            optimizers,
            state,
            step=1_000,
        )


def test_adc_has_first_claim_on_hard_cap() -> None:
    gaussians = _gaussians()
    gaussians.sensitivity_logits.data = torch.logit(
        torch.tensor([0.2, 0.95, 0.2])
    )
    optimizers = setup_optimizers(gaussians)
    _prime(optimizers)
    strategy = _strategy(gaussians, optimizers, cap_max=4)
    state = _state(
        strategy,
        [0.001, 0.0, 0.0],
        contribution=[0.0, 50.0, 0.0],
    )

    strategy.backend._grow_gs(
        strategy.params,
        optimizers,
        state,
        step=1_000,
    )

    assert gaussians.num_gaussians == 4
    assert state["adc_clone_count"] == 1
    assert state["perceptual_clone_count"] == 0
    assert state["perceptual_cap_hit_step"] == 1_000


def test_corrected_diagnostics_separate_adc_and_perceptual_growth(
    tmp_path,
) -> None:
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps({"sensitivity_loss": 0.4}) + "\n",
        encoding="utf-8",
    )
    trainer = SimpleNamespace(
        gaussians=_gaussians(),
        config={
            "candidate_id": "E7-chair-perceptual-adc-corrected-v1",
            "perceptual_low_threshold": 0.3,
            "perceptual_high_threshold": 0.9,
            "perceptual_sensitivity_sha256": "a" * 64,
            "perceptual_scene_sensitivity": 0.5,
            "perceptual_cap_max": 2_100_000,
        },
        strategy_state={
            "adc_clone_count": 7,
            "adc_split_count": 5,
            "perceptual_clone_count": 3,
            "perceptual_split_count": 2,
            "perceptual_cap_hit_step": None,
        },
    )

    density = build_perceptual_diagnostics(trainer, tmp_path)["density"]

    assert density["adc_clone_count"] == 7
    assert density["adc_split_count"] == 5
    assert density["clone_count"] == 3
    assert density["split_count"] == 2
