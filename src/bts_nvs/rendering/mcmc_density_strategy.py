from __future__ import annotations

import math
from typing import Any

import torch

from bts_nvs.models.gaussian_parameters import (
    GaussianParameterMap,
    GaussianParameters,
)

try:
    from gsplat import MCMCStrategy
except ImportError:
    MCMCStrategy = None


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _finite_nonnegative(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be finite and nonnegative")


class GsplatMCMCStrategy:
    """State-safe adapter for gsplat's MCMC density strategy."""

    def __init__(
        self,
        gaussians: GaussianParameters,
        optimizers: dict[str, torch.optim.Optimizer],
        *,
        cap_max: int,
        noise_lr: float = 500_000.0,
        refine_start_step: int = 500,
        refine_stop_step: int = 25_000,
        refine_every: int = 100,
        min_opacity: float = 0.005,
    ) -> None:
        if MCMCStrategy is None:
            raise ImportError("gsplat==1.4.0 with MCMCStrategy is required")
        for name, value in {
            "cap_max": cap_max,
            "refine_start_step": refine_start_step,
            "refine_stop_step": refine_stop_step,
            "refine_every": refine_every,
        }.items():
            _positive_integer(value, name)
        for name, value in {
            "noise_lr": noise_lr,
            "min_opacity": min_opacity,
        }.items():
            _finite_nonnegative(value, name)
        if cap_max < gaussians.num_gaussians:
            raise ValueError("cap_max must cover all initialized Gaussians")
        if refine_stop_step <= refine_start_step:
            raise ValueError("refine_stop_step must be greater than refine_start_step")
        if not 0.0 < min_opacity < 1.0:
            raise ValueError("min_opacity must be in (0, 1)")

        self.params: GaussianParameterMap = gaussians.parameter_map()
        self.optimizers = optimizers
        self.backend = MCMCStrategy(
            cap_max=cap_max,
            noise_lr=float(noise_lr),
            refine_start_iter=refine_start_step,
            refine_stop_iter=refine_stop_step,
            refine_every=refine_every,
            min_opacity=float(min_opacity),
        )
        self.backend.check_sanity(self.params, self.optimizers)

    def initialize_state(self, scene_scale: float = 1.0) -> dict[str, Any]:
        if not math.isfinite(scene_scale) or scene_scale <= 0.0:
            raise ValueError("scene_scale must be positive and finite")
        return self.backend.initialize_state()

    def step_pre_backward(
        self,
        state: dict[str, Any],
        step: int,
        info: dict[str, Any],
    ) -> None:
        del state
        _positive_integer(step, "step")
        means2d = info.get("means2d")
        if not isinstance(means2d, torch.Tensor):
            raise RuntimeError("MCMC requires projected means from the renderer")
        means2d.retain_grad()

    def step_post_backward(
        self,
        state: dict[str, Any],
        step: int,
        info: dict[str, Any],
        *,
        packed: bool = True,
    ) -> None:
        _positive_integer(step, "step")
        if not isinstance(packed, bool):
            raise ValueError("packed must be boolean")
        means_optimizer = self.optimizers.get("means")
        if means_optimizer is None or len(means_optimizer.param_groups) != 1:
            raise RuntimeError("MCMC requires one means optimizer parameter group")
        learning_rate = means_optimizer.param_groups[0].get("lr")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(float(learning_rate))
            or float(learning_rate) <= 0.0
        ):
            raise RuntimeError("MCMC requires a positive finite means learning rate")
        self.backend.step_post_backward(
            params=self.params,
            optimizers=self.optimizers,
            state=state,
            step=step,
            info=info,
            lr=float(learning_rate),
        )
