from __future__ import annotations

import math
from typing import Any

import torch

from bts_nvs.models.gaussian_parameters import (
    GaussianParameterMap,
    GaussianParameters,
)

try:
    from gsplat import DefaultStrategy
except ImportError:
    DefaultStrategy = None


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class GsplatStrategy:
    """Small, state-safe adapter for gsplat's default density strategy."""

    def __init__(
        self,
        gaussians: GaussianParameters,
        optimizers: dict[str, torch.optim.Optimizer],
        *,
        prune_opa: float = 0.005,
        grow_grad2d: float = 0.0002,
        grow_scale3d: float = 0.01,
        refine_start_step: int = 500,
        refine_stop_step: int = 15_000,
        refine_every: int = 100,
        reset_every: int = 3_000,
        absgrad: bool = False,
        revised_opacity: bool = False,
    ) -> None:
        if DefaultStrategy is None:
            raise ImportError("gsplat==1.4.0 is required for density control")
        for name, value in {
            "prune_opa": prune_opa,
            "grow_grad2d": grow_grad2d,
            "grow_scale3d": grow_scale3d,
        }.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if prune_opa >= 1.0:
            raise ValueError("prune_opa must be less than one")
        for name, value in {
            "refine_start_step": refine_start_step,
            "refine_stop_step": refine_stop_step,
            "refine_every": refine_every,
            "reset_every": reset_every,
        }.items():
            _positive_integer(value, name)
        if refine_stop_step <= refine_start_step:
            raise ValueError("refine_stop_step must be greater than refine_start_step")

        self.params: GaussianParameterMap = gaussians.parameter_map()
        self.optimizers = optimizers
        self.backend = DefaultStrategy(
            prune_opa=prune_opa,
            grow_grad2d=grow_grad2d,
            grow_scale3d=grow_scale3d,
            prune_scale3d=math.inf,
            refine_scale2d_stop_iter=0,
            refine_start_iter=refine_start_step - 1,
            refine_stop_iter=refine_stop_step,
            refine_every=refine_every,
            reset_every=reset_every,
            absgrad=absgrad,
            revised_opacity=revised_opacity,
        )
        self.backend.check_sanity(self.params, self.optimizers)

    def initialize_state(self, scene_scale: float = 1.0) -> dict[str, Any]:
        if not math.isfinite(scene_scale) or scene_scale <= 0.0:
            raise ValueError("scene_scale must be positive and finite")
        return self.backend.initialize_state(scene_scale=scene_scale)

    def step_pre_backward(
        self,
        state: dict[str, Any],
        step: int,
        info: dict[str, Any],
    ) -> None:
        _positive_integer(step, "step")
        self.backend.step_pre_backward(
            self.params,
            self.optimizers,
            state,
            step,
            info,
        )

    def step_post_backward(
        self,
        state: dict[str, Any],
        step: int,
        info: dict[str, Any],
        *,
        packed: bool = True,
    ) -> None:
        _positive_integer(step, "step")
        self.backend.step_post_backward(
            params=self.params,
            optimizers=self.optimizers,
            state=state,
            step=step,
            info=info,
            packed=packed,
        )
