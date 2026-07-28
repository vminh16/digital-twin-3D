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
    from gsplat.strategy.ops import duplicate, split
except ImportError:
    DefaultStrategy = None
    duplicate = split = None


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class _PerceptualBackend(DefaultStrategy if DefaultStrategy is not None else object):
    def __init__(
        self,
        *,
        cap_max: int,
        high_threshold: float,
        low_threshold: float,
        high_interval: int,
        medium_interval: int,
        high_contribution: float,
        medium_contribution: float,
        opacity_exponent: float,
        split_only: bool,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cap_max = cap_max
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.high_interval = high_interval
        self.medium_interval = medium_interval
        self.high_contribution = high_contribution
        self.medium_contribution = medium_contribution
        self.opacity_exponent = opacity_exponent
        self.split_only = split_only

    def initialize_state(self, scene_scale: float = 1.0) -> dict[str, Any]:
        state = super().initialize_state(scene_scale)
        state.update(
            {
                "high_contribution": None,
                "medium_contribution": None,
                "perceptual_clone_count": 0,
                "perceptual_split_count": 0,
                "perceptual_cap_hit_step": None,
            }
        )
        return state

    def _update_state(self, params, state, info, packed=False) -> None:
        super()._update_state(params, state, info, packed=packed)
        contribution = info.get("perceptual_contribution")
        if not isinstance(contribution, torch.Tensor):
            raise RuntimeError("perceptual strategy requires contribution weights")
        if contribution.shape != (len(params["means"]),):
            raise RuntimeError("perceptual contribution shape does not match Gaussians")
        for key in ("high_contribution", "medium_contribution"):
            if state[key] is None:
                state[key] = torch.zeros_like(contribution)
            state[key] = torch.maximum(state[key], contribution)

    @torch.no_grad()
    def _grow_gs(self, params, optimizers, state, step):
        high_event = step % self.high_interval == 0
        medium_event = step % self.medium_interval == 0
        if not high_event and not medium_event:
            return 0, 0

        gradients = state["grad2d"] / state["count"].clamp_min(1)
        sensitivity = torch.sigmoid(params["sensitivity_logits"])
        selected = torch.zeros_like(sensitivity, dtype=torch.bool)
        priority = torch.zeros_like(sensitivity)
        if high_event:
            high = (
                (sensitivity > self.high_threshold)
                & (state["high_contribution"] >= self.high_contribution)
            )
            selected |= high
            priority = torch.maximum(priority, state["high_contribution"])
        if medium_event:
            medium = (
                (sensitivity > self.low_threshold)
                & (sensitivity <= self.high_threshold)
                & (state["medium_contribution"] >= self.medium_contribution)
            )
            selected |= medium
            priority = torch.maximum(priority, state["medium_contribution"])
        selected &= gradients > self.grow_grad2d

        available = self.cap_max - len(params["means"])
        if available <= 0:
            state["perceptual_cap_hit_step"] = (
                state["perceptual_cap_hit_step"] or step
            )
            selected.zero_()
        elif int(selected.sum()) > available:
            indices = torch.where(selected)[0]
            keep = indices[torch.topk(priority[indices], available).indices]
            selected.zero_()
            selected[keep] = True
            state["perceptual_cap_hit_step"] = (
                state["perceptual_cap_hit_step"] or step
            )

        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_clone = selected & is_small & (not self.split_only)
        is_split = selected & ~is_clone
        clone_count = int(is_clone.sum())
        split_count = int(is_split.sum())

        if clone_count:
            self._duplicate_with_opacity_decline(
                params,
                optimizers,
                state,
                is_clone,
            )
        is_split = torch.cat(
            (
                is_split,
                torch.zeros(
                    clone_count,
                    dtype=torch.bool,
                    device=is_split.device,
                ),
            )
        )
        if split_count:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                revised_opacity=False,
            )

        state["perceptual_clone_count"] += clone_count
        state["perceptual_split_count"] += split_count
        if high_event:
            state["high_contribution"].zero_()
        if medium_event:
            state["medium_contribution"].zero_()
        return clone_count, split_count

    @torch.no_grad()
    def _duplicate_with_opacity_decline(
        self,
        params,
        optimizers,
        state,
        mask: torch.Tensor,
    ) -> None:
        original_count = len(params["means"])
        selected = torch.where(mask)[0]
        original_alpha = torch.sigmoid(params["opacities"][selected])
        target_alpha = original_alpha.pow(self.opacity_exponent)
        child_alpha = 1.0 - torch.sqrt(1.0 - target_alpha)
        child_logits = torch.logit(child_alpha.clamp(1e-6, 1.0 - 1e-6))
        duplicate(params=params, optimizers=optimizers, state=state, mask=mask)
        params["opacities"].data[selected] = child_logits
        params["opacities"].data[
            original_count : original_count + len(selected)
        ] = child_logits


class GsplatPerceptualStrategy:
    """Perceptual-GS density adapter using gsplat topology operations."""

    def __init__(
        self,
        gaussians: GaussianParameters,
        optimizers: dict[str, torch.optim.Optimizer],
        *,
        cap_max: int,
        high_threshold: float,
        low_threshold: float,
        high_interval: int,
        medium_interval: int,
        high_contribution: float,
        medium_contribution: float,
        opacity_exponent: float,
        scene_sensitivity: float,
        prune_opa: float = 0.005,
        grow_grad2d: float = 0.0002,
        grow_scale3d: float = 0.01,
        refine_start_step: int = 500,
        refine_stop_step: int = 15_000,
        refine_every: int = 100,
        reset_every: int = 3_000,
    ) -> None:
        if DefaultStrategy is None or duplicate is None or split is None:
            raise ImportError("gsplat==1.4.0 is required for perceptual density")
        if not hasattr(gaussians, "sensitivity_logits"):
            raise ValueError("perceptual strategy requires sensitivity logits")
        for name, value in {
            "cap_max": cap_max,
            "high_interval": high_interval,
            "medium_interval": medium_interval,
            "refine_start_step": refine_start_step,
            "refine_stop_step": refine_stop_step,
            "refine_every": refine_every,
            "reset_every": reset_every,
        }.items():
            _positive_integer(value, name)
        if cap_max < gaussians.num_gaussians:
            raise ValueError("cap_max must cover initialized Gaussians")
        finite = {
            "high_threshold": high_threshold,
            "low_threshold": low_threshold,
            "high_contribution": high_contribution,
            "medium_contribution": medium_contribution,
            "opacity_exponent": opacity_exponent,
            "scene_sensitivity": scene_sensitivity,
        }
        if any(not math.isfinite(float(value)) for value in finite.values()):
            raise ValueError("perceptual strategy values must be finite")
        if not 0.0 <= scene_sensitivity <= 1.0:
            raise ValueError("scene_sensitivity must be in [0, 1]")

        self.params: GaussianParameterMap = gaussians.parameter_map()
        self.optimizers = optimizers
        self.backend = _PerceptualBackend(
            cap_max=cap_max,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            high_interval=high_interval,
            medium_interval=medium_interval,
            high_contribution=high_contribution,
            medium_contribution=medium_contribution,
            opacity_exponent=opacity_exponent,
            split_only=scene_sensitivity > 0.85,
            prune_opa=prune_opa,
            grow_grad2d=grow_grad2d,
            grow_scale3d=grow_scale3d,
            prune_scale3d=math.inf,
            refine_scale2d_stop_iter=0,
            refine_start_iter=refine_start_step - 1,
            refine_stop_iter=refine_stop_step,
            refine_every=refine_every,
            reset_every=reset_every,
            absgrad=False,
        )
        self.backend.check_sanity(self.params, self.optimizers)

    def initialize_state(self, scene_scale: float = 1.0) -> dict[str, Any]:
        if not math.isfinite(scene_scale) or scene_scale <= 0.0:
            raise ValueError("scene_scale must be positive and finite")
        return self.backend.initialize_state(scene_scale)

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
            self.params,
            self.optimizers,
            state,
            step,
            info,
            packed=packed,
        )
