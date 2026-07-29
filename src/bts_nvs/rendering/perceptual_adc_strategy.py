from __future__ import annotations

import math
from typing import Any

import torch

from bts_nvs.models.gaussian_parameters import (
    GaussianParameterMap,
    GaussianParameters,
)
from bts_nvs.rendering.perceptual_events import is_perceptual_density_event

try:
    from gsplat import DefaultStrategy
    from gsplat.strategy.ops import duplicate, split
except ImportError:
    DefaultStrategy = None
    duplicate = split = None


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _budget_operations(
    clone_mask: torch.Tensor,
    split_mask: torch.Tensor,
    priority: torch.Tensor,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    selected_clone = torch.zeros_like(clone_mask)
    selected_split = torch.zeros_like(split_mask)
    clone_indices = torch.where(clone_mask)[0]
    split_indices = torch.where(split_mask)[0]
    operation_count = len(clone_indices) + len(split_indices)
    take = min(max(budget, 0), operation_count)
    if take == 0:
        return selected_clone, selected_split, 0

    operation_priority = torch.cat(
        (priority[clone_indices], priority[split_indices])
    )
    chosen = torch.topk(operation_priority, take).indices
    clone_choice = chosen[chosen < len(clone_indices)]
    split_choice = chosen[chosen >= len(clone_indices)] - len(clone_indices)
    selected_clone[clone_indices[clone_choice]] = True
    selected_split[split_indices[split_choice]] = True
    return selected_clone, selected_split, take


class _PerceptualADCBackend(
    DefaultStrategy if DefaultStrategy is not None else object
):
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
                "event_contribution": None,
                "adc_clone_count": 0,
                "adc_split_count": 0,
                "perceptual_clone_count": 0,
                "perceptual_split_count": 0,
                "perceptual_cap_hit_step": None,
            }
        )
        return state

    def _update_state(self, params, state, info, packed=False) -> None:
        super()._update_state(params, state, info, packed=packed)
        contribution = info.get("perceptual_contribution")
        if contribution is None:
            return
        if not isinstance(contribution, torch.Tensor):
            raise RuntimeError("perceptual contribution must be a tensor")
        if contribution.shape != (len(params["means"]),):
            raise RuntimeError("perceptual contribution shape does not match Gaussians")
        state["event_contribution"] = contribution

    @torch.no_grad()
    def _grow_gs(self, params, optimizers, state, step):
        gradients = state["grad2d"] / state["count"].clamp_min(1)
        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        adc_clone = (gradients > self.grow_grad2d) & is_small
        adc_split = (gradients > self.grow_grad2d) & ~is_small

        perceptual_clone = torch.zeros_like(adc_clone)
        perceptual_split = torch.zeros_like(adc_split)
        perceptual_priority = torch.zeros_like(gradients)
        event = is_perceptual_density_event(
            step,
            high_interval=self.high_interval,
            medium_interval=self.medium_interval,
        )
        if event:
            contribution = state["event_contribution"]
            if not isinstance(contribution, torch.Tensor):
                raise RuntimeError(
                    "perceptual ADC event requires full-view contribution"
                )
            sensitivity = torch.sigmoid(params["sensitivity_logits"])
            if step % self.high_interval == 0:
                high = (
                    (sensitivity >= self.high_threshold)
                    & (contribution > self.high_contribution)
                )
                if self.split_only:
                    perceptual_split |= high
                else:
                    perceptual_clone |= high & is_small
                    perceptual_split |= high & ~is_small
                perceptual_priority = torch.maximum(
                    perceptual_priority,
                    contribution * high,
                )
            if step % self.medium_interval == 0:
                medium = (
                    (sensitivity > self.low_threshold)
                    & (sensitivity < self.high_threshold)
                    & (contribution > self.medium_contribution)
                )
                perceptual_split |= medium
                perceptual_priority = torch.maximum(
                    perceptual_priority,
                    contribution * medium,
                )

        available = self.cap_max - len(params["means"])
        selected_adc_clone, selected_adc_split, adc_count = _budget_operations(
            adc_clone,
            adc_split,
            gradients,
            available,
        )
        remaining = max(available - adc_count, 0)
        extra_clone = perceptual_clone & ~selected_adc_clone
        extra_split = perceptual_split & ~selected_adc_split
        requested_count = (
            int(adc_clone.sum())
            + int(adc_split.sum())
            + int(extra_clone.sum())
            + int(extra_split.sum())
        )
        selected_perceptual_clone, selected_perceptual_split, perceptual_count = (
            _budget_operations(
                extra_clone,
                extra_split,
                perceptual_priority,
                remaining,
            )
        )
        is_clone = selected_adc_clone | selected_perceptual_clone
        is_split = selected_adc_split | selected_perceptual_split
        if available <= 0 or adc_count + perceptual_count < requested_count:
            state["perceptual_cap_hit_step"] = (
                state["perceptual_cap_hit_step"] or step
            )

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

        state["adc_clone_count"] += int(selected_adc_clone.sum())
        state["adc_split_count"] += int(selected_adc_split.sum())
        state["perceptual_clone_count"] += int(selected_perceptual_clone.sum())
        state["perceptual_split_count"] += int(selected_perceptual_split.sum())
        if event:
            state["event_contribution"] = None
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


class GsplatPerceptualADCStrategy:
    """Perceptual-GS adapter that preserves standard ADC at every refine event."""

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
            raise ImportError("gsplat==1.4.0 is required for perceptual ADC")
        if not hasattr(gaussians, "sensitivity_logits"):
            raise ValueError("perceptual ADC requires sensitivity logits")
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
            raise ValueError("perceptual ADC values must be finite")
        if not 0.0 <= scene_sensitivity <= 1.0:
            raise ValueError("scene_sensitivity must be in [0, 1]")

        self.params: GaussianParameterMap = gaussians.parameter_map()
        self.optimizers = optimizers
        self.backend = _PerceptualADCBackend(
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
