from __future__ import annotations

import math
from typing import Any

import torch

from bts_nvs.models.gaussian_parameters import (
    GaussianParameterMap,
    GaussianParameters,
)
from bts_nvs.rendering.spectral_math import (
    shape_aware_child_scales,
    spectral_entropy_and_condition,
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


def _take_budget(
    mask: torch.Tensor,
    priority: torch.Tensor,
    budget: int,
) -> torch.Tensor:
    selected = torch.zeros_like(mask)
    indices = torch.where(mask)[0]
    take = min(max(budget, 0), len(indices))
    if take:
        chosen = torch.topk(priority[indices], take).indices
        selected[indices[chosen]] = True
    return selected


def _take_adc_budget(
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
    if not take:
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


class _SpectralBackend(
    DefaultStrategy if DefaultStrategy is not None else object
):
    def __init__(
        self,
        *,
        cap_max: int,
        entropy_threshold: float,
        split_k: float,
        split_k0: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cap_max = cap_max
        self.entropy_threshold = entropy_threshold
        self.split_k = split_k
        self.split_k0 = split_k0

    def initialize_state(self, scene_scale: float = 1.0) -> dict[str, Any]:
        state = super().initialize_state(scene_scale)
        state.update(
            {
                "adc_clone_count": 0,
                "adc_split_count": 0,
                "spectral_split_count": 0,
                "spectral_cap_hit_step": None,
                "spectral_child_condition_violations": 0,
            }
        )
        return state

    @torch.no_grad()
    def _grow_gs(self, params, optimizers, state, step):
        gradients = state["grad2d"] / state["count"].clamp_min(1)
        scales = torch.exp(params["scales"]).float()
        is_small = (
            scales.max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        adc_clone = (gradients > self.grow_grad2d) & is_small
        adc_split = (gradients > self.grow_grad2d) & ~is_small

        available = max(self.cap_max - len(params["means"]), 0)
        selected_adc_clone, selected_adc_split, adc_count = _take_adc_budget(
            adc_clone,
            adc_split,
            gradients,
            available,
        )
        available -= adc_count

        entropy, _ = spectral_entropy_and_condition(scales)
        spectral_candidates = (
            (entropy < self.entropy_threshold)
            & ~selected_adc_clone
            & ~selected_adc_split
        )
        selected_spectral = _take_budget(
            spectral_candidates,
            self.entropy_threshold - entropy,
            available,
        )
        requested = (
            int(adc_clone.sum())
            + int(adc_split.sum())
            + int(spectral_candidates.sum())
        )
        selected_count = (
            int(selected_adc_clone.sum())
            + int(selected_adc_split.sum())
            + int(selected_spectral.sum())
        )
        if selected_count < requested:
            state["spectral_cap_hit_step"] = (
                state["spectral_cap_hit_step"] or step
            )

        clone_count = int(selected_adc_clone.sum())
        if clone_count:
            duplicate(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=selected_adc_clone,
            )
            extension = torch.zeros(
                clone_count,
                dtype=torch.bool,
                device=selected_adc_split.device,
            )
            selected_adc_split = torch.cat((selected_adc_split, extension))
            selected_spectral = torch.cat((selected_spectral, extension))

        split_mask = selected_adc_split | selected_spectral
        split_count = int(split_mask.sum())
        if split_count:
            self._split(
                params,
                optimizers,
                state,
                split_mask,
                selected_spectral,
            )

        state["adc_clone_count"] += clone_count
        state["adc_split_count"] += int(selected_adc_split.sum())
        state["spectral_split_count"] += int(selected_spectral.sum())
        return clone_count, split_count

    @torch.no_grad()
    def _split(
        self,
        params,
        optimizers,
        state,
        split_mask: torch.Tensor,
        spectral_mask: torch.Tensor,
    ) -> None:
        selected = torch.where(split_mask)[0]
        parent_scales = torch.exp(params["scales"][selected]).float()
        child_scales = parent_scales / 1.6
        spectral_selected = spectral_mask[selected]
        if bool(spectral_selected.any()):
            shaped = shape_aware_child_scales(
                parent_scales[spectral_selected],
                k=self.split_k,
                k0=self.split_k0,
            )
            _, parent_condition = spectral_entropy_and_condition(
                parent_scales[spectral_selected]
            )
            _, child_condition = spectral_entropy_and_condition(shaped)
            violations = child_condition > parent_condition * (1.0 + 1e-5)
            state["spectral_child_condition_violations"] += int(
                violations.sum()
            )
            if bool(violations.any()):
                raise RuntimeError(
                    "shape-aware split increased Gaussian condition number"
                )
            child_scales[spectral_selected] = shaped

        split(
            params=params,
            optimizers=optimizers,
            state=state,
            mask=split_mask,
            revised_opacity=False,
        )
        child_count = 2 * len(selected)
        params["scales"].data[-child_count:] = torch.log(
            child_scales.to(params["scales"].dtype)
        ).repeat(2, 1)


class GsplatSpectralStrategy:
    """Default ADC with gradient-independent Spectral-GS 3D splitting."""

    def __init__(
        self,
        gaussians: GaussianParameters,
        optimizers: dict[str, torch.optim.Optimizer],
        *,
        cap_max: int,
        entropy_threshold: float,
        split_k: float,
        split_k0: float,
        prune_opa: float = 0.005,
        grow_grad2d: float = 0.0002,
        grow_scale3d: float = 0.01,
        refine_start_step: int = 500,
        refine_stop_step: int = 15_000,
        refine_every: int = 100,
        reset_every: int = 3_000,
    ) -> None:
        if DefaultStrategy is None or duplicate is None or split is None:
            raise ImportError("gsplat==1.4.0 is required for spectral splitting")
        for name, value in {
            "cap_max": cap_max,
            "refine_start_step": refine_start_step,
            "refine_stop_step": refine_stop_step,
            "refine_every": refine_every,
            "reset_every": reset_every,
        }.items():
            _positive_integer(value, name)
        if cap_max < gaussians.num_gaussians:
            raise ValueError("cap_max must cover initialized Gaussians")
        for name, value in {
            "entropy_threshold": entropy_threshold,
            "split_k": split_k,
            "split_k0": split_k0,
        }.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if split_k0 < 1.0:
            raise ValueError("split_k0 must be at least one")

        self.params: GaussianParameterMap = gaussians.parameter_map()
        self.optimizers = optimizers
        self.backend = _SpectralBackend(
            cap_max=cap_max,
            entropy_threshold=entropy_threshold,
            split_k=split_k,
            split_k0=split_k0,
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
