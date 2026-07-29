from __future__ import annotations

import math

import torch


_EPSILON = 1e-12


def spectral_entropy_and_condition(
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw covariance spectral entropy and condition number."""
    physical = _physical_scales(scales)
    variances = physical.square().clamp_min(_EPSILON)
    probabilities = variances / variances.sum(dim=-1, keepdim=True)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    condition = variances.max(dim=-1).values / variances.min(dim=-1).values
    return entropy, condition


def shape_aware_child_scales(
    scales: torch.Tensor,
    *,
    k: float,
    k0: float,
) -> torch.Tensor:
    """Apply Spectral-GS equations 10–11 to physical parent scales."""
    physical = _physical_scales(scales)
    _positive_finite(k, "k")
    _positive_finite(k0, "k0")
    if k0 < 1.0:
        raise ValueError("k0 must be at least one")
    maximum = physical.square().max(dim=-1, keepdim=True).values
    principal = physical.square() == maximum
    divisor = torch.full_like(physical, float(k0))
    divisor = divisor + principal.to(physical.dtype) * float(k)
    return physical / divisor


def _physical_scales(scales: torch.Tensor) -> torch.Tensor:
    if not isinstance(scales, torch.Tensor):
        raise TypeError("scales must be a tensor")
    if scales.ndim != 2 or scales.shape[-1] != 3:
        raise ValueError("scales must have shape (N, 3)")
    physical = scales.float()
    if not torch.isfinite(physical).all():
        raise ValueError("scales must be finite")
    if bool((physical < 0.0).any()):
        raise ValueError("scales must be nonnegative")
    return physical.clamp_min(_EPSILON)


def _positive_finite(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be positive and finite")
