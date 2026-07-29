from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from bts_nvs.models.gaussian_parameters import GaussianParameters


def _weight(config: Mapping[str, Any], name: str) -> float:
    value = config.get(name, 0.0)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


class DensityRegularizer:
    """Compute method-specific Gaussian regularization without image-loss coupling."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        mode = config.get("density_strategy", "default")
        if mode not in {
            "default",
            "mcmc",
            "perceptual",
            "perceptual-adc",
            "spectral",
        }:
            raise ValueError(f"unsupported density strategy: {mode}")
        self.mode = mode
        self.opacity_weight = _weight(config, "mcmc_opacity_reg")
        self.scale_weight = _weight(config, "mcmc_scale_reg")
        if mode != "mcmc" and (
            self.opacity_weight != 0.0 or self.scale_weight != 0.0
        ):
            raise ValueError("MCMC regularization requires density_strategy=mcmc")

    def __call__(self, gaussians: GaussianParameters) -> torch.Tensor:
        if self.mode != "mcmc":
            return gaussians.means.new_zeros(())
        opacity = torch.sigmoid(gaussians.opacities).mean()
        scale = torch.exp(gaussians.scales).mean()
        return self.opacity_weight * opacity + self.scale_weight * scale
