from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from bts_nvs.models.gaussian_parameters import GaussianParameters
from bts_nvs.rendering.density_strategy import GsplatStrategy
from bts_nvs.rendering.mcmc_density_strategy import GsplatMCMCStrategy
from bts_nvs.rendering.perceptual_density_strategy import (
    GsplatPerceptualStrategy,
)


def build_density_strategy(
    gaussians: GaussianParameters,
    optimizers: dict[str, torch.optim.Optimizer],
    config: Mapping[str, Any],
) -> GsplatStrategy | GsplatMCMCStrategy | GsplatPerceptualStrategy:
    mode = config.get("density_strategy", "default")
    common = {
        "refine_start_step": config.get("refine_start_step", 500),
        "refine_stop_step": config.get("refine_stop_step", 15_000),
        "refine_every": config.get("refine_every", 100),
    }
    if mode == "default":
        return GsplatStrategy(
            gaussians,
            optimizers,
            prune_opa=config.get("prune_opa", 0.005),
            grow_grad2d=config.get("grow_grad2d", 0.0002),
            grow_scale3d=config.get("grow_scale3d", 0.01),
            reset_every=config.get("reset_every", 3_000),
            absgrad=config.get("absgrad", False),
            **common,
        )
    if mode == "mcmc":
        return GsplatMCMCStrategy(
            gaussians,
            optimizers,
            cap_max=config["mcmc_cap_max"],
            noise_lr=config["mcmc_noise_lr"],
            min_opacity=config.get("prune_opa", 0.005),
            **common,
        )
    if mode == "perceptual":
        return GsplatPerceptualStrategy(
            gaussians,
            optimizers,
            cap_max=config["perceptual_cap_max"],
            high_threshold=config["perceptual_high_threshold"],
            low_threshold=config["perceptual_low_threshold"],
            high_interval=config["perceptual_high_interval"],
            medium_interval=config["perceptual_medium_interval"],
            high_contribution=config["perceptual_high_contribution"],
            medium_contribution=config["perceptual_medium_contribution"],
            opacity_exponent=config["perceptual_opacity_exponent"],
            scene_sensitivity=config["perceptual_scene_sensitivity"],
            prune_opa=config.get("prune_opa", 0.005),
            grow_grad2d=config.get("grow_grad2d", 0.0002),
            grow_scale3d=config.get("grow_scale3d", 0.01),
            reset_every=config.get("reset_every", 3_000),
            **common,
        )
    raise ValueError(f"unsupported density strategy: {mode}")
