from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from bts_nvs.models.gaussian_parameters import GaussianParameters
from bts_nvs.rendering.density_strategy import GsplatStrategy
from bts_nvs.rendering.mcmc_density_strategy import GsplatMCMCStrategy


def build_density_strategy(
    gaussians: GaussianParameters,
    optimizers: dict[str, torch.optim.Optimizer],
    config: Mapping[str, Any],
) -> GsplatStrategy | GsplatMCMCStrategy:
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
    raise ValueError(f"unsupported density strategy: {mode}")
