from __future__ import annotations

import torch

from bts_nvs.models.gaussian_parameters import GaussianParameters


_LEARNING_RATES = {
    "means": 1.6e-4,
    "scales": 5.0e-3,
    "quats": 1.0e-3,
    "opacities": 5.0e-2,
    "sh0": 2.5e-3,
    "shN": 1.25e-4,
}


def setup_optimizers(
    gaussians: GaussianParameters,
) -> dict[str, torch.optim.Adam]:
    """Create one Adam optimizer per Gaussian parameter for gsplat strategy updates."""
    parameters = dict(gaussians.named_parameters())
    return {
        name: torch.optim.Adam(
            [{"params": [parameters[name]], "lr": lr, "name": name}],
            eps=1e-15,
        )
        for name, lr in _LEARNING_RATES.items()
    }


def setup_mean_scheduler(
    optimizers: dict[str, torch.optim.Optimizer],
    *,
    max_steps: int = 30_000,
) -> torch.optim.lr_scheduler.ExponentialLR:
    """Decay the means learning rate to one percent over ``max_steps``."""
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    if "means" not in optimizers:
        raise ValueError("optimizers must contain a 'means' optimizer")
    return torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"],
        gamma=0.01 ** (1.0 / max_steps),
    )
