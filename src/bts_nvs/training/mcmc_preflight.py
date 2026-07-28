from __future__ import annotations

import math

import torch

from bts_nvs.models.gaussian_parameters import GaussianParameters
from bts_nvs.models.optimizer import setup_optimizers
from bts_nvs.rendering.mcmc_density_strategy import GsplatMCMCStrategy


def run_mcmc_density_preflight(optimizer_backend: str) -> None:
    """Exercise one real CUDA relocation/growth event before scene loading."""
    if not torch.cuda.is_available():
        raise RuntimeError("MCMC density preflight requires CUDA")
    device = torch.device("cuda")
    count = 20
    gaussians = GaussianParameters(
        means=torch.zeros((count, 3), device=device),
        scales=torch.full((count, 3), math.log(0.1), device=device),
        quats=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * count,
            device=device,
        ),
        opacities=torch.full(
            (count,),
            torch.logit(torch.tensor(0.1)).item(),
            device=device,
        ),
        sh0=torch.zeros((count, 1, 3), device=device),
        shN=torch.zeros((count, 15, 3), device=device),
    )
    optimizers = setup_optimizers(gaussians, backend=optimizer_backend)
    for optimizer in optimizers.values():
        parameter = optimizer.param_groups[0]["params"][0]
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    strategy = GsplatMCMCStrategy(
        gaussians,
        optimizers,
        cap_max=count + 1,
    )
    state = strategy.initialize_state()
    means2d = torch.zeros((count, 2), device=device, requires_grad=True)
    strategy.step_pre_backward(state, 600, {"means2d": means2d})
    strategy.step_post_backward(state, 600, {"means2d": means2d})
    torch.cuda.synchronize(device)

    if gaussians.num_gaussians != count + 1:
        raise RuntimeError("MCMC density preflight did not add one Gaussian")
    for name, optimizer in optimizers.items():
        parameter = optimizer.param_groups[0]["params"][0]
        if parameter.shape[0] != count + 1 or not torch.isfinite(parameter).all():
            raise RuntimeError(
                f"MCMC density preflight left invalid optimizer parameter: {name}"
            )
