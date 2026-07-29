from __future__ import annotations

import torch

from bts_nvs.models.gaussian_parameters import GaussianParameters
from bts_nvs.models.optimizer import setup_optimizers
from bts_nvs.rendering.spectral_density_strategy import GsplatSpectralStrategy


def run_spectral_density_preflight(optimizer_backend: str) -> None:
    """Exercise one real CUDA shape-aware split before scene loading."""
    if not torch.cuda.is_available():
        raise RuntimeError("spectral density preflight requires CUDA")
    device = torch.device("cuda")
    scales = torch.tensor(
        [[0.20, 0.01, 0.01], [0.02, 0.02, 0.02]],
        device=device,
    )
    count = len(scales)
    gaussians = GaussianParameters(
        means=torch.zeros((count, 3), device=device),
        scales=scales.log(),
        quats=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * count,
            device=device,
        ),
        opacities=torch.zeros(count, device=device),
        sh0=torch.zeros((count, 1, 3), device=device),
        shN=torch.zeros((count, 15, 3), device=device),
    )
    optimizers = setup_optimizers(gaussians, backend=optimizer_backend)
    for optimizer in optimizers.values():
        parameter = optimizer.param_groups[0]["params"][0]
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    strategy = GsplatSpectralStrategy(
        gaussians,
        optimizers,
        cap_max=count + 1,
        entropy_threshold=0.5,
        split_k=0.6,
        split_k0=1.0,
    )
    state = strategy.initialize_state()
    state["grad2d"] = torch.zeros(count, device=device)
    state["count"] = torch.ones(count, device=device)
    strategy.backend._grow_gs(
        strategy.params,
        optimizers,
        state,
        step=500,
    )
    torch.cuda.synchronize(device)

    if gaussians.num_gaussians != count + 1:
        raise RuntimeError("spectral density preflight did not split one Gaussian")
    if state["spectral_split_count"] != 1:
        raise RuntimeError("spectral density preflight counter is invalid")
    for name, optimizer in optimizers.items():
        parameter = optimizer.param_groups[0]["params"][0]
        if parameter.shape[0] != count + 1 or not torch.isfinite(parameter).all():
            raise RuntimeError(
                f"spectral preflight left invalid optimizer parameter: {name}"
            )
