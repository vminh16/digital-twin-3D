from __future__ import annotations

from collections.abc import Iterable

import torch

from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.models.gaussian_parameters import GaussianParameters
from bts_nvs.rendering.gsplat_renderer import render_gaussians
from bts_nvs.rendering.render_result import RenderResult


def render_sensitivity(
    gaussians: GaussianParameters,
    viewmat: torch.Tensor,
    intrinsics: CameraIntrinsics,
    *,
    rasterize_mode: str = "classic",
    compute_contribution: bool = True,
    retain_graph: bool = True,
) -> tuple[RenderResult, torch.Tensor | None]:
    sensitivity = gaussians.get_sensitivities()
    colors = sensitivity[:, None].expand(-1, 3)
    result = render_gaussians(
        gaussians,
        viewmat,
        intrinsics,
        active_sh_degree=0,
        rasterize_mode=rasterize_mode,
        override_colors=colors,
    )
    if not compute_contribution:
        return result, None
    contribution = torch.autograd.grad(
        result.rgb[..., 0].sum(),
        sensitivity,
        retain_graph=retain_graph,
        create_graph=False,
    )[0].detach()
    if contribution.shape != (gaussians.num_gaussians,):
        raise RuntimeError("sensitivity contribution has an invalid shape")
    if not torch.isfinite(contribution).all() or torch.any(contribution < 0.0):
        raise FloatingPointError("sensitivity contribution must be finite and nonnegative")
    return result, contribution


def max_perceptual_contribution(
    gaussians: GaussianParameters,
    views: Iterable[tuple[torch.Tensor, CameraIntrinsics]],
    *,
    rasterize_mode: str = "classic",
) -> torch.Tensor:
    maximum = torch.zeros(
        gaussians.num_gaussians,
        dtype=gaussians.means.dtype,
        device=gaussians.means.device,
    )
    view_count = 0
    for viewmat, intrinsics in views:
        _, contribution = render_sensitivity(
            gaussians,
            viewmat,
            intrinsics,
            rasterize_mode=rasterize_mode,
            retain_graph=False,
        )
        assert contribution is not None
        maximum = torch.maximum(maximum, contribution)
        view_count += 1
    if view_count == 0:
        raise ValueError("perceptual contribution sweep requires at least one view")
    return maximum
