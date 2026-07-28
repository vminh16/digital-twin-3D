from __future__ import annotations

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
) -> tuple[RenderResult, torch.Tensor]:
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
    contribution = torch.autograd.grad(
        result.rgb[..., 0].sum(),
        sensitivity,
        retain_graph=True,
        create_graph=False,
    )[0].detach()
    if contribution.shape != (gaussians.num_gaussians,):
        raise RuntimeError("sensitivity contribution has an invalid shape")
    if not torch.isfinite(contribution).all() or torch.any(contribution < 0.0):
        raise FloatingPointError("sensitivity contribution must be finite and nonnegative")
    return result, contribution
