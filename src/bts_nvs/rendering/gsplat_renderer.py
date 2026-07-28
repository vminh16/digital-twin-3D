from __future__ import annotations

import torch

from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.models.gaussian_parameters import GaussianParameters
from bts_nvs.rendering.render_result import RenderResult

try:
    from gsplat import rasterization
except ImportError:
    rasterization = None


def render_gaussians(
    gaussians: GaussianParameters,
    viewmat: torch.Tensor,
    intrinsics: CameraIntrinsics,
    active_sh_degree: int,
    backgrounds: torch.Tensor | None = None,
    render_mode: str = "RGB",
    absgrad: bool = False,
    rasterize_mode: str = "classic",
    gaussian_mask: torch.Tensor | None = None,
    override_colors: torch.Tensor | None = None,
) -> RenderResult:
    """Render one pinhole camera in normalized world coordinates.

    ``viewmat`` must be the normalized world-to-camera pose matching the
    normalized Gaussian means. Phase 3 uses one camera per optimization step.
    """
    if isinstance(active_sh_degree, bool) or not isinstance(active_sh_degree, int):
        raise ValueError("active_sh_degree must be an integer")
    if not 0 <= active_sh_degree <= gaussians.max_sh_degree:
        raise ValueError(
            "active_sh_degree exceeds the available SH coefficients"
        )
    if render_mode not in {"RGB", "RGB+D"}:
        raise ValueError("render_mode must be 'RGB' or 'RGB+D'")
    if not isinstance(absgrad, bool):
        raise ValueError("absgrad must be boolean")
    if rasterize_mode not in {"classic", "antialiased"}:
        raise ValueError("rasterize_mode must be 'classic' or 'antialiased'")
    if gaussian_mask is None:
        means = gaussians.get_means()
        quats = gaussians.quats
        scales = gaussians.get_scales()
        opacities = gaussians.get_opacities()
        colors = gaussians.get_shs()
    else:
        mask = torch.as_tensor(gaussian_mask, device=gaussians.means.device)
        if mask.dtype != torch.bool or mask.shape != (gaussians.num_gaussians,):
            raise ValueError("gaussian_mask must be boolean with shape (N,)")
        if not bool(mask.any()):
            raise ValueError("gaussian_mask must retain at least one Gaussian")
        means = gaussians.get_means()[mask]
        quats = gaussians.quats[mask]
        scales = gaussians.get_scales()[mask]
        opacities = gaussians.get_opacities()[mask]
        colors = gaussians.get_shs()[mask]
    if override_colors is not None:
        if gaussian_mask is not None:
            raise ValueError("override_colors cannot be combined with gaussian_mask")
        colors = torch.as_tensor(
            override_colors,
            dtype=gaussians.means.dtype,
            device=gaussians.means.device,
        )
        if colors.shape != (gaussians.num_gaussians, 3):
            raise ValueError("override_colors must have shape (N, 3)")
        if not torch.isfinite(colors).all():
            raise ValueError("override_colors must be finite")

    device = gaussians.means.device
    dtype = gaussians.means.dtype
    viewmats = torch.as_tensor(viewmat)
    if viewmats.shape == (4, 4):
        viewmats = viewmats.unsqueeze(0)
    elif viewmats.shape != (1, 4, 4):
        raise ValueError(
            f"viewmat must have shape (4, 4) or (1, 4, 4), got {viewmats.shape}"
        )
    if viewmats.device.type == "cpu" and not torch.isfinite(viewmats).all():
        raise ValueError("viewmat must be finite")
    viewmats = viewmats.to(dtype=dtype, device=device)

    K = torch.as_tensor(intrinsics.matrix, dtype=dtype, device=device).unsqueeze(0)
    if backgrounds is None:
        background = torch.zeros((1, 3), dtype=dtype, device=device)
    else:
        background = torch.as_tensor(backgrounds)
        if background.shape != (3,):
            raise ValueError(f"backgrounds must have shape (3,), got {background.shape}")
        if background.device.type == "cpu" and not torch.isfinite(background).all():
            raise ValueError("backgrounds must be finite")
        background = background.to(dtype=dtype, device=device).unsqueeze(0)

    if rasterization is None:
        raise ImportError("gsplat==1.4.0 is required to render Gaussians")

    rendered, alpha, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=K,
        width=intrinsics.width,
        height=intrinsics.height,
        near_plane=0.01,
        far_plane=1e10,
        radius_clip=0.0,
        eps2d=0.3,
        sh_degree=None if override_colors is not None else active_sh_degree,
        packed=True,
        tile_size=16,
        backgrounds=background,
        render_mode=render_mode,
        sparse_grad=False,
        absgrad=absgrad,
        rasterize_mode=rasterize_mode,
    )

    rendered = rendered[0]
    depth = rendered[..., 3:4] if render_mode == "RGB+D" else None
    return RenderResult(
        rgb=rendered[..., :3],
        alpha=alpha[0],
        depth=depth,
        info=info,
    )
