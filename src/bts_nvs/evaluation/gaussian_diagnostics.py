from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bts_nvs.rendering.gsplat_renderer import render_gaussians
from bts_nvs.training.trainer import _normalize_world_to_camera


def _quantiles(values: torch.Tensor, prefix: str) -> dict[str, float]:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0 or not bool(torch.isfinite(flat).all()):
        raise ValueError(f"{prefix} values must be non-empty and finite")
    quantiles = torch.quantile(
        flat,
        torch.tensor((0.5, 0.95, 0.99), device=flat.device),
    ).cpu()
    return {
        f"{prefix}_p50": float(quantiles[0]),
        f"{prefix}_p95": float(quantiles[1]),
        f"{prefix}_p99": float(quantiles[2]),
        f"{prefix}_max": float(flat.max().cpu()),
    }


def summarize_gaussians(
    gaussians,
    *,
    scale_threshold: float,
) -> dict[str, float | int]:
    if not math.isfinite(scale_threshold) or scale_threshold <= 0.0:
        raise ValueError("scale_threshold must be positive and finite")
    scales = gaussians.get_scales().detach().amax(dim=1)
    opacities = gaussians.get_opacities().detach()
    extreme = scales > scale_threshold
    opacity_total = opacities.sum().clamp_min(1e-12)
    return {
        **_quantiles(scales, "scale3d"),
        **_quantiles(opacities, "opacity"),
        "scale_threshold": float(scale_threshold),
        "scale_above_threshold_count": int(extreme.sum().cpu()),
        "scale_above_threshold_opacity_mass": float(opacities[extreme].sum().cpu()),
        "scale_above_threshold_opacity_fraction": float(
            (opacities[extreme].sum() / opacity_total).cpu()
        ),
    }


def _visible_radii(
    info: dict,
    gaussian_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    radii = info.get("radii")
    if not isinstance(radii, torch.Tensor):
        raise RuntimeError("renderer diagnostics require radii")
    radii = radii.reshape(-1)
    ids = info.get("gaussian_ids")
    if ids is None:
        if radii.numel() != gaussian_count:
            raise RuntimeError("packed radii require gaussian_ids")
        ids = torch.arange(gaussian_count, device=radii.device)
    elif not isinstance(ids, torch.Tensor) or ids.numel() != radii.numel():
        raise RuntimeError("renderer gaussian_ids do not match radii")
    ids = ids.reshape(-1).long()
    visible = radii > 0
    return ids[visible], radii[visible].float()


@torch.no_grad()
def build_gaussian_diagnostics(
    trainer,
    dataset,
    output_dir: Path,
    *,
    scale_threshold: float,
    radius_threshold_pixels: float,
    density_summary: dict[str, int],
) -> dict[str, object]:
    if (
        not math.isfinite(radius_threshold_pixels)
        or radius_threshold_pixels <= 0.0
    ):
        raise ValueError("radius_threshold_pixels must be positive and finite")
    if len(dataset) <= 0:
        raise ValueError("diagnostic dataset is empty")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    count = trainer.gaussians.num_gaussians
    projected_extreme = torch.zeros(
        count, dtype=torch.bool, device=trainer.gaussians.means.device
    )
    view_reports: dict[str, dict[str, float | int]] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        normalized_w2c = _normalize_world_to_camera(
            sample.world_to_camera,
            dataset.manifest.normalization_transform,
        )
        viewmat = torch.from_numpy(normalized_w2c).to(trainer.device)
        result = render_gaussians(
            gaussians=trainer.gaussians,
            viewmat=viewmat,
            intrinsics=sample.intrinsics,
            active_sh_degree=trainer.active_sh_degree,
            render_mode="RGB",
        )
        ids, radii = _visible_radii(result.info, count)
        if radii.numel() == 0:
            raise RuntimeError("diagnostic view contains no visible Gaussians")
        extreme_ids = ids[radii > radius_threshold_pixels]
        projected_extreme[extreme_ids] = True
        view_reports[sample.image_name] = {
            **_quantiles(radii, "projected_radius"),
            "visible_gaussians": int(torch.unique(ids).numel()),
            "radius_above_threshold_count": int(torch.unique(extreme_ids).numel()),
        }

        keep = torch.ones(count, dtype=torch.bool, device=trainer.device)
        keep[extreme_ids] = False
        if bool(keep.any()):
            filtered = render_gaussians(
                gaussians=trainer.gaussians,
                viewmat=viewmat,
                intrinsics=sample.intrinsics,
                active_sh_degree=trainer.active_sh_degree,
                render_mode="RGB",
                gaussian_mask=keep,
            )
            image = (
                filtered.rgb.float().clamp(0.0, 1.0).mul(255.0).round().byte()
            )
            Image.fromarray(image.cpu().numpy()).save(
                root / Path(sample.image_name).with_suffix(".png").name
            )
        else:
            Image.fromarray(
                np.zeros(
                    (sample.intrinsics.height, sample.intrinsics.width, 3),
                    dtype=np.uint8,
                )
            ).save(root / Path(sample.image_name).with_suffix(".png").name)

    opacities = trainer.gaussians.get_opacities().detach()
    opacity_total = opacities.sum().clamp_min(1e-12)
    radius_opacity = opacities[projected_extreme].sum()
    projected = {
        "quantile_aggregation": "max_of_per_view_quantile_v1",
        "radius_threshold_pixels": float(radius_threshold_pixels),
        "radius_above_threshold_count": int(projected_extreme.sum().cpu()),
        "radius_above_threshold_opacity_mass": float(radius_opacity.cpu()),
        "radius_above_threshold_opacity_fraction": float(
            (radius_opacity / opacity_total).cpu()
        ),
        "projected_radius_p95": max(
            float(report["projected_radius_p95"])
            for report in view_reports.values()
        ),
        "projected_radius_p99": max(
            float(report["projected_radius_p99"])
            for report in view_reports.values()
        ),
        "projected_radius_max": max(
            float(report["projected_radius_max"])
            for report in view_reports.values()
        ),
    }
    return {
        "schema_version": 1,
        "scene_id": dataset.manifest.scene_id,
        "image_count": len(view_reports),
        "scale_opacity": summarize_gaussians(
            trainer.gaussians,
            scale_threshold=scale_threshold,
        ),
        "projected_radius": projected,
        "density_control": density_summary,
        "images": view_reports,
        "diagnostic_only": True,
    }
