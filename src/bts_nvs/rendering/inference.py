from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

import cv2
import numpy as np
import torch

from bts_nvs.cameras.distortion import (
    CameraDistortion,
    undistort_normalized_points,
)
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.cameras.poses import camera_center_from_world_to_camera
from bts_nvs.models.gaussian_parameters import GaussianParameters
from bts_nvs.rendering.gsplat_renderer import render_gaussians


_GAUSSIAN_KEYS = ("means", "scales", "quats", "opacities", "sh0", "shN")


def gaussians_from_checkpoint(
    checkpoint: Mapping[str, object],
    device: torch.device,
) -> GaussianParameters:
    try:
        state = checkpoint["gaussians"]
        if not isinstance(state, Mapping):
            raise TypeError
        tensors = {
            name: state[name].detach().to(device=device)
            for name in _GAUSSIAN_KEYS
            if isinstance(state[name], torch.Tensor)
        }
        if tuple(tensors) != _GAUSSIAN_KEYS:
            raise TypeError
        gaussians = GaussianParameters(**tensors)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint Gaussian state is invalid") from error
    gaussians.eval()
    gaussians.requires_grad_(False)
    return gaussians


def normalized_test_world_to_camera(
    raw_world_to_camera: np.ndarray,
    normalization_transform: np.ndarray,
) -> np.ndarray:
    pose = np.asarray(raw_world_to_camera, dtype=np.float64)
    transform = np.asarray(normalization_transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("normalization transform must be finite with shape (4, 4)")
    center = camera_center_from_world_to_camera(pose)
    normalized_center = transform[:3, :3] @ center + transform[:3, 3]
    normalized = pose.copy()
    normalized[:3, 3] = -pose[:3, :3] @ normalized_center
    return normalized


def redistort_render(
    image: np.ndarray,
    intrinsics: CameraIntrinsics,
    distortion: CameraDistortion,
) -> np.ndarray:
    source = np.asarray(image)
    expected_shape = (intrinsics.height, intrinsics.width, 3)
    if source.shape != expected_shape or not np.all(np.isfinite(source)):
        raise ValueError(f"render must be finite with shape {expected_shape}")
    if distortion.model == "PINHOLE":
        return source.copy()

    map_x, map_y = _redistortion_maps(intrinsics, distortion)
    return cv2.remap(
        source,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


@lru_cache(maxsize=32)
def _redistortion_maps(
    intrinsics: CameraIntrinsics,
    distortion: CameraDistortion,
) -> tuple[np.ndarray, np.ndarray]:
    pixel_x, pixel_y = np.meshgrid(
        np.arange(intrinsics.width, dtype=np.float64),
        np.arange(intrinsics.height, dtype=np.float64),
    )
    distorted = np.stack(
        (
            (pixel_x - intrinsics.cx) / intrinsics.fx,
            (pixel_y - intrinsics.cy) / intrinsics.fy,
        ),
        axis=-1,
    )
    undistorted = undistort_normalized_points(distorted, distortion)
    map_x = (intrinsics.fx * undistorted[..., 0] + intrinsics.cx).astype(
        np.float32
    )
    map_y = (intrinsics.fy * undistorted[..., 1] + intrinsics.cy).astype(
        np.float32
    )
    return map_x, map_y


def render_test_camera(
    gaussians: GaussianParameters,
    raw_world_to_camera: np.ndarray,
    intrinsics: CameraIntrinsics,
    distortion: CameraDistortion,
    normalization_transform: np.ndarray,
    active_sh_degree: int,
) -> np.ndarray:
    normalized = normalized_test_world_to_camera(
        raw_world_to_camera, normalization_transform
    )
    with torch.inference_mode():
        rendered = render_gaussians(
            gaussians,
            normalized,
            intrinsics,
            active_sh_degree,
        ).rgb
    image = rendered.detach().to(device="cpu", dtype=torch.float32).numpy()
    if not np.all(np.isfinite(image)):
        raise ValueError("render contains non-finite RGB values")
    return redistort_render(image, intrinsics, distortion)
