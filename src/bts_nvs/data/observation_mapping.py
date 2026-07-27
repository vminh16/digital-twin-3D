from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from bts_nvs.cameras.distortion import distort_normalized_points

from .manifest import SceneManifest
from .validation import DataContractError


@dataclass(frozen=True)
class ObservationMappingFit:
    scale: tuple[float, float]
    legacy_scale: tuple[float, float]
    observation_count: int
    mapped_reprojection_p50_px: float
    mapped_reprojection_p95_px: float
    legacy_reprojection_p50_px: float
    legacy_reprojection_p95_px: float


@dataclass(frozen=True)
class ObservationMappingDiagnostics:
    mode: str
    scale: tuple[float, float]
    legacy_scale: tuple[float, float]
    fit_observation_count: int
    mapped_reprojection_p50_px: float
    mapped_reprojection_p95_px: float
    legacy_reprojection_p50_px: float
    legacy_reprojection_p95_px: float
    color_comparison_point_count: int
    selected_color_mae_mean: float
    selected_color_mae_median: float
    selected_color_mae_p90: float
    legacy_color_mae_mean: float
    legacy_color_mae_median: float
    legacy_color_mae_p90: float


def fit_continuous_observation_mapping(
    manifest: SceneManifest,
    images_by_name: Mapping[str, object],
    valid_points: Mapping[int, object],
    legacy_scale: np.ndarray,
    image_names: tuple[str, ...],
) -> ObservationMappingFit:
    stored, projected = _project_observed_points(
        manifest,
        images_by_name,
        valid_points,
        image_names,
    )
    scale = np.asarray(
        [
            _fit_axis_scale(stored, projected, manifest, axis)
            for axis in range(2)
        ],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(scale))
        or np.any(scale < 1.0 / 64.0)
        or np.any(scale > 64.0)
    ):
        raise DataContractError(
            f"implausible continuous COLMAP observation scale: {scale.tolist()}"
        )

    mapped_quantiles = _residual_quantiles(stored / scale, projected)
    legacy_quantiles = _residual_quantiles(stored / legacy_scale, projected)
    max_dimension = max(
        max(intrinsics.width, intrinsics.height)
        for intrinsics in manifest.train_intrinsics
    )
    if mapped_quantiles[1] > max(2.0, 0.01 * max_dimension):
        raise DataContractError(
            "continuous observation mapping does not agree with current reprojections"
        )
    if (
        not np.allclose(scale, legacy_scale, atol=1e-8, rtol=0.0)
        and mapped_quantiles[1] >= legacy_quantiles[1]
    ):
        raise DataContractError(
            "continuous observation mapping does not improve reprojection residual"
        )
    return ObservationMappingFit(
        scale=tuple(float(value) for value in scale),
        legacy_scale=tuple(float(value) for value in legacy_scale),
        observation_count=len(stored),
        mapped_reprojection_p50_px=mapped_quantiles[0],
        mapped_reprojection_p95_px=mapped_quantiles[1],
        legacy_reprojection_p50_px=legacy_quantiles[0],
        legacy_reprojection_p95_px=legacy_quantiles[1],
    )


def build_observation_mapping_diagnostics(
    fit: ObservationMappingFit,
    point_ids: np.ndarray,
    selected_colors: np.ndarray,
    legacy_colors: np.ndarray,
    valid_points: Mapping[int, object],
) -> ObservationMappingDiagnostics:
    selected_error = _color_error_quantiles(
        point_ids,
        selected_colors,
        valid_points,
    )
    legacy_error = _color_error_quantiles(
        point_ids,
        legacy_colors,
        valid_points,
    )
    return ObservationMappingDiagnostics(
        mode="continuous-reprojection",
        scale=fit.scale,
        legacy_scale=fit.legacy_scale,
        fit_observation_count=fit.observation_count,
        mapped_reprojection_p50_px=fit.mapped_reprojection_p50_px,
        mapped_reprojection_p95_px=fit.mapped_reprojection_p95_px,
        legacy_reprojection_p50_px=fit.legacy_reprojection_p50_px,
        legacy_reprojection_p95_px=fit.legacy_reprojection_p95_px,
        color_comparison_point_count=len(point_ids),
        selected_color_mae_mean=selected_error[0],
        selected_color_mae_median=selected_error[1],
        selected_color_mae_p90=selected_error[2],
        legacy_color_mae_mean=legacy_error[0],
        legacy_color_mae_median=legacy_error[1],
        legacy_color_mae_p90=legacy_error[2],
    )


def _project_observed_points(
    manifest: SceneManifest,
    images_by_name: Mapping[str, object],
    valid_points: Mapping[int, object],
    image_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    stored_chunks: list[np.ndarray] = []
    projected_chunks: list[np.ndarray] = []
    index_by_name = {
        name: index for index, name in enumerate(manifest.train_image_names)
    }
    for name in image_names:
        index = index_by_name[name]
        registration = images_by_name.get(name)
        if registration is None:
            raise DataContractError(f"train image lacks COLMAP registration: {name}")
        supported = np.fromiter(
            (
                int(point_id) in valid_points
                for point_id in registration.point3d_ids
            ),
            dtype=bool,
            count=len(registration.point3d_ids),
        )
        if not np.any(supported):
            continue

        stored = registration.points2d_xy[supported]
        point_ids = registration.point3d_ids[supported]
        points = np.stack(
            [valid_points[int(point_id)].xyz for point_id in point_ids]
        )
        world_to_camera = manifest.train_world_to_camera[index]
        camera_points = (
            points @ world_to_camera[:3, :3].T
            + world_to_camera[:3, 3]
        )
        in_front = camera_points[:, 2] > 1e-8
        if not np.any(in_front):
            continue

        camera_points = camera_points[in_front]
        stored = stored[in_front]
        normalized = camera_points[:, :2] / camera_points[:, 2, None]
        distorted = distort_normalized_points(
            normalized,
            manifest.train_distortion[index],
        )
        intrinsics = manifest.train_intrinsics[index]
        projected = np.column_stack(
            (
                distorted[:, 0] * intrinsics.fx + intrinsics.cx,
                distorted[:, 1] * intrinsics.fy + intrinsics.cy,
            )
        )
        finite = np.all(np.isfinite(projected), axis=1)
        if np.any(finite):
            stored_chunks.append(stored[finite])
            projected_chunks.append(projected[finite])

    if not stored_chunks:
        raise DataContractError(
            "cannot fit COLMAP observation scale without supported projections"
        )
    return np.concatenate(stored_chunks), np.concatenate(projected_chunks)


def _fit_axis_scale(
    stored: np.ndarray,
    projected: np.ndarray,
    manifest: SceneManifest,
    axis: int,
) -> float:
    dimension = (
        manifest.train_intrinsics[0].width
        if axis == 0
        else manifest.train_intrinsics[0].height
    )
    usable = (
        (projected[:, axis] >= max(1.0, 0.05 * (dimension - 1)))
        & (projected[:, axis] <= dimension - 1 + 1e-6)
        & (stored[:, axis] >= 0.0)
    )
    if np.count_nonzero(usable) < 2:
        raise DataContractError(
            "insufficient supported observations for continuous scale fit"
        )
    return float(np.median(stored[usable, axis] / projected[usable, axis]))


def _residual_quantiles(
    mapped: np.ndarray,
    projected: np.ndarray,
) -> tuple[float, float]:
    residual = np.linalg.norm(mapped - projected, axis=1)
    return tuple(
        float(value) for value in np.percentile(residual, (50.0, 95.0))
    )


def _color_error_quantiles(
    point_ids: np.ndarray,
    colors: np.ndarray,
    valid_points: Mapping[int, object],
) -> tuple[float, float, float]:
    reference = np.stack(
        [valid_points[int(point_id)].rgb for point_id in point_ids]
    ).astype(np.float64)
    errors = np.mean(np.abs(colors.astype(np.float64) - reference), axis=1)
    return (
        float(np.mean(errors)),
        float(np.median(errors)),
        float(np.percentile(errors, 90.0)),
    )
