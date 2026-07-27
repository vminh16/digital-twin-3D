from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .colmap import read_colmap_model
from .holdout import HoldoutSplit, validate_holdout_split
from .manifest import SceneManifest
from .observation_mapping import (
    ObservationMappingDiagnostics,
    ObservationMappingFit,
    build_observation_mapping_diagnostics,
    fit_continuous_observation_mapping,
)
from .validation import DataContractError


def _readonly(values: object, dtype: np.dtype) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SparseInitialization:
    point_ids: np.ndarray
    points: np.ndarray
    colors: np.ndarray
    observation_mapping_diagnostics: ObservationMappingDiagnostics | None = None

    def __post_init__(self) -> None:
        color_source = np.asarray(self.colors)
        if (
            not np.all(np.isfinite(color_source))
            or np.any(color_source < 0)
            or np.any(color_source > 255)
            or not np.all(color_source == np.floor(color_source))
        ):
            raise ValueError("sparse colors must be integers in [0, 255]")
        point_ids = _readonly(self.point_ids, np.int64)
        points = _readonly(self.points, np.float64)
        colors = _readonly(color_source, np.uint8)
        if point_ids.ndim != 1 or points.shape != (len(point_ids), 3):
            raise ValueError("sparse point arrays must have shapes (N,) and (N, 3)")
        if colors.shape != points.shape or not np.all(np.isfinite(points)):
            raise ValueError("sparse colors must match finite points")
        object.__setattr__(self, "point_ids", point_ids)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "colors", colors)


def _bilinear_samples(image: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, image.shape[1] - 1)
    y1 = np.minimum(y0 + 1, image.shape[0] - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    top = image[y0, x0].astype(np.float64) * (1.0 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0].astype(np.float64) * (1.0 - wx) + image[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def _infer_observation_scale(
    manifest: SceneManifest,
    images_by_name: dict[str, object],
    image_names: tuple[str, ...] | None = None,
) -> np.ndarray:
    ratios_x: list[np.ndarray] = []
    ratios_y: list[np.ndarray] = []
    intrinsics_by_name = dict(
        zip(manifest.train_image_names, manifest.train_intrinsics)
    )
    selected_names = manifest.train_image_names if image_names is None else image_names
    for name in selected_names:
        intrinsics = intrinsics_by_name[name]
        registration = images_by_name.get(name)
        if registration is None:
            raise DataContractError(f"train image lacks COLMAP registration: {name}")
        coordinates = registration.points2d_xy
        if len(coordinates) == 0:
            continue
        ratios_x.append(
            coordinates[coordinates[:, 0] >= 0.0, 0]
            / max(1, intrinsics.width - 1)
        )
        ratios_y.append(
            coordinates[coordinates[:, 1] >= 0.0, 1]
            / max(1, intrinsics.height - 1)
        )
    if not ratios_x or not ratios_y:
        return np.ones(2, dtype=np.float64)
    quantiles = (
        float(np.percentile(np.concatenate(ratios_x), 99.9)),
        float(np.percentile(np.concatenate(ratios_y), 99.9)),
    )
    scale = np.asarray(
        [max(1, math.ceil(value - 1e-9)) for value in quantiles],
        dtype=np.float64,
    )
    if np.any(scale > 64):
        raise DataContractError(
            f"implausible COLMAP observation scale: {scale.tolist()}"
        )
    return scale


def _aggregate_observed_colors(
    point_id_chunks: list[np.ndarray],
    color_chunks: list[np.ndarray],
    valid_points: dict[int, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not point_id_chunks:
        raise DataContractError(
            "internal train split has no valid sparse color observations"
        )
    observed_ids = np.concatenate(point_id_chunks).astype(np.int64, copy=False)
    observed_colors = np.concatenate(color_chunks)
    order = np.argsort(observed_ids, kind="stable")
    observed_ids = observed_ids[order]
    observed_colors = observed_colors[order]
    point_ids, starts = np.unique(observed_ids, return_index=True)
    ends = np.r_[starts[1:], len(observed_ids)]
    colors = np.stack(
        [
            np.median(observed_colors[start:end], axis=0)
            for start, end in zip(starts, ends)
        ]
    )
    colors = np.floor(colors + 0.5).astype(np.uint8)
    points = np.stack([valid_points[int(point_id)].xyz for point_id in point_ids])
    return point_ids, points, colors


def build_split_sparse_initialization(
    manifest: SceneManifest,
    scene_root: Path,
    split: HoldoutSplit,
    *,
    observation_mapping_mode: str = "legacy-ceil",
) -> SparseInitialization:
    if observation_mapping_mode not in {
        "legacy-ceil",
        "continuous-reprojection",
    }:
        raise ValueError("observation_mapping_mode is unsupported")
    validate_holdout_split(split, manifest)
    root = Path(scene_root)
    model = read_colmap_model(root / "train" / "sparse" / "0")
    images_by_name = {image.name: image for image in model.images.values()}
    paths_by_name = dict(zip(manifest.train_image_names, manifest.train_image_paths))
    intrinsics_by_name = dict(zip(manifest.train_image_names, manifest.train_intrinsics))
    valid_points = {
        point_id: point
        for point_id, point in model.points3d.items()
        if np.all(np.isfinite(point.xyz))
    }
    legacy_scale = _infer_observation_scale(
        manifest,
        images_by_name,
        (
            split.train_image_names
            if observation_mapping_mode == "continuous-reprojection"
            else None
        ),
    )
    observation_scale = legacy_scale
    mapping_fit: ObservationMappingFit | None = None
    if observation_mapping_mode == "continuous-reprojection":
        mapping_fit = fit_continuous_observation_mapping(
            manifest,
            images_by_name,
            valid_points,
            legacy_scale,
            split.train_image_names,
        )
        observation_scale = np.asarray(mapping_fit.scale, dtype=np.float64)

    selected_id_chunks: list[np.ndarray] = []
    selected_color_chunks: list[np.ndarray] = []
    legacy_id_chunks: list[np.ndarray] = []
    legacy_color_chunks: list[np.ndarray] = []
    for name in split.train_image_names:
        registration = images_by_name.get(name)
        if registration is None:
            raise DataContractError(
                f"internal train image lacks COLMAP registration: {name}"
            )
        path = root / paths_by_name[name]
        with Image.open(path) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
        intrinsics = intrinsics_by_name[name]
        if image.shape[:2] != (intrinsics.height, intrinsics.width):
            raise DataContractError(f"image resolution does not match intrinsics: {name}")

        mappings = [
            (
                observation_scale,
                selected_id_chunks,
                selected_color_chunks,
            )
        ]
        if observation_mapping_mode == "continuous-reprojection":
            mappings.append(
                (
                    legacy_scale,
                    legacy_id_chunks,
                    legacy_color_chunks,
                )
            )
        for scale, point_id_chunks, color_chunks in mappings:
            coordinates = registration.points2d_xy / scale
            point_ids = registration.point3d_ids
            inside = (
                (coordinates[:, 0] >= 0.0)
                & (coordinates[:, 0] <= image.shape[1] - 1)
                & (coordinates[:, 1] >= 0.0)
                & (coordinates[:, 1] <= image.shape[0] - 1)
            )
            candidate_ids = point_ids[inside]
            candidate_coordinates = coordinates[inside]
            supported = np.fromiter(
                (int(point_id) in valid_points for point_id in candidate_ids),
                dtype=bool,
                count=len(candidate_ids),
            )
            if np.any(supported):
                point_id_chunks.append(candidate_ids[supported])
                color_chunks.append(
                    _bilinear_samples(
                        image,
                        candidate_coordinates[supported],
                    )
                )

    point_ids, points, colors = _aggregate_observed_colors(
        selected_id_chunks,
        selected_color_chunks,
        valid_points,
    )
    diagnostics = None
    if observation_mapping_mode == "continuous-reprojection":
        legacy_ids, _, legacy_colors = _aggregate_observed_colors(
            legacy_id_chunks,
            legacy_color_chunks,
            valid_points,
        )
        common_ids, selected_indices, legacy_indices = np.intersect1d(
            point_ids,
            legacy_ids,
            assume_unique=True,
            return_indices=True,
        )
        if len(common_ids) == 0:
            raise DataContractError(
                "continuous and legacy mappings have no common sparse points"
            )
        if mapping_fit is None:
            raise RuntimeError("continuous observation mapping fit is missing")
        diagnostics = build_observation_mapping_diagnostics(
            mapping_fit,
            common_ids,
            colors[selected_indices],
            legacy_colors[legacy_indices],
            valid_points,
        )
    return SparseInitialization(
        point_ids=point_ids,
        points=points,
        colors=colors,
        observation_mapping_diagnostics=diagnostics,
    )
