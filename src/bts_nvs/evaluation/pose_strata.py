from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from bts_nvs.data.holdout import (
    HoldoutSplit,
    manifest_pose_distance_matrix,
    validate_holdout_split,
)
from bts_nvs.data.manifest import SceneManifest


ALGORITHM = "nearest_train_tertiles_v1"
STRATA = ("easy", "medium", "hard")


def assign_pose_strata(distances: Mapping[str, float]) -> dict[str, str]:
    if len(distances) < len(STRATA):
        raise ValueError("pose strata require at least three validation images")
    validated: dict[str, float] = {}
    for name, value in distances.items():
        if not isinstance(name, str) or not name:
            raise ValueError("pose distance names must be non-empty strings")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError("pose distances must be finite and non-negative")
        validated[name] = float(value)

    ordered_names = tuple(
        sorted(validated, key=lambda name: (validated[name], name))
    )
    assignments: dict[str, str] = {}
    for label, group in zip(STRATA, np.array_split(ordered_names, len(STRATA))):
        for name in group:
            assignments[str(name)] = label
    return {name: assignments[name] for name in sorted(assignments)}


def _pose_geometry(
    manifest: SceneManifest,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    centers = manifest.train_camera_to_world[:, :3, 3]
    transform = manifest.normalization_transform
    normalized_centers = centers @ transform[:3, :3].T + transform[:3, 3]
    axes = manifest.train_camera_to_world[:, :3, 2]
    norms = np.linalg.norm(axes, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("camera optical axes must be non-zero")
    unit_axes = axes / norms[:, None]
    return (
        {
            name: normalized_centers[index]
            for index, name in enumerate(manifest.train_image_names)
        },
        {
            name: unit_axes[index]
            for index, name in enumerate(manifest.train_image_names)
        },
    )


def build_pose_strata(
    manifest: SceneManifest,
    split: HoldoutSplit,
) -> dict[str, object]:
    validate_holdout_split(split, manifest)
    names, distances = manifest_pose_distance_matrix(manifest)
    index_by_name = {name: index for index, name in enumerate(names)}
    centers, axes = _pose_geometry(manifest)

    nearest: dict[str, tuple[str, float]] = {}
    for validation_name in split.validation_image_names:
        validation_index = index_by_name[validation_name]
        train_name = min(
            split.train_image_names,
            key=lambda candidate: (
                float(distances[validation_index, index_by_name[candidate]]),
                candidate,
            ),
        )
        nearest[validation_name] = (
            train_name,
            float(distances[validation_index, index_by_name[train_name]]),
        )

    assignments = assign_pose_strata(
        {name: value[1] for name, value in nearest.items()}
    )
    images: dict[str, dict[str, float | str]] = {}
    for validation_name in sorted(split.validation_image_names):
        train_name, pose_distance = nearest[validation_name]
        center_distance = float(
            np.linalg.norm(centers[validation_name] - centers[train_name])
        )
        cosine = float(
            np.clip(axes[validation_name] @ axes[train_name], -1.0, 1.0)
        )
        rotation_angle_deg = float(np.degrees(np.arccos(cosine)))
        values = (pose_distance, center_distance, rotation_angle_deg)
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("pose stratum values must be finite and non-negative")
        images[validation_name] = {
            "nearest_train_image_name": train_name,
            "pose_distance": pose_distance,
            "center_distance": center_distance,
            "rotation_angle_deg": rotation_angle_deg,
            "stratum": assignments[validation_name],
        }

    return {
        "schema_version": 1,
        "scene_id": manifest.scene_id,
        "algorithm": ALGORITHM,
        "holdout_algorithm": split.algorithm,
        "holdout_manifest_sha256": split.manifest_sha256,
        "image_count": len(images),
        "images": images,
    }


def save_pose_strata(report: Mapping[str, object], path: Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, output)
