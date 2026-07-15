from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .manifest import SceneManifest
from .validation import DataContractError


SCHEMA_VERSION = 1
ALGORITHM = "pose_fps_guard2_v1"


@dataclass(frozen=True)
class HoldoutSplit:
    schema_version: int
    scene_id: str
    manifest_sha256: str
    algorithm: str
    train_image_names: tuple[str, ...]
    validation_image_names: tuple[str, ...]
    guard_image_names: tuple[str, ...]


def _update_text(hasher: object, value: str) -> None:
    encoded = value.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, "little"))
    hasher.update(encoded)


def manifest_holdout_sha256(manifest: SceneManifest) -> str:
    if len(set(manifest.train_image_names)) != len(manifest.train_image_names):
        raise DataContractError("train image names must be unique")
    indices = sorted(
        range(len(manifest.train_image_names)),
        key=lambda index: manifest.train_image_names[index],
    )
    hasher = hashlib.sha256()
    hasher.update(b"bts_nvs_holdout_manifest_v1\0")
    _update_text(hasher, manifest.scene_id)
    for index in indices:
        name = manifest.train_image_names[index]
        intrinsics = manifest.train_intrinsics[index]
        distortion = manifest.train_distortion[index]
        _update_text(hasher, name)
        _update_text(hasher, manifest.train_image_paths[index])
        _update_text(hasher, distortion.model)
        hasher.update(
            np.asarray(manifest.train_camera_to_world[index], dtype="<f8").tobytes(
                order="C"
            )
        )
        hasher.update(
            np.asarray(
                [
                    intrinsics.width,
                    intrinsics.height,
                    intrinsics.fx,
                    intrinsics.fy,
                    intrinsics.cx,
                    intrinsics.cy,
                    *distortion.coefficients,
                ],
                dtype="<f8",
            ).tobytes(order="C")
        )
    hasher.update(
        np.asarray(manifest.normalization_transform, dtype="<f8").tobytes(order="C")
    )
    return hasher.hexdigest()


def _pose_distance_matrix(centers: np.ndarray, axes: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    axes = np.asarray(axes, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 3 or axes.shape != centers.shape:
        raise DataContractError("camera centers and axes must have shape (N, 3)")
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(axes)):
        raise DataContractError("camera centers and axes must be finite")
    norms = np.linalg.norm(axes, axis=1)
    if np.any(norms <= 1e-12):
        raise DataContractError("camera optical axes must be non-zero")
    unit_axes = axes / norms[:, None]
    translation = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    cosine = np.clip(unit_axes @ unit_axes.T, -1.0, 1.0)
    result = translation + 0.25 * np.arccos(cosine) / np.pi
    np.fill_diagonal(result, 0.0)
    return result


def _ordered_pose_geometry(
    manifest: SceneManifest,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    ordered = sorted(
        zip(manifest.train_image_names, manifest.train_camera_to_world),
        key=lambda item: item[0],
    )
    names = tuple(item[0] for item in ordered)
    camera_to_world = np.stack([item[1] for item in ordered])
    centers = camera_to_world[:, :3, 3]
    transform = manifest.normalization_transform
    normalized_centers = centers @ transform[:3, :3].T + transform[:3, 3]
    return names, normalized_centers, camera_to_world[:, :3, 2]


def manifest_pose_distance_matrix(
    manifest: SceneManifest,
) -> tuple[tuple[str, ...], np.ndarray]:
    names, centers, axes = _ordered_pose_geometry(manifest)
    distances = _pose_distance_matrix(centers, axes)
    distances.setflags(write=False)
    return names, distances


def _guard_indices(
    validation: list[int],
    distances: np.ndarray,
    names: tuple[str, ...],
) -> set[int]:
    validation_set = set(validation)
    guard: set[int] = set()
    for index in validation:
        candidates = (
            candidate
            for candidate in range(len(names))
            if candidate not in validation_set
        )
        nearest = sorted(
            candidates,
            key=lambda candidate: (distances[index, candidate], names[candidate]),
        )[:2]
        guard.update(nearest)
    return guard


def build_pose_holdout(manifest: SceneManifest) -> HoldoutSplit:
    count = len(manifest.train_image_names)
    if count < 8:
        raise DataContractError("holdout requires at least 8 train cameras")
    if len(set(manifest.train_image_names)) != count:
        raise DataContractError("train image names must be unique")
    overlap = set(manifest.train_image_names) & set(manifest.test_image_names)
    if overlap:
        raise DataContractError("official test names overlap physical train names")

    names, normalized_centers, axes = _ordered_pose_geometry(manifest)
    distances = _pose_distance_matrix(normalized_centers, axes)

    mean_center = normalized_centers.mean(axis=0)
    seed = min(
        range(count),
        key=lambda index: (
            np.linalg.norm(normalized_centers[index] - mean_center),
            names[index],
        ),
    )
    target = max(8, math.floor(count / 8 + 0.5))
    validation = [seed]
    while len(validation) < target:
        selected = set(validation)
        candidate = min(
            (index for index in range(count) if index not in selected),
            key=lambda index: (
                -float(distances[index, validation].min()),
                names[index],
            ),
        )
        validation.append(candidate)

    minimum_train = max(120, math.ceil(0.70 * count))
    while True:
        guard = _guard_indices(validation, distances, names)
        train = set(range(count)) - set(validation) - guard
        if len(train) >= minimum_train:
            break
        validation.pop()
        if len(validation) < 8:
            raise DataContractError(
                "cannot retain 8 validation images and minimum train coverage"
            )

    split = HoldoutSplit(
        schema_version=SCHEMA_VERSION,
        scene_id=manifest.scene_id,
        manifest_sha256=manifest_holdout_sha256(manifest),
        algorithm=ALGORITHM,
        train_image_names=tuple(
            names[index] for index in sorted(train, key=lambda i: names[i])
        ),
        validation_image_names=tuple(names[index] for index in validation),
        guard_image_names=tuple(
            names[index] for index in sorted(guard, key=lambda i: names[i])
        ),
    )
    _validate_partition(split, manifest)
    return split


def _validate_partition(split: HoldoutSplit, manifest: SceneManifest) -> None:
    if split.schema_version != SCHEMA_VERSION:
        raise DataContractError(f"unsupported holdout schema: {split.schema_version}")
    if split.algorithm != ALGORITHM:
        raise DataContractError(f"unsupported holdout algorithm: {split.algorithm}")
    if split.scene_id != manifest.scene_id:
        raise DataContractError("holdout scene_id does not match manifest")
    if split.manifest_sha256 != manifest_holdout_sha256(manifest):
        raise DataContractError("holdout manifest hash mismatch")

    groups = (
        split.train_image_names,
        split.validation_image_names,
        split.guard_image_names,
    )
    if any(len(group) != len(set(group)) for group in groups):
        raise DataContractError("holdout groups must not contain duplicate names")
    train, validation, guard = map(set, groups)
    if train & validation or train & guard or validation & guard:
        raise DataContractError("holdout groups must be pairwise disjoint")
    if train | validation | guard != set(manifest.train_image_names):
        raise DataContractError("holdout groups must cover exact manifest train names")
    if (train | validation | guard) & set(manifest.test_image_names):
        raise DataContractError("holdout contains official test names")
    if len(validation) < 8:
        raise DataContractError("holdout requires at least 8 validation images")
    if len(train) < max(120, math.ceil(0.70 * len(manifest.train_image_names))):
        raise DataContractError("holdout does not retain enough internal train images")


def validate_holdout_split(split: HoldoutSplit, manifest: SceneManifest) -> None:
    _validate_partition(split, manifest)
    if split != build_pose_holdout(manifest):
        raise DataContractError("holdout partition does not match deterministic algorithm")


def save_holdout_split(split: HoldoutSplit, path: Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(asdict(split), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_holdout_split(path: Path, manifest: SceneManifest) -> HoldoutSplit:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if set(payload) != {
            "schema_version",
            "scene_id",
            "manifest_sha256",
            "algorithm",
            "train_image_names",
            "validation_image_names",
            "guard_image_names",
        }:
            raise DataContractError("holdout JSON fields are invalid")
        split = HoldoutSplit(
            schema_version=payload["schema_version"],
            scene_id=payload["scene_id"],
            manifest_sha256=payload["manifest_sha256"],
            algorithm=payload["algorithm"],
            train_image_names=tuple(payload["train_image_names"]),
            validation_image_names=tuple(payload["validation_image_names"]),
            guard_image_names=tuple(payload["guard_image_names"]),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise DataContractError("cannot load holdout JSON") from error
    validate_holdout_split(split, manifest)
    return split
