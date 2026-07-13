from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from bts_nvs.cameras.distortion import CameraDistortion
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.cameras.poses import world_to_camera_from_qt

from .colmap import (
    ColmapCameraRecord,
    ColmapImageRecord,
    ColmapModel,
    ColmapPointRecord,
    read_colmap_model,
)
from .validation import DataContractError


TEST_POSE_COLUMNS = (
    "image_name",
    "qw",
    "qx",
    "qy",
    "qz",
    "tx",
    "ty",
    "tz",
    "fx",
    "fy",
    "cx",
    "cy",
    "width",
    "height",
)
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
WINDOWS_FORBIDDEN_CHARACTERS = set('<>:"|?*')


def _readonly_float64(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array, dtype=np.float64).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TestPoseRecord:
    image_name: str
    output_name: str
    world_to_camera: np.ndarray
    intrinsics: CameraIntrinsics

    def __post_init__(self) -> None:
        transform = _readonly_float64(self.world_to_camera)
        if transform.shape != (4, 4):
            raise ValueError("test world_to_camera must have shape (4, 4)")
        object.__setattr__(self, "world_to_camera", transform)


@dataclass(frozen=True)
class SceneSourceData:
    scene_id: str
    train_image_paths: tuple[str, ...]
    train_image_names: tuple[str, ...]
    train_images: tuple[ColmapImageRecord, ...]
    test_poses: tuple[TestPoseRecord, ...]
    test_distortions: tuple[CameraDistortion, ...]
    sparse_points: tuple[ColmapPointRecord, ...]
    cameras: Mapping[int, ColmapCameraRecord]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cameras", MappingProxyType(dict(self.cameras)))


def output_name_from_test_image_name(name: str) -> str:
    if not name or name in {".", ".."}:
        raise DataContractError("test image name must be a non-empty filename")
    if "/" in name or "\\" in name or Path(name).is_absolute() or Path(name).name != name:
        raise DataContractError(f"test image name must be a basename: {name!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise DataContractError(f"test image name contains control characters: {name!r}")
    if any(character in WINDOWS_FORBIDDEN_CHARACTERS for character in name):
        raise DataContractError(f"test image name contains forbidden characters: {name!r}")
    if name.endswith((" ", ".")):
        raise DataContractError(f"test image name cannot end with a space or dot: {name!r}")
    device_name = name.split(".", maxsplit=1)[0].casefold()
    if device_name in WINDOWS_RESERVED_NAMES:
        raise DataContractError(f"test image name is reserved on Windows: {name!r}")
    source = Path(name)
    if not source.suffix:
        raise DataContractError(f"test image name must have a suffix: {name!r}")
    return source.with_suffix(".png").name


def _finite_float(row: Mapping[str, str], field: str, row_number: int) -> float:
    try:
        value = float(row[field])
    except (TypeError, ValueError) as error:
        raise DataContractError(f"row {row_number}: {field} must be numeric") from error
    if not np.isfinite(value):
        raise DataContractError(f"row {row_number}: {field} must be finite")
    return value


def _positive_int(row: Mapping[str, str], field: str, row_number: int) -> int:
    try:
        value = int(row[field])
    except (TypeError, ValueError) as error:
        raise DataContractError(f"row {row_number}: {field} must be an integer") from error
    if value <= 0:
        raise DataContractError(f"row {row_number}: {field} must be positive")
    return value


def read_test_poses_csv(path: Path) -> tuple[TestPoseRecord, ...]:
    csv_path = Path(path)
    try:
        handle = csv_path.open("r", newline="", encoding="utf-8-sig")
    except OSError as error:
        raise DataContractError(f"cannot open test pose CSV: {csv_path}") from error
    with handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != TEST_POSE_COLUMNS:
            raise DataContractError(
                f"test pose CSV columns must be exactly {TEST_POSE_COLUMNS}"
            )
        poses: list[TestPoseRecord] = []
        image_names: set[str] = set()
        output_names: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(row.get(column) is None for column in TEST_POSE_COLUMNS):
                raise DataContractError(
                    f"row {row_number}: expected exactly 14 values"
                )
            image_name = row["image_name"]
            output_name = output_name_from_test_image_name(image_name)
            image_key = image_name.casefold()
            output_key = output_name.casefold()
            if image_key in image_names:
                raise DataContractError(f"duplicate test image name: {image_name}")
            if output_key in output_names:
                raise DataContractError(f"duplicate canonical output name: {output_name}")
            image_names.add(image_key)
            output_names.add(output_key)
            qvec = np.asarray(
                [_finite_float(row, field, row_number) for field in ("qw", "qx", "qy", "qz")],
                dtype=np.float64,
            )
            tvec = np.asarray(
                [_finite_float(row, field, row_number) for field in ("tx", "ty", "tz")],
                dtype=np.float64,
            )
            try:
                intrinsics = CameraIntrinsics(
                    width=_positive_int(row, "width", row_number),
                    height=_positive_int(row, "height", row_number),
                    fx=_finite_float(row, "fx", row_number),
                    fy=_finite_float(row, "fy", row_number),
                    cx=_finite_float(row, "cx", row_number),
                    cy=_finite_float(row, "cy", row_number),
                )
                world_to_camera = world_to_camera_from_qt(qvec, tvec)
            except ValueError as error:
                raise DataContractError(f"row {row_number}: {error}") from error
            poses.append(
                TestPoseRecord(
                    image_name=image_name,
                    output_name=output_name,
                    world_to_camera=world_to_camera,
                    intrinsics=intrinsics,
                )
            )
    if not poses:
        raise DataContractError("test pose CSV must contain at least one pose")
    return tuple(poses)


def _rotation_angle(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    relative = rotation_a @ rotation_b.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.arccos(cosine))


def _cross_check_test_camera(
    pose: TestPoseRecord,
    image: ColmapImageRecord,
    camera: ColmapCameraRecord,
) -> None:
    rotation_error = _rotation_angle(
        pose.world_to_camera[:3, :3], image.world_to_camera[:3, :3]
    )
    if rotation_error > 1e-5:
        raise DataContractError(
            f"test pose rotation mismatch for {pose.image_name}: {rotation_error} rad"
        )
    translation_error = float(
        np.max(np.abs(pose.world_to_camera[:3, 3] - image.world_to_camera[:3, 3]))
    )
    if translation_error > 1e-6:
        raise DataContractError(
            f"test pose translation mismatch for {pose.image_name}: {translation_error}"
        )
    csv_values = np.asarray(
        [
            pose.intrinsics.width,
            pose.intrinsics.height,
            pose.intrinsics.fx,
            pose.intrinsics.fy,
            pose.intrinsics.cx,
            pose.intrinsics.cy,
        ],
        dtype=np.float64,
    )
    colmap_values = np.asarray(
        [
            camera.intrinsics.width,
            camera.intrinsics.height,
            camera.intrinsics.fx,
            camera.intrinsics.fy,
            camera.intrinsics.cx,
            camera.intrinsics.cy,
        ],
        dtype=np.float64,
    )
    if not np.allclose(csv_values, colmap_values, atol=1e-6, rtol=0.0):
        raise DataContractError(f"test intrinsics mismatch for {pose.image_name}")


def load_scene_source_data(
    scene_root: Path,
    *,
    colmap_model: ColmapModel | None = None,
) -> SceneSourceData:
    root = Path(scene_root)
    train_image_dir = root / "train" / "images"
    if not train_image_dir.is_dir():
        raise DataContractError(f"missing train image directory: {train_image_dir}")
    model = colmap_model or read_colmap_model(root / "train" / "sparse" / "0")
    physical_names = sorted(
        path.name
        for path in train_image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    if not physical_names:
        raise DataContractError(f"no train images found in {train_image_dir}")
    images_by_name = {image.name: image for image in model.images.values()}
    missing_poses = [name for name in physical_names if name not in images_by_name]
    if missing_poses:
        raise DataContractError(f"train images missing COLMAP poses: {missing_poses}")
    train_images = tuple(images_by_name[name] for name in physical_names)
    train_image_ids = {image.image_id for image in train_images}

    test_poses = read_test_poses_csv(root / "test" / "test_poses.csv")
    test_distortions: list[CameraDistortion] = []
    for pose in test_poses:
        image = images_by_name.get(pose.image_name)
        if image is None:
            raise DataContractError(
                f"test image lacks matching COLMAP registration: {pose.image_name}"
            )
        camera = model.cameras[image.camera_id]
        _cross_check_test_camera(pose, image, camera)
        test_distortions.append(camera.distortion)

    sparse_points = tuple(
        point
        for point in model.points3d.values()
        if np.all(np.isfinite(point.xyz))
        and any(image_id in train_image_ids for image_id in point.image_ids)
    )
    return SceneSourceData(
        scene_id=root.name,
        train_image_paths=tuple(f"train/images/{name}" for name in physical_names),
        train_image_names=tuple(physical_names),
        train_images=train_images,
        test_poses=test_poses,
        test_distortions=tuple(test_distortions),
        sparse_points=sparse_points,
        cameras=model.cameras,
    )
