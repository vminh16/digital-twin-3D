from __future__ import annotations

import csv
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from bts_nvs.cameras.distortion import CameraDistortion
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.cameras.poses import (
    compute_scene_normalization,
    invert_rigid_transform,
    world_to_camera_from_qt,
)

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
SCHEMA_VERSION = 1


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


def _readonly_array(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    result = np.asarray(array, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SceneManifest:
    schema_version: int
    scene_id: str
    train_image_paths: tuple[str, ...]
    train_image_names: tuple[str, ...]
    train_world_to_camera: np.ndarray
    train_camera_to_world: np.ndarray
    train_intrinsics: tuple[CameraIntrinsics, ...]
    train_distortion: tuple[CameraDistortion, ...]
    test_image_names: tuple[str, ...]
    test_output_names: tuple[str, ...]
    test_world_to_camera: np.ndarray
    test_intrinsics: tuple[CameraIntrinsics, ...]
    test_distortion: tuple[CameraDistortion, ...]
    sparse_points: np.ndarray
    sparse_colors: np.ndarray
    normalization_transform: np.ndarray
    inverse_normalization_transform: np.ndarray

    def __post_init__(self) -> None:
        colors = np.asarray(self.sparse_colors)
        if (
            not np.all(np.isfinite(colors))
            or np.any(colors < 0)
            or np.any(colors > 255)
            or not np.all(colors == np.floor(colors))
        ):
            raise ValueError("sparse colors must be integers in [0, 255]")
        for field in (
            "train_world_to_camera",
            "train_camera_to_world",
            "test_world_to_camera",
            "sparse_points",
            "normalization_transform",
            "inverse_normalization_transform",
        ):
            object.__setattr__(self, field, _readonly_array(getattr(self, field), np.float64))
        object.__setattr__(self, "sparse_colors", _readonly_array(colors, np.uint8))


def build_scene_manifest(scene_root: Path) -> SceneManifest:
    root = Path(scene_root)
    source = load_scene_source_data(root)
    train_w2c = np.stack([image.world_to_camera for image in source.train_images])
    train_c2w = np.stack([invert_rigid_transform(pose) for pose in train_w2c])
    normalization = compute_scene_normalization(train_c2w)
    manifest = SceneManifest(
        schema_version=SCHEMA_VERSION,
        scene_id=source.scene_id,
        train_image_paths=source.train_image_paths,
        train_image_names=source.train_image_names,
        train_world_to_camera=train_w2c,
        train_camera_to_world=train_c2w,
        train_intrinsics=tuple(source.cameras[image.camera_id].intrinsics for image in source.train_images),
        train_distortion=tuple(source.cameras[image.camera_id].distortion for image in source.train_images),
        test_image_names=tuple(pose.image_name for pose in source.test_poses),
        test_output_names=tuple(pose.output_name for pose in source.test_poses),
        test_world_to_camera=np.stack([pose.world_to_camera for pose in source.test_poses]),
        test_intrinsics=tuple(pose.intrinsics for pose in source.test_poses),
        test_distortion=source.test_distortions,
        sparse_points=np.stack([point.xyz for point in source.sparse_points]) if source.sparse_points else np.empty((0, 3)),
        sparse_colors=np.stack([point.rgb for point in source.sparse_points]) if source.sparse_points else np.empty((0, 3), dtype=np.uint8),
        normalization_transform=normalization.transform,
        inverse_normalization_transform=normalization.inverse_transform,
    )
    validate_scene_manifest(manifest, root)
    return manifest


def _validate_pose_array(name: str, poses: np.ndarray, count: int) -> None:
    if poses.shape != (count, 4, 4):
        raise DataContractError(f"{name} must have shape ({count}, 4, 4)")
    if poses.dtype != np.float64 or not np.all(np.isfinite(poses)):
        raise DataContractError(f"{name} must be finite float64")
    try:
        for pose in poses:
            invert_rigid_transform(pose)
    except ValueError as error:
        raise DataContractError(f"invalid {name}: {error}") from error


def validate_scene_manifest(manifest: SceneManifest, scene_root: Path) -> None:
    if manifest.schema_version != SCHEMA_VERSION:
        raise DataContractError(f"unsupported schema version: {manifest.schema_version}")
    if not manifest.scene_id:
        raise DataContractError("scene_id must not be empty")
    train_count = len(manifest.train_image_names)
    test_count = len(manifest.test_image_names)
    if train_count == 0 or test_count == 0:
        raise DataContractError("manifest must contain train and test cameras")
    train_fields = (
        manifest.train_image_paths,
        manifest.train_intrinsics,
        manifest.train_distortion,
    )
    test_fields = (
        manifest.test_output_names,
        manifest.test_intrinsics,
        manifest.test_distortion,
    )
    if any(len(field) != train_count for field in train_fields):
        raise DataContractError("train manifest fields have inconsistent lengths")
    if any(len(field) != test_count for field in test_fields):
        raise DataContractError("test manifest fields have inconsistent lengths")
    if len(set(manifest.train_image_names)) != train_count:
        raise DataContractError("train image names must be unique")
    if len({name.casefold() for name in manifest.test_output_names}) != test_count:
        raise DataContractError("test output names must be unique")
    for image_name, output_name in zip(manifest.test_image_names, manifest.test_output_names):
        if output_name != output_name_from_test_image_name(image_name):
            raise DataContractError(f"invalid canonical test output name: {output_name}")

    root = Path(scene_root).resolve()
    for relative, image_name in zip(manifest.train_image_paths, manifest.train_image_names):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or path.name != image_name:
            raise DataContractError(f"invalid relative train image path: {relative}")
        try:
            resolved = (root / path).resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as error:
            raise DataContractError(f"train image path escapes or is missing from scene root: {relative}") from error
        if not resolved.is_file():
            raise DataContractError(f"train image path is not a file: {relative}")

    _validate_pose_array("train_world_to_camera", manifest.train_world_to_camera, train_count)
    _validate_pose_array("train_camera_to_world", manifest.train_camera_to_world, train_count)
    _validate_pose_array("test_world_to_camera", manifest.test_world_to_camera, test_count)
    identity = np.eye(4, dtype=np.float64)
    for w2c, c2w in zip(manifest.train_world_to_camera, manifest.train_camera_to_world):
        if not np.allclose(w2c @ c2w, identity, atol=1e-10, rtol=0.0):
            raise DataContractError("train W2C/C2W transforms are not inverses")
    if manifest.sparse_points.shape != manifest.sparse_colors.shape or manifest.sparse_points.ndim != 2 or manifest.sparse_points.shape[1:] != (3,):
        raise DataContractError("sparse points and colors must have matching shape (N, 3)")
    if manifest.sparse_points.dtype != np.float64 or manifest.sparse_colors.dtype != np.uint8:
        raise DataContractError("sparse arrays use non-canonical dtype")
    if not np.all(np.isfinite(manifest.sparse_points)):
        raise DataContractError("sparse points must be finite")
    for field in ("normalization_transform", "inverse_normalization_transform"):
        value = getattr(manifest, field)
        if value.shape != (4, 4) or value.dtype != np.float64 or not np.all(np.isfinite(value)):
            raise DataContractError(f"{field} must be finite float64 with shape (4, 4)")
    if not np.allclose(manifest.normalization_transform @ manifest.inverse_normalization_transform, identity, atol=1e-10, rtol=0.0):
        raise DataContractError("normalization transforms are not inverses")
    try:
        expected_normalization = compute_scene_normalization(manifest.train_camera_to_world)
    except ValueError as error:
        raise DataContractError(f"invalid train trajectory normalization: {error}") from error
    if not np.allclose(
        manifest.normalization_transform,
        expected_normalization.transform,
        atol=1e-10,
        rtol=0.0,
    ):
        raise DataContractError("normalization transform does not match train trajectory")
    for field in SceneManifest.__dataclass_fields__:
        value = getattr(manifest, field)
        if isinstance(value, np.ndarray) and value.flags.writeable:
            raise DataContractError(f"manifest array must be read-only: {field}")


def _intrinsics_dict(value: CameraIntrinsics) -> dict[str, int | float]:
    return {field: getattr(value, field) for field in ("width", "height", "fx", "fy", "cx", "cy")}


def _distortion_dict(value: CameraDistortion) -> dict[str, object]:
    return {"model": value.model, "coefficients": list(value.coefficients)}


def save_scene_manifest(manifest: SceneManifest, output_dir: Path) -> None:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"manifest artifact already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        np.savez_compressed(
            temporary / "arrays.npz",
            train_world_to_camera=manifest.train_world_to_camera,
            train_camera_to_world=manifest.train_camera_to_world,
            test_world_to_camera=manifest.test_world_to_camera,
            sparse_points=manifest.sparse_points,
            sparse_colors=manifest.sparse_colors,
            normalization_transform=manifest.normalization_transform,
            inverse_normalization_transform=manifest.inverse_normalization_transform,
        )
        metadata = {
            "schema_version": manifest.schema_version,
            "scene_id": manifest.scene_id,
            "arrays_file": "arrays.npz",
            "train_image_paths": list(manifest.train_image_paths),
            "train_image_names": list(manifest.train_image_names),
            "train_intrinsics": [_intrinsics_dict(value) for value in manifest.train_intrinsics],
            "train_distortion": [_distortion_dict(value) for value in manifest.train_distortion],
            "test_image_names": list(manifest.test_image_names),
            "test_output_names": list(manifest.test_output_names),
            "test_intrinsics": [_intrinsics_dict(value) for value in manifest.test_intrinsics],
            "test_distortion": [_distortion_dict(value) for value in manifest.test_distortion],
        }
        (temporary / "manifest.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_scene_manifest(manifest_json: Path, scene_root: Path) -> SceneManifest:
    json_path = Path(manifest_json)
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DataContractError(f"cannot read manifest JSON: {json_path}") from error
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise DataContractError(f"unsupported schema version: {metadata.get('schema_version')}")
    if metadata.get("arrays_file") != "arrays.npz":
        raise DataContractError("manifest arrays_file must be arrays.npz")
    try:
        with np.load(json_path.parent / "arrays.npz", allow_pickle=False) as arrays:
            manifest = SceneManifest(
                schema_version=metadata["schema_version"],
                scene_id=metadata["scene_id"],
                train_image_paths=tuple(metadata["train_image_paths"]),
                train_image_names=tuple(metadata["train_image_names"]),
                train_world_to_camera=arrays["train_world_to_camera"],
                train_camera_to_world=arrays["train_camera_to_world"],
                train_intrinsics=tuple(CameraIntrinsics(**value) for value in metadata["train_intrinsics"]),
                train_distortion=tuple(CameraDistortion(value["model"], tuple(value["coefficients"])) for value in metadata["train_distortion"]),
                test_image_names=tuple(metadata["test_image_names"]),
                test_output_names=tuple(metadata["test_output_names"]),
                test_world_to_camera=arrays["test_world_to_camera"],
                test_intrinsics=tuple(CameraIntrinsics(**value) for value in metadata["test_intrinsics"]),
                test_distortion=tuple(CameraDistortion(value["model"], tuple(value["coefficients"])) for value in metadata["test_distortion"]),
                sparse_points=arrays["sparse_points"],
                sparse_colors=arrays["sparse_colors"],
                normalization_transform=arrays["normalization_transform"],
                inverse_normalization_transform=arrays["inverse_normalization_transform"],
            )
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise DataContractError("manifest artifact is invalid") from error
    validate_scene_manifest(manifest, scene_root)
    return manifest
