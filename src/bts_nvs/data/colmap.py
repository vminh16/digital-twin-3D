from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from bts_nvs.cameras.distortion import CameraDistortion
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.cameras.poses import invert_rigid_transform


PYCOLMAP_VERSION = "4.1.0"


def _readonly(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    result = np.asarray(array, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ColmapCameraRecord:
    camera_id: int
    intrinsics: CameraIntrinsics
    distortion: CameraDistortion

    def __post_init__(self) -> None:
        if isinstance(self.camera_id, bool) or not isinstance(self.camera_id, int):
            raise ValueError("camera_id must be an integer")
        if self.distortion.model == "SIMPLE_RADIAL" and not np.isclose(
            self.intrinsics.fx,
            self.intrinsics.fy,
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError("SIMPLE_RADIAL requires fx approximately equal to fy")


@dataclass(frozen=True)
class ColmapImageRecord:
    image_id: int
    name: str
    camera_id: int
    world_to_camera: np.ndarray
    points2d_xy: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.float64)
    )
    point3d_ids: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int64)
    )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("COLMAP image name must not be empty")
        transform = _readonly(self.world_to_camera, np.float64)
        invert_rigid_transform(transform)
        points2d_xy = _readonly(self.points2d_xy, np.float64)
        point3d_ids = _readonly(self.point3d_ids, np.int64)
        if (
            points2d_xy.ndim != 2
            or points2d_xy.shape[1:] != (2,)
            or point3d_ids.shape != (len(points2d_xy),)
            or not np.all(np.isfinite(points2d_xy))
        ):
            raise ValueError("COLMAP observation arrays must have aligned shapes")
        object.__setattr__(self, "world_to_camera", transform)
        object.__setattr__(self, "points2d_xy", points2d_xy)
        object.__setattr__(self, "point3d_ids", point3d_ids)


@dataclass(frozen=True)
class ColmapPointRecord:
    point_id: int
    xyz: np.ndarray
    rgb: np.ndarray
    reprojection_error: float
    image_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        xyz = _readonly(self.xyz, np.float64)
        rgb_source = np.asarray(self.rgb)
        if xyz.shape != (3,):
            raise ValueError("point xyz must have shape (3,)")
        if rgb_source.shape != (3,):
            raise ValueError("point rgb must have shape (3,)")
        if not np.all(np.isfinite(rgb_source)) or np.any(rgb_source < 0) or np.any(rgb_source > 255):
            raise ValueError("point rgb must contain values in [0, 255]")
        if not np.all(rgb_source == np.floor(rgb_source)):
            raise ValueError("point rgb must contain integer values")
        rgb = _readonly(rgb_source, np.uint8)
        object.__setattr__(self, "point_id", int(self.point_id))
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "reprojection_error", float(self.reprojection_error))
        object.__setattr__(
            self, "image_ids", tuple(int(image_id) for image_id in self.image_ids)
        )


@dataclass(frozen=True)
class ColmapModel:
    cameras: Mapping[int, ColmapCameraRecord]
    images: Mapping[int, ColmapImageRecord]
    points3d: Mapping[int, ColmapPointRecord]

    def __post_init__(self) -> None:
        cameras = dict(sorted(self.cameras.items()))
        images = dict(sorted(self.images.items()))
        points = dict(sorted(self.points3d.items()))
        if any(key != value.camera_id for key, value in cameras.items()):
            raise ValueError("camera map keys must match camera_id")
        if any(key != value.image_id for key, value in images.items()):
            raise ValueError("image map keys must match image_id")
        if any(key != value.point_id for key, value in points.items()):
            raise ValueError("point map keys must match point_id")
        names = [image.name for image in images.values()]
        if len(names) != len(set(names)):
            raise ValueError("COLMAP image names must be unique")
        missing_camera_ids = {
            image.camera_id for image in images.values() if image.camera_id not in cameras
        }
        if missing_camera_ids:
            raise ValueError(f"images reference missing cameras: {missing_camera_ids}")
        object.__setattr__(self, "cameras", MappingProxyType(cameras))
        object.__setattr__(self, "images", MappingProxyType(images))
        object.__setattr__(self, "points3d", MappingProxyType(points))


def _camera_record(camera_id: int, camera: object) -> ColmapCameraRecord:
    model = camera.model.name
    params = np.asarray(camera.params, dtype=np.float64)
    width = int(camera.width)
    height = int(camera.height)
    if model == "PINHOLE":
        if params.shape != (4,):
            raise ValueError("COLMAP PINHOLE camera must have four parameters")
        fx, fy, cx, cy = params
        distortion = CameraDistortion("PINHOLE", ())
    elif model == "SIMPLE_RADIAL":
        if params.shape != (4,):
            raise ValueError("COLMAP SIMPLE_RADIAL camera must have four parameters")
        fx, cx, cy, k1 = params
        fy = fx
        distortion = CameraDistortion("SIMPLE_RADIAL", (float(k1),))
    else:
        raise ValueError(f"unsupported COLMAP camera model: {model}")
    return ColmapCameraRecord(
        camera_id=int(camera_id),
        intrinsics=CameraIntrinsics(
            width=width,
            height=height,
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
        ),
        distortion=distortion,
    )


def read_colmap_model(sparse_dir: Path) -> ColmapModel:
    path = Path(sparse_dir)
    required = ("cameras.bin", "images.bin", "points3D.bin")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing COLMAP binary files in {path}: {missing}")

    try:
        installed_version = version("pycolmap")
    except PackageNotFoundError as error:
        raise RuntimeError(
            f"pycolmap=={PYCOLMAP_VERSION} is required to read COLMAP models"
        ) from error
    if installed_version != PYCOLMAP_VERSION:
        raise RuntimeError(
            f"pycolmap=={PYCOLMAP_VERSION} is required, found {installed_version}"
        )

    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(path))
    cameras = {
        int(camera_id): _camera_record(int(camera_id), camera)
        for camera_id, camera in reconstruction.cameras.items()
    }
    images: dict[int, ColmapImageRecord] = {}
    for image_id, image in reconstruction.images.items():
        if not image.has_pose:
            continue
        matrix = np.asarray(image.cam_from_world().matrix(), dtype=np.float64)
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :] = matrix
        images[int(image_id)] = ColmapImageRecord(
            image_id=int(image_id),
            name=str(image.name),
            camera_id=int(image.camera_id),
            world_to_camera=world_to_camera,
            points2d_xy=np.asarray(
                [point.xy for point in image.points2D], dtype=np.float64
            ).reshape(-1, 2),
            point3d_ids=np.asarray(
                [
                    int(point.point3D_id) if point.has_point3D() else -1
                    for point in image.points2D
                ],
                dtype=np.int64,
            ),
        )
    points = {
        int(point_id): ColmapPointRecord(
            point_id=int(point_id),
            xyz=point.xyz,
            rgb=point.color,
            reprojection_error=float(point.error),
            image_ids=tuple(int(element.image_id) for element in point.track.elements),
        )
        for point_id, point in reconstruction.points3D.items()
    }
    return ColmapModel(cameras=cameras, images=images, points3d=points)
