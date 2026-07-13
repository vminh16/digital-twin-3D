from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from bts_nvs.cameras.distortion import CameraDistortion
from bts_nvs.cameras.intrinsics import CameraIntrinsics

from .manifest import SceneManifest


@dataclass(frozen=True)
class CameraSample:
    image: np.ndarray
    world_to_camera: np.ndarray
    intrinsics: CameraIntrinsics
    distortion: CameraDistortion
    valid_mask: np.ndarray
    image_name: str

    def __post_init__(self) -> None:
        image = np.asarray(self.image).copy()
        pose = np.asarray(self.world_to_camera, dtype=np.float64).copy()
        mask = np.asarray(self.valid_mask, dtype=bool).copy()
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("sample image must be RGB uint8")
        if pose.shape != (4, 4):
            raise ValueError("sample pose must have shape (4, 4)")
        if mask.shape != image.shape[:2]:
            raise ValueError("valid mask must match image dimensions")
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "world_to_camera", pose)
        object.__setattr__(self, "valid_mask", mask)


class SceneDataset:
    def __init__(
        self,
        manifest: SceneManifest,
        scene_root: Path,
        *,
        undistort: bool = False,
        resize: tuple[int, int] | None = None,
    ) -> None:
        if resize is not None and (
            len(resize) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in resize)
        ):
            raise ValueError("resize must be a positive (width, height) pair")
        self.manifest = manifest
        self.scene_root = Path(scene_root)
        self.undistort = bool(undistort)
        self.resize = resize

    def __len__(self) -> int:
        return len(self.manifest.train_image_names)

    def __getitem__(self, index: int) -> CameraSample:
        path = self.scene_root / self.manifest.train_image_paths[index]
        with Image.open(path) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
        intrinsics = self.manifest.train_intrinsics[index]
        distortion = self.manifest.train_distortion[index]
        if image.shape[:2] != (intrinsics.height, intrinsics.width):
            raise ValueError(
                f"image resolution does not match intrinsics: {path}"
            )
        valid_mask = np.ones(image.shape[:2], dtype=bool)

        if self.undistort and distortion.model == "SIMPLE_RADIAL":
            image, valid_mask = _undistort(image, intrinsics, distortion)
            distortion = CameraDistortion("PINHOLE", ())
        elif self.undistort:
            image = image.copy()

        if self.resize is not None:
            width, height = self.resize
            interpolation = cv2.INTER_AREA if width < image.shape[1] or height < image.shape[0] else cv2.INTER_LINEAR
            image = cv2.resize(image, (width, height), interpolation=interpolation)
            valid_mask = cv2.resize(
                valid_mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            intrinsics = intrinsics.resized(width=width, height=height)

        return CameraSample(
            image=image,
            world_to_camera=self.manifest.train_world_to_camera[index],
            intrinsics=intrinsics,
            distortion=distortion,
            valid_mask=valid_mask,
            image_name=self.manifest.train_image_names[index],
        )


def _undistort(
    image: np.ndarray,
    intrinsics: CameraIntrinsics,
    distortion: CameraDistortion,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = intrinsics.matrix
    coefficients = np.asarray([distortion.coefficients[0], 0, 0, 0, 0], dtype=np.float64)
    size = (intrinsics.width, intrinsics.height)
    map_x, map_y = cv2.initUndistortRectifyMap(
        matrix, coefficients, None, matrix, size, cv2.CV_32FC1
    )
    result = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    valid = (
        (map_x >= 0.0)
        & (map_x <= intrinsics.width - 1)
        & (map_y >= 0.0)
        & (map_y <= intrinsics.height - 1)
    )
    return result, valid
