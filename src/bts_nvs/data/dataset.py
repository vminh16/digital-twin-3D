from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from bts_nvs.cameras.distortion import CameraDistortion
from bts_nvs.cameras.intrinsics import CameraIntrinsics

from .manifest import SceneManifest


_CACHE_RAM_HEADROOM_BYTES = 4 * 1024**3


def available_host_ram_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("cannot query available host RAM")
    return int(status.available_physical)


def estimate_image_cache_bytes(
    manifest: SceneManifest,
    *,
    resize: tuple[int, int] | None = None,
    indices: tuple[int, ...] | None = None,
    include_loss_weight: bool = False,
) -> int:
    selected = range(len(manifest.train_intrinsics)) if indices is None else indices
    bytes_per_pixel = 5 if include_loss_weight else 4
    if resize is not None:
        width, height = resize
        return len(selected) * width * height * bytes_per_pixel
    return sum(
        manifest.train_intrinsics[index].width
        * manifest.train_intrinsics[index].height
        * bytes_per_pixel
        for index in selected
    )


@dataclass(frozen=True)
class CameraSample:
    image: np.ndarray
    world_to_camera: np.ndarray
    intrinsics: CameraIntrinsics
    distortion: CameraDistortion
    valid_mask: np.ndarray
    image_name: str
    loss_weight: np.ndarray | None = None

    def __post_init__(self) -> None:
        image = np.asarray(self.image).copy()
        pose = np.asarray(self.world_to_camera, dtype=np.float64).copy()
        mask = np.asarray(self.valid_mask, dtype=bool).copy()
        weight = (
            None
            if self.loss_weight is None
            else np.asarray(self.loss_weight).copy()
        )
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("sample image must be RGB uint8")
        if pose.shape != (4, 4):
            raise ValueError("sample pose must have shape (4, 4)")
        if mask.shape != image.shape[:2]:
            raise ValueError("valid mask must match image dimensions")
        if weight is not None:
            if weight.shape != image.shape[:2]:
                raise ValueError("loss weight must match image dimensions")
            if weight.dtype != np.uint8:
                raise ValueError("loss weight must be uint8")
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "world_to_camera", pose)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "loss_weight", weight)


class SceneDataset:
    def __init__(
        self,
        manifest: SceneManifest,
        scene_root: Path,
        *,
        image_names: tuple[str, ...] | None = None,
        undistort: bool = False,
        resize: tuple[int, int] | None = None,
        cache_images: bool = False,
        loss_weight_mode: str = "uniform",
        loss_weight_floor: float = 0.5,
        loss_weight_patch_size: int = 31,
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
        if loss_weight_mode not in {"uniform", "local-laplacian"}:
            raise ValueError("loss_weight_mode is unsupported")
        if (
            isinstance(loss_weight_floor, bool)
            or not isinstance(loss_weight_floor, (int, float))
            or not 0.0 < float(loss_weight_floor) <= 1.0
        ):
            raise ValueError("loss_weight_floor must be in (0, 1]")
        if (
            isinstance(loss_weight_patch_size, bool)
            or not isinstance(loss_weight_patch_size, int)
            or loss_weight_patch_size < 3
            or loss_weight_patch_size % 2 == 0
        ):
            raise ValueError(
                "loss_weight_patch_size must be an odd integer of at least 3"
            )
        self.loss_weight_mode = loss_weight_mode
        self.loss_weight_floor = float(loss_weight_floor)
        self.loss_weight_patch_size = loss_weight_patch_size
        if image_names is None:
            self._indices = tuple(range(len(manifest.train_image_names)))
        else:
            names = tuple(image_names)
            if len(names) != len(set(names)):
                raise ValueError("image_names must not contain duplicates")
            index_by_name = {
                name: index for index, name in enumerate(manifest.train_image_names)
            }
            unknown = [name for name in names if name not in index_by_name]
            if unknown:
                raise ValueError(f"image_names contains unknown train images: {unknown}")
            self._indices = tuple(index_by_name[name] for name in names)
        self.cache_images = bool(cache_images)
        self._cache: tuple[CameraSample, ...] | None = None
        if self.cache_images:
            cache_bytes = estimate_image_cache_bytes(
                self.manifest,
                resize=self.resize,
                indices=self._indices,
                include_loss_weight=self.loss_weight_mode != "uniform",
            )
            if available_host_ram_bytes() < cache_bytes + _CACHE_RAM_HEADROOM_BYTES:
                raise MemoryError(
                    "image cache requires its estimated bytes plus 4 GiB free host RAM"
                )
            cached = tuple(self._build_sample(index) for index in self._indices)
            for sample in cached:
                sample.image.setflags(write=False)
                sample.world_to_camera.setflags(write=False)
                sample.valid_mask.setflags(write=False)
                if sample.loss_weight is not None:
                    sample.loss_weight.setflags(write=False)
            self._cache = cached

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> CameraSample:
        if self._cache is not None:
            sample = self._cache[index]
            return CameraSample(
                image=sample.image,
                world_to_camera=sample.world_to_camera,
                intrinsics=sample.intrinsics,
                distortion=sample.distortion,
                valid_mask=sample.valid_mask,
                image_name=sample.image_name,
                loss_weight=sample.loss_weight,
            )
        return self._build_sample(self._indices[index])

    def camera_geometry(
        self,
        index: int,
    ) -> tuple[np.ndarray, CameraIntrinsics]:
        manifest_index = self._indices[index]
        intrinsics = self.manifest.train_intrinsics[manifest_index]
        if self.resize is not None:
            width, height = self.resize
            intrinsics = intrinsics.resized(width=width, height=height)
        return (
            np.asarray(
                self.manifest.train_world_to_camera[manifest_index],
                dtype=np.float64,
            ).copy(),
            intrinsics,
        )

    def _build_sample(self, index: int) -> CameraSample:
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

        loss_weight = (
            _local_laplacian_weights(
                image,
                valid_mask,
                floor=self.loss_weight_floor,
                patch_size=self.loss_weight_patch_size,
            )
            if self.loss_weight_mode == "local-laplacian"
            else None
        )

        return CameraSample(
            image=image,
            world_to_camera=self.manifest.train_world_to_camera[index],
            intrinsics=intrinsics,
            distortion=distortion,
            valid_mask=valid_mask,
            image_name=self.manifest.train_image_names[index],
            loss_weight=loss_weight,
        )


def _local_laplacian_weights(
    image: np.ndarray,
    valid_mask: np.ndarray,
    *,
    floor: float,
    patch_size: int,
) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    local_mean = cv2.boxFilter(
        laplacian,
        cv2.CV_32F,
        (patch_size, patch_size),
        normalize=True,
    )
    local_square_mean = cv2.boxFilter(
        np.square(laplacian),
        cv2.CV_32F,
        (patch_size, patch_size),
        normalize=True,
    )
    local_variance = np.maximum(
        local_square_mean - np.square(local_mean),
        0.0,
    )
    score = np.log1p(local_variance)
    valid_scores = score[valid_mask]
    lower, upper = np.percentile(valid_scores, (20.0, 80.0))
    if upper <= lower:
        normalized = np.zeros_like(score)
    else:
        normalized = np.clip((score - lower) / (upper - lower), 0.0, 1.0)
    weights = floor + (1.0 - floor) * normalized
    weights[~valid_mask] = 0.0
    return np.rint(weights * 255.0).astype(np.uint8)


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
