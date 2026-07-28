from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from bts_nvs.data.dataset import SceneDataset


SENSITIVITY_SCHEMA_VERSION = 1
GAMMA = 1.5
EDGE_THRESHOLD = 0.05
SMOOTH_THRESHOLD = 0.3
POOL_SIZE = 5


@dataclass(frozen=True)
class SensitivityMapSet:
    maps: dict[str, np.ndarray]
    manifest_sha256: str
    scene_mean: float


def extract_perceptual_sensitivity(
    image: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("image must be RGB uint8")
    if min(rgb.shape[:2]) < POOL_SIZE:
        raise ValueError("image is smaller than the sensitivity pooling kernel")
    if valid_mask is not None:
        mask = np.asarray(valid_mask)
        if mask.dtype != bool or mask.shape != rgb.shape[:2]:
            raise ValueError("valid_mask must be boolean and match image dimensions")
    else:
        mask = np.ones(rgb.shape[:2], dtype=bool)

    # Match the reference implementation's PIL grayscale conversion and
    # gamma lookup, including both 8-bit quantization points.
    gray_image = Image.fromarray(rgb).convert("L")
    gamma_image = gray_image.point(
        lambda value: 255.0 * (value / 255.0) ** (1.0 / GAMMA)
    )
    gray = torch.from_numpy(
        np.asarray(gamma_image, dtype=np.float32) / 255.0
    )[None, None]
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    )[None, None]
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    )[None, None]
    edge_x = F.conv2d(gray, sobel_x, padding=1)
    edge_y = F.conv2d(gray, sobel_y, padding=1)
    binary = (torch.sqrt(edge_x.square() + edge_y.square()) > EDGE_THRESHOLD).float()
    pooled = F.avg_pool2d(binary, kernel_size=POOL_SIZE)
    smoothed = F.interpolate(
        pooled,
        size=rgb.shape[:2],
        mode="nearest",
    )
    result = (smoothed[0, 0] > SMOOTH_THRESHOLD).numpy()
    result &= mask
    return result.astype(np.uint8) * 255


def prepare_sensitivity_artifact(
    dataset: SceneDataset,
    output_dir: str | Path,
) -> SensitivityMapSet:
    if not isinstance(dataset, SceneDataset):
        raise TypeError("dataset must be a SceneDataset")
    if len(dataset) == 0:
        raise ValueError("sensitivity extraction requires train images")
    root = Path(output_dir)
    maps_dir = root / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    maps: dict[str, np.ndarray] = {}
    records: list[dict[str, object]] = []
    output_names: set[str] = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        output_name = Path(sample.image_name).with_suffix(".png").name
        if output_name.casefold() in output_names:
            raise ValueError("sensitivity map output names collide")
        output_names.add(output_name.casefold())
        sensitivity = extract_perceptual_sensitivity(
            sample.image,
            sample.valid_mask,
        )
        map_path = maps_dir / output_name
        Image.fromarray(sensitivity).save(map_path)
        maps[sample.image_name] = sensitivity
        records.append(
            {
                "image_name": sample.image_name,
                "map_name": output_name,
                "processed_image_sha256": hashlib.sha256(
                    sample.image.tobytes()
                ).hexdigest(),
                "map_sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
                "mean_sensitivity": float(sensitivity.mean() / 255.0),
                "width": int(sensitivity.shape[1]),
                "height": int(sensitivity.shape[0]),
            }
        )

    manifest = {
        "schema_version": SENSITIVITY_SCHEMA_VERSION,
        "algorithm": "perceptual-gs-sobel-binary-v1",
        "gamma": GAMMA,
        "edge_threshold": EDGE_THRESHOLD,
        "smooth_threshold": SMOOTH_THRESHOLD,
        "pool_size": POOL_SIZE,
        "image_count": len(records),
        "scene_mean_sensitivity": float(
            sum(record["mean_sensitivity"] for record in records) / len(records)
        ),
        "images": records,
    }
    payload = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    manifest_sha256 = hashlib.sha256(payload).hexdigest()
    manifest["manifest_sha256"] = manifest_sha256
    path = root / "sensitivity_manifest.json"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return SensitivityMapSet(
        maps=maps,
        manifest_sha256=manifest_sha256,
        scene_mean=float(manifest["scene_mean_sensitivity"]),
    )


def validate_sensitivity_artifact(
    path: str | Path,
    expected_sha256: str,
    expected_image_names: tuple[str, ...],
) -> None:
    root = Path(path)
    manifest_path = root / "sensitivity_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"required sensitivity manifest does not exist: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_sha256 = manifest.pop("manifest_sha256", None)
    payload = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if recorded_sha256 != expected_sha256 or actual_sha256 != expected_sha256:
        raise ValueError("sensitivity manifest hash mismatch")
    records = manifest.get("images")
    if not isinstance(records, list):
        raise ValueError("sensitivity manifest images must be a list")
    names = tuple(record.get("image_name") for record in records)
    if names != expected_image_names:
        raise ValueError("sensitivity manifest does not match internal train split")
    for record in records:
        map_path = root / "maps" / str(record["map_name"])
        if (
            not map_path.is_file()
            or hashlib.sha256(map_path.read_bytes()).hexdigest()
            != record["map_sha256"]
        ):
            raise ValueError(f"sensitivity map hash mismatch: {map_path}")
