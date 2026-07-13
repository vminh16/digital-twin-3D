from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from .metrics import LpipsCallable, MetricConfig, evaluate_image


class EvaluationError(ValueError):
    pass


def load_image_pairs(
    expected_images: Mapping[str, tuple[int, int]],
    prediction_dir: Path,
    reference_dir: Path,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    predictions = Path(prediction_dir)
    references = Path(reference_dir)
    expected_names = set(expected_images)
    for directory, label in ((predictions, "prediction"), (references, "reference")):
        if not directory.is_dir():
            raise EvaluationError(f"missing {label} directory: {directory}")
        actual_names = {path.name for path in directory.iterdir() if path.is_file()}
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        if missing or extra:
            raise EvaluationError(f"{label} filenames mismatch; missing={missing}, extra={extra}")

    pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in sorted(expected_names):
        expected_size = expected_images[name]
        loaded: list[np.ndarray] = []
        for directory in (predictions, references):
            try:
                with Image.open(directory / name) as image:
                    if image.mode != "RGB":
                        raise EvaluationError(f"image must be RGB: {name}")
                    if image.size != expected_size:
                        raise EvaluationError(
                            f"image resolution mismatch for {name}: {image.size} != {expected_size}"
                        )
                    loaded.append(np.asarray(image, dtype=np.float64).copy() / 255.0)
            except OSError as error:
                raise EvaluationError(f"cannot decode image: {directory / name}") from error
        pairs[name] = (loaded[0], loaded[1])
    return pairs


def evaluate_benchmark(
    scenes: Mapping[str, Mapping[str, tuple[np.ndarray, np.ndarray]]],
    config: MetricConfig,
    lpips_backend: LpipsCallable,
) -> dict[str, object]:
    if not scenes:
        raise EvaluationError("benchmark contains zero scenes")
    scene_reports: dict[str, object] = {}
    scene_scores: list[float] = []
    for scene_id in sorted(scenes):
        pairs = scenes[scene_id]
        if not pairs:
            raise EvaluationError(f"scene {scene_id} contains zero images")
        image_reports = {
            name: evaluate_image(*pairs[name], config, lpips_backend)
            for name in sorted(pairs)
        }
        scene_score = float(np.mean([item["composite"] for item in image_reports.values()]))
        scene_reports[scene_id] = {"images": image_reports, "score": scene_score}
        scene_scores.append(scene_score)
    metadata = {
        "psnr_max": config.psnr_max,
        "crop_border": config.crop_border,
        "data_range": config.data_range,
        "ssim_kernel_size": config.ssim_kernel_size,
        "ssim_sigma": config.ssim_sigma,
        "ssim_k1": config.ssim_k1,
        "ssim_k2": config.ssim_k2,
        "ssim_padding": config.ssim_padding,
        "lpips": {
            "package": lpips_backend.package,
            "package_version": lpips_backend.version,
            "backbone": config.lpips_backbone,
            "weight_version": config.lpips_weight_version,
            "input_normalization": "RGB [0,1] to [-1,1]",
            "device": lpips_backend.device,
            "dtype": lpips_backend.dtype,
        },
    }
    return {
        "metadata": metadata,
        "scenes": scene_reports,
        "final_score": float(np.mean(scene_scores)),
    }


def save_metric_report(report: Mapping[str, object], path: Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

