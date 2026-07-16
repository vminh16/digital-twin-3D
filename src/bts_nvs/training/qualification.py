from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from bts_nvs.data.inventory import CALIBRATION_SCENE_IDS
from bts_nvs.evaluation.metrics import MetricConfig, evaluate_image
from bts_nvs.rendering.gsplat_renderer import render_gaussians
from bts_nvs.training.trainer import _normalize_world_to_camera


CALIBRATION_SCENES = CALIBRATION_SCENE_IDS
CANDIDATES = ("B0-reference", "B0-compact")


@torch.no_grad()
def evaluate_internal_validation(trainer, dataset, lpips_backend, render_dir: Path) -> dict:
    output = Path(render_dir)
    output.mkdir(parents=True, exist_ok=True)
    metric_config = MetricConfig(psnr_max=100.0)
    image_reports: dict[str, dict] = {}
    valid_fractions = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if sample.distortion.model != "PINHOLE":
            raise ValueError("qualification requires undistorted PINHOLE samples")
        normalized_w2c = _normalize_world_to_camera(
            sample.world_to_camera,
            dataset.manifest.normalization_transform,
        )
        result = render_gaussians(
            gaussians=trainer.gaussians,
            viewmat=torch.from_numpy(normalized_w2c).to(trainer.device),
            intrinsics=sample.intrinsics,
            active_sh_degree=trainer.active_sh_degree,
            render_mode="RGB",
        )
        prediction = result.rgb.float().clamp(0.0, 1.0).cpu().numpy()
        target = sample.image.astype(np.float32) / 255.0
        mask = np.asarray(sample.valid_mask, dtype=bool)
        prediction[~mask] = target[~mask]
        evaluated = evaluate_image(prediction, target, metric_config, lpips_backend)
        if evaluated["psnr_db"] is None:
            raise ValueError("qualification PSNR must be finite")
        metrics = {
            "psnr_db": evaluated["psnr_db"],
            "ssim": evaluated["ssim"],
            "lpips": evaluated["lpips"],
        }
        image_reports[sample.image_name] = metrics
        valid_fractions.append(float(mask.mean()))
        Image.fromarray((prediction * 255.0).round().astype(np.uint8)).save(
            output / Path(sample.image_name).with_suffix(".png").name
        )
    if not image_reports:
        raise ValueError("qualification validation set is empty")
    values = tuple(image_reports.values())
    return {
        "image_count": len(values),
        "psnr_db_mean": float(np.mean([item["psnr_db"] for item in values])),
        "ssim_mean": float(np.mean([item["ssim"] for item in values])),
        "lpips_mean": float(np.mean([item["lpips"] for item in values])),
        "valid_fraction_mean": float(np.mean(valid_fractions)),
        "images": image_reports,
    }


def _validate_report(report: dict) -> None:
    if report.get("schema_version") != 1 or report.get("step") != 7000:
        raise ValueError("qualification report must use schema 1 at step 7000")
    if report.get("scene_id") not in CALIBRATION_SCENES:
        raise ValueError("qualification report contains an unexpected scene")
    if report.get("candidate_id") not in CANDIDATES:
        raise ValueError("qualification report contains an unexpected candidate")
    if not isinstance(report.get("image_count"), int) or report["image_count"] <= 0:
        raise ValueError("qualification report image_count must be positive")
    for key in (
        "psnr_db_mean",
        "ssim_mean",
        "lpips_mean",
        "max_vram_mb",
        "total_time_seconds",
    ):
        value = report.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"qualification report has invalid {key}")
    if not isinstance(report.get("peak_gaussians"), int) or report["peak_gaussians"] <= 0:
        raise ValueError("qualification report peak_gaussians must be positive")
    if not -1.0 <= report["ssim_mean"] <= 1.0:
        raise ValueError("qualification SSIM is out of range")
    if not 0.0 <= report["lpips_mean"] <= 1.0:
        raise ValueError("qualification LPIPS is out of range")


def build_qualification_decision(reports: list[dict] | tuple[dict, ...]) -> dict:
    expected = len(CALIBRATION_SCENES) * len(CANDIDATES)
    if len(reports) != expected:
        raise ValueError(f"qualification requires exactly {expected} reports")
    indexed: dict[tuple[str, str], dict] = {}
    for report in reports:
        _validate_report(report)
        key = (report["scene_id"], report["candidate_id"])
        if key in indexed:
            raise ValueError("duplicate qualification report")
        indexed[key] = report
    required = {(scene, candidate) for scene in CALIBRATION_SCENES for candidate in CANDIDATES}
    if set(indexed) != required:
        raise ValueError("qualification reports do not cover the locked matrix")

    aggregates = {}
    for candidate in CANDIDATES:
        selected = [indexed[(scene, candidate)] for scene in CALIBRATION_SCENES]
        aggregates[candidate] = {
            "psnr_db_mean": float(np.mean([item["psnr_db_mean"] for item in selected])),
            "ssim_mean": float(np.mean([item["ssim_mean"] for item in selected])),
            "lpips_mean": float(np.mean([item["lpips_mean"] for item in selected])),
            "peak_gaussians_mean": float(np.mean([item["peak_gaussians"] for item in selected])),
            "max_vram_mb": float(max(item["max_vram_mb"] for item in selected)),
            "total_time_seconds": float(sum(item["total_time_seconds"] for item in selected)),
        }
    reference = aggregates["B0-reference"]
    compact = aggregates["B0-compact"]
    quality = (
        compact["psnr_db_mean"] >= reference["psnr_db_mean"] - 0.25
        and compact["ssim_mean"] >= reference["ssim_mean"] - 0.005
        and compact["lpips_mean"] <= reference["lpips_mean"] + 0.01
    )
    resource = (
        compact["peak_gaussians_mean"] <= 0.85 * reference["peak_gaussians_mean"]
        or compact["total_time_seconds"] <= 0.85 * reference["total_time_seconds"]
    )
    passed = bool(quality and resource)
    return {
        "schema_version": 1,
        "calibration_scenes": list(CALIBRATION_SCENES),
        "candidate_aggregates": aggregates,
        "compact_passed": passed,
        "selected_candidate": "B0-compact" if passed else "B0-reference",
    }


def save_qualification_decision(decision: dict, path: Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, output)


def save_qualification_report(report: dict, path: Path) -> None:
    _validate_report(report)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, output)


def load_qualification_reports(root: Path) -> list[dict]:
    reports = []
    for scene in CALIBRATION_SCENES:
        for candidate in CANDIDATES:
            path = Path(root) / scene / candidate / "qualification_report.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing qualification report: {path}")
            report = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                raise ValueError(f"qualification report must contain an object: {path}")
            reports.append(report)
    return reports
