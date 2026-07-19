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
from bts_nvs.training.c1_candidates import QUALIFICATION_CANDIDATES


CALIBRATION_SCENES = CALIBRATION_SCENE_IDS
CANDIDATES = ("B0-reference", "B0-compact")


@torch.no_grad()
def evaluate_internal_validation(
    trainer, dataset, lpips_backend, render_dir: Path | None
) -> dict:
    output = Path(render_dir) if render_dir is not None else None
    if output is not None:
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
        if output is not None:
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


def build_full_length_report(
    *,
    scene_id: str,
    git_commit: str,
    initial_validation: dict,
    final_train: dict,
    final_validation: dict,
    summary: dict,
    metric_records: list[dict],
    timing_records: dict[str, dict],
    convergence: dict,
) -> dict:
    if scene_id != "HCM0181":
        raise ValueError("full-length qualification requires HCM0181")
    if len(git_commit) != 40:
        raise ValueError("git_commit must be a full SHA-1")
    if len(metric_records) != 30_000 or [
        item.get("step") for item in metric_records
    ] != list(range(1, 30_001)):
        raise ValueError("full-length qualification requires 30000 ordered records")
    if summary.get("total_steps") != 30_000:
        raise ValueError("full-length qualification summary must end at step 30000")
    if list(timing_records) != [str(step) for step in range(1, 30_001)]:
        raise ValueError("full-length qualification requires 30000 ordered timings")
    if any(
        not math.isfinite(float(item.get("total", math.nan)))
        for item in timing_records.values()
    ):
        raise ValueError("full-length qualification contains non-finite timing")
    for item in metric_records:
        if not math.isfinite(float(item.get("loss", math.nan))):
            raise ValueError("full-length qualification contains non-finite loss")

    psnr_delta = float(final_validation["psnr_db_mean"]) - float(
        initial_validation["psnr_db_mean"]
    )
    ssim_delta = float(final_validation["ssim_mean"]) - float(
        initial_validation["ssim_mean"]
    )
    lpips_improvement = float(initial_validation["lpips_mean"]) - float(
        final_validation["lpips_mean"]
    )
    train_validation_gap = float(final_train["psnr_db_mean"]) - float(
        final_validation["psnr_db_mean"]
    )
    peak_gaussians = max(int(item["num_gaussians"]) for item in metric_records)
    max_vram_mb = float(summary["max_vram_mb"])
    values = (
        psnr_delta,
        ssim_delta,
        lpips_improvement,
        train_validation_gap,
        max_vram_mb,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("full-length qualification report contains non-finite value")

    gates = {
        "psnr_improved": psnr_delta > 3.0,
        "ssim_improved": ssim_delta > 0.05,
        "lpips_improved": lpips_improvement > 0.05,
        "train_validation_gap_bounded": train_validation_gap < 8.0,
        "peak_gaussians_bounded": peak_gaussians < 10_000_000,
        "peak_vram_bounded": max_vram_mb < 20 * 1024,
        "final_render_non_blank": bool(convergence.get("final_render_non_blank")),
    }
    return {
        "schema_version": 1,
        "scene_id": scene_id,
        "step": 30_000,
        "git_commit": git_commit,
        "automated_gates": gates,
        "automated_gates_passed": all(gates.values()),
        "validation_psnr_delta_db": psnr_delta,
        "validation_ssim_delta": ssim_delta,
        "validation_lpips_improvement": lpips_improvement,
        "train_validation_psnr_gap_db": train_validation_gap,
        "peak_gaussians": peak_gaussians,
        "max_vram_mb": max_vram_mb,
        "timing_record_count": len(timing_records),
        "manual_visual_review_required": True,
        "initial_validation": initial_validation,
        "final_train": final_train,
        "final_validation": final_validation,
    }


def save_full_length_report(report: dict, path: Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, output)


def _validate_report(report: dict) -> None:
    if report.get("schema_version") != 1 or report.get("step") != 7000:
        raise ValueError("qualification report must use schema 1 at step 7000")
    if report.get("scene_id") not in CALIBRATION_SCENES:
        raise ValueError("qualification report contains an unexpected scene")
    if report.get("candidate_id") not in QUALIFICATION_CANDIDATES:
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
