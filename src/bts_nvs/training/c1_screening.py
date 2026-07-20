from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Mapping

import yaml

from bts_nvs.data.dataset import SceneDataset
from bts_nvs.data.holdout import load_holdout_split
from bts_nvs.data.manifest import load_scene_manifest
from bts_nvs.evaluation.high_frequency import evaluate_render_directory
from bts_nvs.training.c1_candidates import candidate_settings
from bts_nvs.training.full_training import BackendDecision
from bts_nvs.training.trainer import compute_config_sha256


BASELINE_CANDIDATE = "B0-reference"
MAX_VRAM_MB = 23 * 1024


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def score50(report: Mapping[str, object]) -> float:
    psnr = _finite_number(report.get("psnr_db_mean"), "psnr_db_mean")
    ssim = _finite_number(report.get("ssim_mean"), "ssim_mean")
    lpips = _finite_number(report.get("lpips_mean"), "lpips_mean")
    return 40.0 - 40.0 * lpips + 30.0 * ssim + 0.6 * psnr


def atomic_json(path: Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, output)


def load_report(path: Path, scene_id: str, candidate_id: str) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"missing qualification report: {source}")
    report = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"qualification report must contain an object: {source}")
    if (
        report.get("schema_version") != 1
        or report.get("step") != 7000
        or report.get("scene_id") != scene_id
        or report.get("candidate_id") != candidate_id
    ):
        raise ValueError(f"qualification report identity mismatch: {source}")
    score50(report)
    return report


def load_completed_run(
    run_dir: Path,
    scene_id: str,
    candidate_id: str,
    decision: BackendDecision,
) -> dict:
    run = Path(run_dir)
    report = load_report(
        run / "qualification_report.json",
        scene_id,
        candidate_id,
    )
    config_path = run / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"completed run config does not exist: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    settings = candidate_settings(candidate_id)
    expected = {
        "scene_id": scene_id,
        "qualification_candidate": candidate_id,
        "resize_factor": 1,
        "max_steps": 7000,
        "seed": 0,
        "cache_images": True,
        "pinned_transfer": True,
        "internal_holdout": True,
        "optimizer_backend": decision.optimizer_backend,
        "precision": decision.precision,
        "rolling_checkpoint": False,
        "grow_grad2d": settings.grow_grad2d,
        "absgrad": settings.absgrad,
        "revised_opacity": settings.revised_opacity,
    }
    if not isinstance(config, dict) or any(
        config.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"completed run config violates C1 screening: {config_path}")
    if report.get("config_sha256") != compute_config_sha256(config):
        raise ValueError("completed run config hash does not match its report")
    if not (run / "validation_renders").is_dir():
        raise FileNotFoundError("completed run validation renders do not exist")
    if any(run.rglob("*.pt")) or any(run.rglob("*.pth")):
        raise ValueError("C1 screening run must not contain model checkpoints")
    return report


def build_screening_command(
    *,
    repo_root: Path,
    scenes_root: Path,
    manifests_root: Path,
    run_dir: Path,
    scene_id: str,
    candidate_id: str,
    decision: BackendDecision,
    python_bin: str,
) -> list[str]:
    candidate_settings(candidate_id)
    return [
        python_bin,
        str(Path(repo_root) / "src" / "bts_nvs" / "training" / "run_training.py"),
        "--scene_dir",
        str(Path(scenes_root) / scene_id),
        "--manifest_dir",
        str(Path(manifests_root) / scene_id),
        "--output_dir",
        str(Path(run_dir)),
        "--resize_factor",
        "1",
        "--max_steps",
        "7000",
        "--seed",
        "0",
        "--cache_images",
        "--pinned_transfer",
        "--qualification_candidate",
        candidate_id,
        "--optimizer_backend",
        decision.optimizer_backend,
        "--precision",
        decision.precision,
    ]


def diagnostics_for_run(
    *,
    scene_id: str,
    candidate_id: str,
    scene_root: Path,
    manifest_root: Path,
    render_dir: Path,
) -> dict:
    manifest = load_scene_manifest(Path(manifest_root) / "manifest.json", scene_root)
    split = load_holdout_split(Path(manifest_root) / "holdout.json", manifest)
    reference = manifest.train_intrinsics[0]
    dataset = SceneDataset(
        manifest,
        scene_root,
        image_names=split.validation_image_names,
        undistort=True,
        resize=(reference.width, reference.height),
        cache_images=False,
    )
    report = evaluate_render_directory(dataset, render_dir)
    if report["scene_id"] != scene_id:
        raise ValueError("diagnostic scene identity mismatch")
    report["candidate_id"] = candidate_id
    return report
