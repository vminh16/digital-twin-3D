from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from bts_nvs.data.manifest import load_scene_manifest
from bts_nvs.data.scene_pool import validate_scene_pool, validate_scene_selection
from bts_nvs.rendering.inference import (
    gaussians_from_checkpoint,
    render_test_camera,
)
from bts_nvs.submission.validator import image_format_from_name, validate_submission
from bts_nvs.training.full_training import (
    load_or_create_backend_decision,
    load_trained_checkpoint,
)


DEFAULT_JPEG_QUALITY = 98


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, output)


def _rgb_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3 or not np.all(np.isfinite(array)):
        raise ValueError("inference RGB must be finite with shape (H, W, 3)")
    return np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def _validate_jpeg_quality(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise ValueError("jpeg_quality must be an integer from 1 to 100")
    return value


def _save_submission_image(image: np.ndarray, path: Path, jpeg_quality: int) -> None:
    image_format = image_format_from_name(path.name)
    options = {}
    if image_format == "JPEG":
        options = {
            "quality": jpeg_quality,
            "subsampling": 0,
            "optimize": True,
            "progressive": False,
        }
    Image.fromarray(_rgb_uint8(image)).save(path, format=image_format, **options)


def _select_scenes(
    scenes_root: Path,
    manifests_root: Path,
    scene_ids: Sequence[str] | None,
    allow_noncanonical_scenes: bool,
) -> tuple[str, ...]:
    if not allow_noncanonical_scenes:
        canonical = validate_scene_pool(scenes_root, manifests_root)
        selected = validate_scene_selection(scene_ids)
        if any(scene_id not in canonical for scene_id in selected):
            raise ValueError("selected scene is missing from the validated scene pool")
        return selected

    if scene_ids is None:
        raise ValueError("noncanonical inference requires explicit scene_ids")
    selected = tuple(scene_ids)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("explicit scene_ids must be non-empty and unique")
    for scene_id in selected:
        if not scene_id or Path(scene_id).name != scene_id or scene_id in {".", ".."}:
            raise ValueError(f"unsafe explicit scene_id: {scene_id!r}")
        if not (Path(scenes_root) / scene_id).is_dir():
            raise FileNotFoundError(f"scene directory does not exist: {scene_id}")
        if not (Path(manifests_root) / scene_id / "manifest.json").is_file():
            raise FileNotFoundError(f"scene manifest does not exist: {scene_id}")
    return selected


def run_inference(
    *,
    scenes_root: Path,
    manifests_root: Path,
    backend_root: Path,
    full_root: Path,
    output_root: Path,
    report_path: Path,
    scene_ids: Sequence[str] | None = None,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    allow_noncanonical_scenes: bool = False,
    device: torch.device | None = None,
) -> dict:
    output = Path(output_root)
    report_file = Path(report_path)
    if output.exists():
        raise FileExistsError(f"output root already exists: {output}")
    if report_file.exists():
        raise FileExistsError(f"inference report already exists: {report_file}")
    jpeg_quality = _validate_jpeg_quality(jpeg_quality)
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("test inference requires CUDA")
        device = torch.device("cuda")

    selected = _select_scenes(
        scenes_root,
        manifests_root,
        scene_ids,
        allow_noncanonical_scenes,
    )
    decision = load_or_create_backend_decision(backend_root)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    manifests = {}
    scene_reports = []
    total_start = time.perf_counter()
    try:
        for scene_id in selected:
            scene_start = time.perf_counter()
            manifest_path = Path(manifests_root) / scene_id / "manifest.json"
            manifest = load_scene_manifest(
                manifest_path, Path(scenes_root) / scene_id
            )
            if manifest.scene_id != scene_id:
                raise ValueError(f"manifest scene identity mismatch: {scene_id}")
            manifests[scene_id] = manifest
            run_dir = Path(full_root) / "scenes" / scene_id
            trained, checkpoint = load_trained_checkpoint(
                run_dir, scene_id, manifest_path, decision
            )
            checkpoint_path = run_dir / "checkpoints" / "recovery.pt"
            active_sh_degree = checkpoint.get("active_sh_degree")
            if (
                isinstance(active_sh_degree, bool)
                or not isinstance(active_sh_degree, int)
                or not 0 <= active_sh_degree <= 3
            ):
                raise ValueError("checkpoint active_sh_degree must be from 0 to 3")
            gaussians = gaussians_from_checkpoint(checkpoint, device)
            del checkpoint
            scene_output = staging / scene_id
            scene_output.mkdir()
            for name, pose, intrinsics, distortion in zip(
                manifest.test_image_names,
                manifest.test_world_to_camera,
                manifest.test_intrinsics,
                manifest.test_distortion,
                strict=True,
            ):
                rendered = render_test_camera(
                    gaussians,
                    pose,
                    intrinsics,
                    distortion,
                    manifest.normalization_transform,
                    active_sh_degree,
                )
                _save_submission_image(rendered, scene_output / name, jpeg_quality)
            del gaussians
            if device.type == "cuda":
                torch.cuda.empty_cache()
            scene_reports.append(
                {
                    "scene_id": scene_id,
                    "image_count": len(manifest.test_image_names),
                    "completed_step": trained.completed_step,
                    "config_sha256": trained.config_sha256,
                    "manifest_sha256": trained.manifest_sha256,
                    "checkpoint_sha256": _sha256(checkpoint_path),
                    "elapsed_seconds": time.perf_counter() - scene_start,
                }
            )

        issues = validate_submission(staging, manifests)
        if issues:
            details = "; ".join(
                f"{item.scene_id}/{item.filename}:{item.code}" for item in issues
            )
            raise ValueError(f"rendered output contract validation failed: {details}")
        report = {
            "schema_version": 1,
            "scene_ids": list(selected),
            "jpeg_quality": jpeg_quality,
            "jpeg_subsampling": "4:4:4",
            "total_images": sum(item["image_count"] for item in scene_reports),
            "elapsed_seconds": time.perf_counter() - total_start,
            "scenes": scene_reports,
        }
        staging.replace(output)
        _write_json(report_file, report)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render trained Gaussian scenes at canonical test poses."
    )
    parser.add_argument("--scenes_root", type=Path, required=True)
    parser.add_argument("--manifests_root", type=Path, required=True)
    parser.add_argument("--backend_root", type=Path, required=True)
    parser.add_argument("--full_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--report_path", type=Path, required=True)
    parser.add_argument("--scene_ids", nargs="+")
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help="JPEG quality for .jpg/.jpeg CSV outputs; PNG outputs are lossless.",
    )
    parser.add_argument(
        "--allow_noncanonical_scenes",
        action="store_true",
        help="Allow only explicitly named scenes outside the canonical BTS pool.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_inference(
        scenes_root=args.scenes_root,
        manifests_root=args.manifests_root,
        backend_root=args.backend_root,
        full_root=args.full_root,
        output_root=args.output_root,
        report_path=args.report_path,
        scene_ids=args.scene_ids,
        jpeg_quality=args.jpeg_quality,
        allow_noncanonical_scenes=args.allow_noncanonical_scenes,
    )


if __name__ == "__main__":
    main()
