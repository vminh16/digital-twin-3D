from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from bts_nvs.data.manifest import load_scene_manifest
from bts_nvs.data.scene_pool import validate_scene_pool, validate_scene_selection
from bts_nvs.evaluation.evaluator import (
    evaluate_benchmark,
    load_image_pairs,
    save_metric_report,
)
from bts_nvs.evaluation.metrics import LpipsBackend, LpipsCallable, MetricConfig


def _exact_scene_directories(root: Path, selected: tuple[str, ...], label: str) -> None:
    directory = Path(root)
    if not directory.is_dir():
        raise FileNotFoundError(f"{label} root does not exist: {directory}")
    actual = {entry.name for entry in directory.iterdir()}
    expected = set(selected)
    if actual != expected or any(not (directory / name).is_dir() for name in actual):
        raise ValueError(
            f"{label} scene directories mismatch; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def run_local_benchmark(
    *,
    outputs_root: Path,
    reference_root: Path,
    manifests_root: Path,
    scenes_root: Path,
    scene_ids: Sequence[str] | None,
    config: MetricConfig,
    lpips_backend: LpipsCallable,
    report_path: Path,
) -> dict[str, object]:
    canonical = validate_scene_pool(scenes_root, manifests_root)
    selected = validate_scene_selection(scene_ids)
    if any(scene_id not in canonical for scene_id in selected):
        raise ValueError("selected scene is missing from the validated scene pool")
    _exact_scene_directories(outputs_root, selected, "output")
    _exact_scene_directories(reference_root, selected, "reference")

    scenes = {}
    for scene_id in selected:
        manifest = load_scene_manifest(
            Path(manifests_root) / scene_id / "manifest.json",
            Path(scenes_root) / scene_id,
        )
        expected = {
            name: (intrinsics.width, intrinsics.height)
            for name, intrinsics in zip(
                manifest.test_image_names, manifest.test_intrinsics, strict=True
            )
        }
        scenes[scene_id] = load_image_pairs(
            expected,
            Path(outputs_root) / scene_id,
            Path(reference_root) / scene_id,
        )
    report = evaluate_benchmark(scenes, config, lpips_backend)
    save_metric_report(report, report_path)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate existing renders against an explicit local reference root."
    )
    parser.add_argument("--outputs_root", type=Path, required=True)
    parser.add_argument("--reference_root", type=Path, required=True)
    parser.add_argument("--manifests_root", type=Path, required=True)
    parser.add_argument("--scenes_root", type=Path, required=True)
    parser.add_argument("--scene_ids", nargs="+")
    parser.add_argument("--psnr_max", type=float, required=True)
    parser.add_argument("--lpips_backbone", choices=("alex", "vgg"), default="alex")
    parser.add_argument("--crop_border", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report_path", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = MetricConfig(
        psnr_max=args.psnr_max,
        lpips_backbone=args.lpips_backbone,
        crop_border=args.crop_border,
    )
    backend = LpipsBackend(backbone=args.lpips_backbone, device=args.device)
    run_local_benchmark(
        outputs_root=args.outputs_root,
        reference_root=args.reference_root,
        manifests_root=args.manifests_root,
        scenes_root=args.scenes_root,
        scene_ids=args.scene_ids,
        config=config,
        lpips_backend=backend,
        report_path=args.report_path,
    )


if __name__ == "__main__":
    main()
