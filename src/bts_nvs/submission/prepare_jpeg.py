from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

from PIL import Image, JpegImagePlugin, UnidentifiedImageError

from bts_nvs.data.manifest import load_scene_manifest


DEFAULT_MAX_BYTES = 350_000_000
DEFAULT_QUALITY = 99


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_scene_ids(scene_ids: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(scene_ids)
    if not selected:
        raise ValueError("at least one scene ID is required")
    if len(set(selected)) != len(selected):
        raise ValueError("scene IDs must be unique")
    if any(
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or Path(value).name != value
        or Path(value).is_absolute()
        for value in selected
    ):
        raise ValueError("scene IDs must be plain directory names")
    return selected


def _validate_target_names(names: Sequence[str]) -> None:
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("JPEG target names collide case-insensitively")
    for name in names:
        path = Path(name)
        if path.name != name or path.is_absolute():
            raise ValueError("JPEG target names must be plain filenames")
        if path.suffix.casefold() not in {".jpg", ".jpeg"}:
            raise ValueError(f"target does not have a JPEG suffix: {name}")


def _encode_and_validate(
    source: Path,
    target: Path,
    expected_size: tuple[int, int],
    quality: int,
) -> None:
    try:
        with Image.open(source) as image:
            image.load()
            if image.format != "PNG":
                raise ValueError(f"source payload must be PNG: {source}")
            if image.mode != "RGB":
                raise ValueError(f"source image must be RGB: {source}")
            if image.size != expected_size:
                raise ValueError(
                    f"source resolution {image.size} does not match {expected_size}: {source}"
                )
            image.save(
                target,
                format="JPEG",
                quality=quality,
                subsampling=0,
                optimize=True,
                progressive=False,
            )
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"cannot decode source image: {source}") from error

    try:
        with Image.open(target) as encoded:
            encoded.load()
            valid = (
                encoded.format == "JPEG"
                and encoded.mode == "RGB"
                and encoded.size == expected_size
                and JpegImagePlugin.get_sampling(encoded) == 0
            )
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"cannot decode encoded JPEG: {target}") from error
    if not valid:
        raise ValueError(f"encoded JPEG violates the submission contract: {target}")


def _write_report_temporary(path: Path, report: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return temporary


def prepare_jpeg_submission(
    *,
    source_root: Path,
    output_root: Path,
    scenes_root: Path,
    manifests_root: Path,
    report_path: Path,
    scene_ids: Sequence[str],
    quality: int = DEFAULT_QUALITY,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict:
    if isinstance(quality, bool) or not isinstance(quality, int) or not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be an integer from 1 to 100")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("maximum bytes must be a positive integer")
    selected = _validate_scene_ids(scene_ids)
    source = Path(source_root)
    output = Path(output_root)
    report_file = Path(report_path)
    if not source.is_dir():
        raise FileNotFoundError(f"source root does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"output root already exists: {output}")
    if report_file.exists():
        raise FileExistsError(f"JPEG report already exists: {report_file}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    report_temporary: Path | None = None
    try:
        scene_reports = []
        total_bytes = 0
        total_images = 0
        for scene_id in selected:
            manifest = load_scene_manifest(
                Path(manifests_root) / scene_id / "manifest.json",
                Path(scenes_root) / scene_id,
            )
            source_names = tuple(manifest.test_output_names)
            target_names = tuple(manifest.test_image_names)
            intrinsics = tuple(manifest.test_intrinsics)
            if not (len(source_names) == len(target_names) == len(intrinsics)):
                raise ValueError(f"manifest test fields have different lengths: {scene_id}")
            _validate_target_names(target_names)

            source_scene = source / scene_id
            if not source_scene.is_dir():
                raise FileNotFoundError(f"source scene does not exist: {source_scene}")
            actual_entries = {path.name for path in source_scene.iterdir()}
            if actual_entries != set(source_names):
                raise ValueError(f"source entries do not match the manifest: {scene_id}")

            target_scene = staging / scene_id
            target_scene.mkdir()
            image_reports = []
            for source_name, target_name, camera in zip(
                source_names, target_names, intrinsics, strict=True
            ):
                source_path = source_scene / source_name
                if not source_path.is_file() or source_path.is_symlink():
                    raise ValueError(f"source image must be a regular file: {source_path}")
                target_path = target_scene / target_name
                _encode_and_validate(
                    source_path,
                    target_path,
                    (camera.width, camera.height),
                    quality,
                )
                size = target_path.stat().st_size
                total_bytes += size
                total_images += 1
                image_reports.append(
                    {
                        "filename": target_name,
                        "bytes": size,
                        "sha256": _sha256(target_path),
                    }
                )
            scene_reports.append(
                {
                    "scene_id": scene_id,
                    "image_count": len(image_reports),
                    "total_bytes": sum(item["bytes"] for item in image_reports),
                    "images": image_reports,
                }
            )

        if total_bytes > max_bytes:
            raise ValueError(
                f"JPEG submission exceeds byte limit: {total_bytes} > {max_bytes}"
            )
        report = {
            "schema_version": 1,
            "scene_ids": list(selected),
            "quality": quality,
            "subsampling": "4:4:4",
            "progressive": False,
            "max_bytes": max_bytes,
            "total_bytes": total_bytes,
            "total_images": total_images,
            "scenes": scene_reports,
        }
        report_temporary = _write_report_temporary(report_file, report)
        staging.replace(output)
        os.replace(report_temporary, report_file)
        report_temporary = None
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if report_temporary is not None:
            report_temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert validated PNG renders into exact-name JPEG submission files."
    )
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--scenes_root", type=Path, required=True)
    parser.add_argument("--manifests_root", type=Path, required=True)
    parser.add_argument("--report_path", type=Path, required=True)
    parser.add_argument("--quality", type=int, default=DEFAULT_QUALITY)
    parser.add_argument("--max_bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--scene_ids", nargs="+", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = prepare_jpeg_submission(
        source_root=args.source_root,
        output_root=args.output_root,
        scenes_root=args.scenes_root,
        manifests_root=args.manifests_root,
        report_path=args.report_path,
        scene_ids=args.scene_ids,
        quality=args.quality,
        max_bytes=args.max_bytes,
    )
    print(
        f"Prepared {report['total_images']} JPEG(s), "
        f"{report['total_bytes']} bytes at quality {report['quality']}."
    )


if __name__ == "__main__":
    main()
