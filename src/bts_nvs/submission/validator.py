from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from PIL import Image, UnidentifiedImageError

from bts_nvs.data.manifest import SceneManifest


@dataclass(frozen=True, order=True)
class ValidationIssue:
    scene_id: str
    filename: str
    code: str
    message: str


def _is_link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _issue(scene_id: str, filename: str, code: str, message: str) -> ValidationIssue:
    return ValidationIssue(scene_id, filename, code, message)


def image_format_from_name(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "JPEG"
    if suffix == ".png":
        return "PNG"
    raise ValueError(f"unsupported submission image suffix: {Path(name).suffix}")


def validate_submission(
    output_root: Path,
    manifests: Mapping[str, SceneManifest],
) -> tuple[ValidationIssue, ...]:
    root = Path(output_root)
    issues: list[ValidationIssue] = []
    if not root.exists() or not root.is_dir():
        return (_issue("", "", "missing_root", f"missing output root: {root}"),)
    if _is_link(root):
        return (_issue("", "", "symlink", f"output root must not be a link: {root}"),)
    resolved_root = root.resolve()
    expected_scenes = set(manifests)
    actual_entries = {path.name: path for path in root.iterdir()}
    actual_scenes = set(actual_entries)

    for scene_id in sorted(expected_scenes - actual_scenes):
        issues.append(_issue(scene_id, "", "missing_scene", "scene directory is missing"))
    for scene_id in sorted(actual_scenes - expected_scenes):
        issues.append(_issue(scene_id, "", "extra_scene", "unexpected scene entry"))

    for scene_id in sorted(expected_scenes & actual_scenes):
        manifest = manifests[scene_id]
        scene_dir = actual_entries[scene_id]
        if _is_link(scene_dir):
            issues.append(_issue(scene_id, "", "symlink", "scene directory must not be a link"))
            continue
        if not scene_dir.is_dir():
            issues.append(_issue(scene_id, "", "wrong_scene_type", "scene entry is not a directory"))
            continue
        try:
            scene_dir.resolve(strict=True).relative_to(resolved_root)
        except (OSError, ValueError):
            issues.append(_issue(scene_id, "", "path_escape", "scene directory escapes output root"))
            continue

        expected = {
            name: (
                (intrinsics.width, intrinsics.height),
                image_format_from_name(name),
            )
            for name, intrinsics in zip(
                manifest.test_image_names, manifest.test_intrinsics, strict=True
            )
        }
        entries = {path.name: path for path in scene_dir.iterdir()}
        actual = set(entries)
        for filename in sorted(set(expected) - actual):
            issues.append(_issue(scene_id, filename, "missing_file", "render is missing"))
        for filename in sorted(actual - set(expected)):
            issues.append(_issue(scene_id, filename, "extra_file", "unexpected output entry"))

        for filename in sorted(set(expected) & actual):
            path = entries[filename]
            if _is_link(path):
                issues.append(_issue(scene_id, filename, "symlink", "output file must not be a link"))
                continue
            if not path.is_file():
                issues.append(_issue(scene_id, filename, "wrong_file_type", "output entry is not a file"))
                continue
            try:
                path.resolve(strict=True).relative_to(resolved_root)
            except (OSError, ValueError):
                issues.append(_issue(scene_id, filename, "path_escape", "output file escapes output root"))
                continue
            try:
                with Image.open(path) as image:
                    image_format = image.format
                    mode = image.mode
                    size = image.size
                    image.load()
            except (OSError, UnidentifiedImageError):
                issues.append(_issue(scene_id, filename, "decode_error", "output cannot be decoded"))
                continue
            expected_size, expected_format = expected[filename]
            if image_format != expected_format:
                issues.append(
                    _issue(
                        scene_id,
                        filename,
                        "wrong_format",
                        f"output payload is not {expected_format}",
                    )
                )
            if mode != "RGB":
                issues.append(_issue(scene_id, filename, "wrong_mode", f"output mode must be RGB, found {mode}"))
            if size != expected_size:
                issues.append(
                    _issue(
                        scene_id,
                        filename,
                        "wrong_resolution",
                        f"output resolution {size} does not match {expected_size}",
                    )
                )
    return tuple(sorted(issues))
