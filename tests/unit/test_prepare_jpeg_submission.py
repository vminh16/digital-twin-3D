from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, JpegImagePlugin

from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.submission.prepare_jpeg import (
    parse_args,
    prepare_jpeg_submission,
)


def _write_rgb(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.full((6, 8, 3), value, dtype=np.uint8)
    pixels[:, ::2, 1] = 255 - value
    Image.fromarray(pixels).save(path, format="PNG")


def _manifest(scene_id: str):
    intrinsics = CameraIntrinsics(8, 6, 5.0, 5.0, 4.0, 3.0)
    return SimpleNamespace(
        scene_id=scene_id,
        test_output_names=("first.png", "second.png"),
        test_image_names=("first.JPG", "second.jpg"),
        test_intrinsics=(intrinsics, intrinsics),
    )


def _patch_manifests(monkeypatch: pytest.MonkeyPatch, scene_ids: tuple[str, ...]):
    import bts_nvs.submission.prepare_jpeg as module

    manifests = {scene_id: _manifest(scene_id) for scene_id in scene_ids}
    monkeypatch.setattr(
        module,
        "load_scene_manifest",
        lambda path, scene_root: manifests[Path(path).parent.name],
    )


def test_converter_preserves_exact_jpeg_names_and_writes_444_rgb_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    selected = ("HCM0644", "HCM0421")
    _patch_manifests(monkeypatch, selected)
    source = tmp_path / "outputs"
    for index, scene_id in enumerate(selected):
        _write_rgb(source / scene_id / "first.png", 40 + index)
        _write_rgb(source / scene_id / "second.png", 180 - index)

    output = tmp_path / "submission"
    report_path = tmp_path / "jpeg_report.json"
    report = prepare_jpeg_submission(
        source_root=source,
        output_root=output,
        scenes_root=tmp_path / "scenes",
        manifests_root=tmp_path / "manifests",
        report_path=report_path,
        scene_ids=selected,
        quality=99,
        max_bytes=1_000_000,
    )

    assert sorted(path.name for path in output.iterdir()) == list(sorted(selected))
    for scene_id in selected:
        assert sorted(path.name for path in (output / scene_id).iterdir()) == [
            "first.JPG",
            "second.jpg",
        ]
        for name in ("first.JPG", "second.jpg"):
            with Image.open(output / scene_id / name) as image:
                image.load()
                assert image.format == "JPEG"
                assert image.mode == "RGB"
                assert image.size == (8, 6)
                assert JpegImagePlugin.get_sampling(image) == 0
    assert (source / selected[0] / "first.png").is_file()
    assert report["scene_ids"] == list(selected)
    assert report["total_images"] == 4
    assert report["quality"] == 99
    assert report["subsampling"] == "4:4:4"
    assert report["total_bytes"] == sum(
        path.stat().st_size for path in output.glob("*/*")
    )
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert not list(tmp_path.glob(".submission.*"))


def test_converter_rejects_budget_overflow_without_publishing_partial_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    selected = ("HCM0644",)
    _patch_manifests(monkeypatch, selected)
    source = tmp_path / "outputs"
    _write_rgb(source / "HCM0644" / "first.png", 40)
    _write_rgb(source / "HCM0644" / "second.png", 180)

    with pytest.raises(ValueError, match="byte limit"):
        prepare_jpeg_submission(
            source_root=source,
            output_root=tmp_path / "submission",
            scenes_root=tmp_path / "scenes",
            manifests_root=tmp_path / "manifests",
            report_path=tmp_path / "report.json",
            scene_ids=selected,
            quality=99,
            max_bytes=1,
        )

    assert not (tmp_path / "submission").exists()
    assert not (tmp_path / "report.json").exists()
    assert not list(tmp_path.glob(".submission.*"))


def test_converter_rejects_extra_source_file_and_non_jpeg_target_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    selected = ("HCM0644",)
    _patch_manifests(monkeypatch, selected)
    source = tmp_path / "outputs"
    _write_rgb(source / "HCM0644" / "first.png", 40)
    _write_rgb(source / "HCM0644" / "second.png", 180)
    _write_rgb(source / "HCM0644" / "extra.png", 10)

    with pytest.raises(ValueError, match="source entries"):
        prepare_jpeg_submission(
            source_root=source,
            output_root=tmp_path / "submission",
            scenes_root=tmp_path / "scenes",
            manifests_root=tmp_path / "manifests",
            report_path=tmp_path / "report.json",
            scene_ids=selected,
        )

    import bts_nvs.submission.prepare_jpeg as module

    broken = _manifest("HCM0644")
    broken.test_image_names = ("first.png", "second.jpg")
    monkeypatch.setattr(module, "load_scene_manifest", lambda *args: broken)
    (source / "HCM0644" / "extra.png").unlink()
    with pytest.raises(ValueError, match="JPEG suffix"):
        prepare_jpeg_submission(
            source_root=source,
            output_root=tmp_path / "submission",
            scenes_root=tmp_path / "scenes",
            manifests_root=tmp_path / "manifests",
            report_path=tmp_path / "report.json",
            scene_ids=selected,
        )


def test_converter_rejects_jpeg_payload_as_source_to_prevent_recompression(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    selected = ("HCM0644",)
    _patch_manifests(monkeypatch, selected)
    source = tmp_path / "outputs" / "HCM0644"
    source.mkdir(parents=True)
    Image.new("RGB", (8, 6), "red").save(source / "first.png", format="JPEG")
    _write_rgb(source / "second.png", 180)

    with pytest.raises(ValueError, match="source payload must be PNG"):
        prepare_jpeg_submission(
            source_root=tmp_path / "outputs",
            output_root=tmp_path / "submission",
            scenes_root=tmp_path / "scenes",
            manifests_root=tmp_path / "manifests",
            report_path=tmp_path / "report.json",
            scene_ids=selected,
        )


def test_converter_cli_parses_quality_budget_and_scene_subset(tmp_path: Path):
    args = parse_args(
        [
            "--source_root",
            str(tmp_path / "outputs"),
            "--output_root",
            str(tmp_path / "submission"),
            "--scenes_root",
            str(tmp_path / "scenes"),
            "--manifests_root",
            str(tmp_path / "manifests"),
            "--report_path",
            str(tmp_path / "report.json"),
            "--quality",
            "98",
            "--max_bytes",
            "330000000",
            "--scene_ids",
            "HCM0644",
            "HCM0421",
        ]
    )

    assert args.quality == 98
    assert args.max_bytes == 330_000_000
    assert args.scene_ids == ["HCM0644", "HCM0421"]


@pytest.mark.parametrize("scene_id", (".", "..", "folder/scene", "folder\\scene"))
def test_converter_rejects_scene_ids_that_are_not_plain_names(
    tmp_path: Path,
    scene_id: str,
):
    (tmp_path / "outputs").mkdir()

    with pytest.raises(ValueError, match="plain directory names"):
        prepare_jpeg_submission(
            source_root=tmp_path / "outputs",
            output_root=tmp_path / "submission",
            scenes_root=tmp_path / "scenes",
            manifests_root=tmp_path / "manifests",
            report_path=tmp_path / "report.json",
            scene_ids=(scene_id,),
        )
