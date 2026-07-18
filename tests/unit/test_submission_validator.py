from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bts_nvs.cameras.distortion import CameraDistortion
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.data.manifest import SceneManifest
from bts_nvs.submission.validator import ValidationIssue, validate_submission


def _manifest(scene_id="scene_001"):
    identity = np.eye(4)
    intrinsics = CameraIntrinsics(8, 6, 5.0, 5.0, 4.0, 3.0)
    distortion = CameraDistortion("PINHOLE", ())
    return SceneManifest(
        1,
        scene_id,
        ("train/images/a.png",),
        ("a.png",),
        identity[None],
        identity[None],
        (intrinsics,),
        (distortion,),
        ("target.JPG",),
        ("target.png",),
        identity[None],
        (intrinsics,),
        (distortion,),
        np.empty((0, 3)),
        np.empty((0, 3), dtype=np.uint8),
        identity,
        identity,
    )


def _valid_output(tmp_path: Path) -> Path:
    root = tmp_path / "outputs"
    scene = root / "scene_001"
    scene.mkdir(parents=True)
    Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8)).save(
        scene / "target.JPG", format="JPEG"
    )
    return root


def _codes(issues):
    return [issue.code for issue in issues]


def test_valid_submission_has_no_issues(tmp_path):
    output = _valid_output(tmp_path)

    assert validate_submission(output, {"scene_001": _manifest()}) == ()


def test_missing_extra_and_wrong_case_are_reported(tmp_path):
    output = tmp_path / "outputs"
    scene = output / "scene_001"
    scene.mkdir(parents=True)
    Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8)).save(scene / "TARGET.JPG")
    (scene / "extra.txt").write_text("extra", encoding="utf-8")

    issues = validate_submission(output, {"scene_001": _manifest()})

    assert issues == tuple(sorted(issues))
    assert {issue.code for issue in issues} == {"extra_file", "missing_file"}


@pytest.mark.parametrize(
    ("writer", "expected_code"),
    [
        (lambda path: path.write_bytes(b"not png"), "decode_error"),
        (
            lambda path: Image.fromarray(np.zeros((6, 8), dtype=np.uint8)).save(path),
            "wrong_mode",
        ),
        (
            lambda path: Image.fromarray(np.zeros((5, 8, 3), dtype=np.uint8)).save(path),
            "wrong_resolution",
        ),
    ],
)
def test_invalid_jpeg_contract_is_reported(tmp_path, writer, expected_code):
    output = tmp_path / "outputs"
    scene = output / "scene_001"
    scene.mkdir(parents=True)
    writer(scene / "target.JPG")

    issues = validate_submission(output, {"scene_001": _manifest()})

    assert expected_code in _codes(issues)


def test_jpeg_extension_with_png_payload_is_rejected(tmp_path):
    output = tmp_path / "outputs"
    scene = output / "scene_001"
    scene.mkdir(parents=True)
    Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8)).save(
        scene / "target.JPG", format="PNG"
    )

    assert "wrong_format" in _codes(
        validate_submission(output, {"scene_001": _manifest()})
    )


@pytest.mark.parametrize("mode", ["L", "CMYK"])
def test_non_rgb_jpeg_modes_are_rejected(tmp_path, mode):
    output = tmp_path / "outputs"
    scene = output / "scene_001"
    scene.mkdir(parents=True)
    Image.new(mode, (8, 6)).save(scene / "target.JPG", format="JPEG")

    assert "wrong_mode" in _codes(
        validate_submission(output, {"scene_001": _manifest()})
    )


def test_cmyk_payload_is_rejected(tmp_path):
    output = tmp_path / "outputs"
    scene = output / "scene_001"
    scene.mkdir(parents=True)
    Image.new("CMYK", (8, 6)).save(scene / "target.JPG", format="JPEG")

    codes = _codes(validate_submission(output, {"scene_001": _manifest()}))
    assert "wrong_mode" in codes


def test_missing_and_extra_scenes_are_reported(tmp_path):
    output = tmp_path / "outputs"
    (output / "unexpected").mkdir(parents=True)

    issues = validate_submission(output, {"scene_001": _manifest()})

    assert {issue.code for issue in issues} == {"extra_scene", "missing_scene"}


def test_output_symlink_is_rejected(tmp_path):
    outside = tmp_path / "outside.JPG"
    Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8)).save(outside, format="JPEG")
    output = tmp_path / "outputs"
    scene = output / "scene_001"
    scene.mkdir(parents=True)
    try:
        (scene / "target.JPG").symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    issues = validate_submission(output, {"scene_001": _manifest()})

    assert any(
        isinstance(issue, ValidationIssue) and issue.code == "symlink"
        for issue in issues
    )
