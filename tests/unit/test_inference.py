from types import SimpleNamespace
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from bts_nvs.cameras.distortion import (
    CameraDistortion,
    undistort_normalized_points,
)
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.rendering.inference import (
    gaussians_from_checkpoint,
    normalized_test_world_to_camera,
    redistort_render,
    render_test_camera,
)
from bts_nvs.rendering.run_inference import parse_args, run_inference
from bts_nvs.training.full_training import TrainedRun


def _checkpoint() -> dict:
    count = 2
    return {
        "gaussians": {
            "means": torch.arange(count * 3, dtype=torch.float32).reshape(count, 3),
            "scales": torch.full((count, 3), -2.0),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
            "opacities": torch.zeros(count),
            "sh0": torch.zeros((count, 1, 3)),
            "shN": torch.zeros((count, 15, 3)),
        }
    }


def test_gaussians_are_reconstructed_from_checkpoint_on_requested_device():
    checkpoint = _checkpoint()

    gaussians = gaussians_from_checkpoint(checkpoint, torch.device("cpu"))

    assert gaussians.num_gaussians == 2
    assert gaussians.means.device.type == "cpu"
    assert torch.equal(gaussians.state_dict()["means"], checkpoint["gaussians"]["means"])
    assert gaussians.training is False
    assert all(not parameter.requires_grad for parameter in gaussians.parameters())


def test_checkpoint_reconstruction_rejects_missing_or_wrong_shaped_state():
    with pytest.raises(ValueError, match="checkpoint Gaussian state"):
        gaussians_from_checkpoint({}, torch.device("cpu"))
    broken = _checkpoint()
    broken["gaussians"]["means"] = torch.zeros((2, 2))
    with pytest.raises(ValueError, match="checkpoint Gaussian state"):
        gaussians_from_checkpoint(broken, torch.device("cpu"))


def test_test_pose_normalization_preserves_rotation_and_transforms_center():
    raw = np.eye(4, dtype=np.float64)
    raw[:3, 3] = [-2.0, -4.0, -6.0]
    normalization = np.eye(4, dtype=np.float64)
    normalization[:3, :3] *= 0.5
    normalization[:3, 3] = [-0.5, -1.0, -1.5]

    normalized = normalized_test_world_to_camera(raw, normalization)

    np.testing.assert_array_equal(normalized[:3, :3], raw[:3, :3])
    np.testing.assert_allclose(normalized[:3, 3], [-0.5, -1.0, -1.5])


def test_pinhole_redistortion_is_an_exact_copy():
    image = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    intrinsics = CameraIntrinsics(5, 4, 4.0, 4.0, 2.0, 1.5)

    output = redistort_render(
        image, intrinsics, CameraDistortion("PINHOLE", ())
    )

    np.testing.assert_array_equal(output, image)
    assert output is not image


def test_simple_radial_redistortion_maps_destination_to_undistorted_source(
    monkeypatch: pytest.MonkeyPatch,
):
    import bts_nvs.rendering.inference as module

    image = np.zeros((3, 3, 3), dtype=np.float32)
    intrinsics = CameraIntrinsics(3, 3, 2.0, 2.0, 1.0, 1.0)
    distortion = CameraDistortion("SIMPLE_RADIAL", (0.1,))
    captured = {}

    def fake_remap(source, map_x, map_y, **kwargs):
        captured["x"] = map_x.copy()
        captured["y"] = map_y.copy()
        return source.copy()

    monkeypatch.setattr(module.cv2, "remap", fake_remap)

    redistort_render(image, intrinsics, distortion)

    destination = np.array([[[-0.5, -0.5]]], dtype=np.float64)
    source = undistort_normalized_points(destination, distortion)[0, 0]
    assert captured["x"][0, 0] == pytest.approx(2.0 * source[0] + 1.0)
    assert captured["y"][0, 0] == pytest.approx(2.0 * source[1] + 1.0)
    assert captured["x"][1, 1] == pytest.approx(1.0)
    assert captured["y"][1, 1] == pytest.approx(1.0)


def test_redistortion_map_is_cached_for_shared_camera_calibration(
    monkeypatch: pytest.MonkeyPatch,
):
    import bts_nvs.rendering.inference as module

    module._redistortion_maps.cache_clear()
    intrinsics = CameraIntrinsics(3, 3, 2.0, 2.0, 1.0, 1.0)
    distortion = CameraDistortion("SIMPLE_RADIAL", (0.1,))
    image = np.zeros((3, 3, 3), dtype=np.float32)
    original = module.undistort_normalized_points
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "undistort_normalized_points", counted)
    redistort_render(image, intrinsics, distortion)
    redistort_render(image, intrinsics, distortion)

    assert calls == 1


def test_render_test_camera_uses_normalized_pose_and_native_camera(
    monkeypatch: pytest.MonkeyPatch,
):
    import bts_nvs.rendering.inference as module

    gaussians = gaussians_from_checkpoint(_checkpoint(), torch.device("cpu"))
    intrinsics = CameraIntrinsics(5, 4, 4.0, 4.0, 2.0, 1.5)
    raw = np.eye(4, dtype=np.float64)
    raw[0, 3] = -2.0
    normalization = np.eye(4, dtype=np.float64)
    normalization[:3, :3] *= 0.5
    captured = {}

    def fake_render(model, viewmat, camera, active_sh_degree):
        captured.update(
            model=model,
            viewmat=np.asarray(viewmat),
            camera=camera,
            degree=active_sh_degree,
        )
        return SimpleNamespace(rgb=torch.full((4, 5, 3), 0.25))

    monkeypatch.setattr(module, "render_gaussians", fake_render)

    image = render_test_camera(
        gaussians,
        raw,
        intrinsics,
        CameraDistortion("PINHOLE", ()),
        normalization,
        active_sh_degree=3,
    )

    assert captured["model"] is gaussians
    assert captured["camera"] == intrinsics
    assert captured["degree"] == 3
    assert captured["viewmat"][0, 3] == pytest.approx(-1.0)
    assert image.shape == (4, 5, 3)
    assert image.dtype == np.float32
    assert np.all(image == 0.25)


def _fake_manifest(scene_id: str):
    intrinsics = CameraIntrinsics(3, 2, 2.0, 2.0, 1.0, 0.5)
    return SimpleNamespace(
        scene_id=scene_id,
        test_image_names=(f"{scene_id}_a.JPG", f"{scene_id}_b.png"),
        test_output_names=(f"{scene_id}_a.png", f"{scene_id}_b.png"),
        test_world_to_camera=np.stack((np.eye(4), np.eye(4))),
        test_intrinsics=(intrinsics, intrinsics),
        test_distortion=(
            CameraDistortion("PINHOLE", ()),
            CameraDistortion("PINHOLE", ()),
        ),
        normalization_transform=np.eye(4),
    )


def _batch_layout(tmp_path: Path, scene_ids: tuple[str, ...]):
    scenes = tmp_path / "scenes"
    manifests = tmp_path / "manifests"
    full = tmp_path / "full"
    for scene_id in scene_ids:
        (scenes / scene_id).mkdir(parents=True)
        manifest_dir = manifests / scene_id
        manifest_dir.mkdir(parents=True)
        (manifest_dir / "manifest.json").write_text("{}", encoding="utf-8")
        checkpoint_dir = full / "scenes" / scene_id / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "recovery.pt").write_bytes(scene_id.encode())
    return scenes, manifests, full


def _patch_batch_dependencies(monkeypatch, selected, render_error=None):
    import bts_nvs.rendering.run_inference as module

    manifests = {scene_id: _fake_manifest(scene_id) for scene_id in selected}
    rendered = []
    monkeypatch.setattr(module, "validate_scene_pool", lambda *args: selected)
    monkeypatch.setattr(
        module, "load_or_create_backend_decision", lambda *args: object()
    )
    monkeypatch.setattr(
        module,
        "load_scene_manifest",
        lambda path, scene_root: manifests[Path(path).parent.name],
    )
    monkeypatch.setattr(
        module,
        "load_trained_checkpoint",
        lambda run, scene, manifest, decision: (
            TrainedRun(30_000, "c" * 64, "m" * 64),
            {"scene_id": scene, "active_sh_degree": 3, "gaussians": {}},
        ),
    )
    monkeypatch.setattr(
        module, "gaussians_from_checkpoint", lambda state, device: state["scene_id"]
    )

    def fake_render(model, *args, **kwargs):
        rendered.append(model)
        if render_error is not None:
            raise render_error
        return np.full((2, 3, 3), 0.5, dtype=np.float32)

    monkeypatch.setattr(module, "render_test_camera", fake_render)
    return rendered


def test_batch_inference_renders_selected_scenes_and_validates_exact_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    selected = ("HCM0644", "HCM0421")
    scenes, manifests, full = _batch_layout(tmp_path, selected)
    rendered = _patch_batch_dependencies(monkeypatch, selected)
    output = tmp_path / "outputs"
    report_path = tmp_path / "inference_report.json"

    report = run_inference(
        scenes_root=scenes,
        manifests_root=manifests,
        backend_root=tmp_path / "backend",
        full_root=full,
        output_root=output,
        report_path=report_path,
        scene_ids=selected,
        device=torch.device("cpu"),
    )

    assert rendered == ["HCM0644", "HCM0644", "HCM0421", "HCM0421"]
    assert sorted(path.name for path in output.iterdir()) == list(sorted(selected))
    for scene_id in selected:
        assert sorted(path.name for path in (output / scene_id).iterdir()) == [
            f"{scene_id}_a.JPG",
            f"{scene_id}_b.png",
        ]
        with Image.open(output / scene_id / f"{scene_id}_a.JPG") as image:
            assert image.format == "JPEG"
        with Image.open(output / scene_id / f"{scene_id}_b.png") as image:
            assert image.format == "PNG"
    assert report["scene_ids"] == list(selected)
    assert report["jpeg_quality"] == 98
    assert [item["scene_id"] for item in report["scenes"]] == list(selected)
    encoded = report_path.read_text(encoding="utf-8").lower()
    assert json.loads(encoded)["total_images"] == 4
    assert all(token not in encoded for token in ("psnr", "ssim", "lpips", "score"))


def test_batch_inference_rejects_existing_output_and_cleans_staging_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    selected = ("HCM0644",)
    scenes, manifests, full = _batch_layout(tmp_path, selected)
    output = tmp_path / "outputs"
    output.mkdir()
    with pytest.raises(FileExistsError, match="output root"):
        run_inference(
            scenes_root=scenes,
            manifests_root=manifests,
            backend_root=tmp_path / "backend",
            full_root=full,
            output_root=output,
            report_path=tmp_path / "report.json",
            scene_ids=selected,
            device=torch.device("cpu"),
        )

    output.rmdir()
    _patch_batch_dependencies(monkeypatch, selected, RuntimeError("render failed"))
    with pytest.raises(RuntimeError, match="render failed"):
        run_inference(
            scenes_root=scenes,
            manifests_root=manifests,
            backend_root=tmp_path / "backend",
            full_root=full,
            output_root=output,
            report_path=tmp_path / "report.json",
            scene_ids=selected,
            device=torch.device("cpu"),
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".outputs.*"))


def test_inference_cli_parses_paths_and_selected_scenes(tmp_path: Path):
    args = parse_args(
        [
            "--scenes_root",
            str(tmp_path / "scenes"),
            "--manifests_root",
            str(tmp_path / "manifests"),
            "--backend_root",
            str(tmp_path / "backend"),
            "--full_root",
            str(tmp_path / "full"),
            "--output_root",
            str(tmp_path / "outputs"),
            "--report_path",
            str(tmp_path / "report.json"),
            "--scene_ids",
            "HCM0644",
            "HCM0421",
            "--jpeg_quality",
            "97",
            "--allow_noncanonical_scenes",
        ]
    )
    assert args.output_root == tmp_path / "outputs"
    assert args.scene_ids == ["HCM0644", "HCM0421"]
    assert args.jpeg_quality == 97
    assert args.allow_noncanonical_scenes is True


def test_noncanonical_inference_requires_explicit_safe_scene_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    selected = ("chair",)
    scenes, manifests, full = _batch_layout(tmp_path, selected)
    _patch_batch_dependencies(monkeypatch, selected)

    report = run_inference(
        scenes_root=scenes,
        manifests_root=manifests,
        backend_root=tmp_path / "backend",
        full_root=full,
        output_root=tmp_path / "outputs",
        report_path=tmp_path / "report.json",
        scene_ids=selected,
        allow_noncanonical_scenes=True,
        device=torch.device("cpu"),
    )

    assert report["scene_ids"] == ["chair"]

    with pytest.raises(ValueError, match="explicit"):
        run_inference(
            scenes_root=scenes,
            manifests_root=manifests,
            backend_root=tmp_path / "backend",
            full_root=full,
            output_root=tmp_path / "other",
            report_path=tmp_path / "other.json",
            allow_noncanonical_scenes=True,
            device=torch.device("cpu"),
        )
