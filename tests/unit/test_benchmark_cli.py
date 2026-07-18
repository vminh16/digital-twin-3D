import json
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.evaluation.metrics import MetricConfig
from bts_nvs.evaluation.run_benchmark import (
    parse_args,
    run_local_benchmark,
)
import bts_nvs.evaluation.run_benchmark as benchmark_module


class ConstantLpips:
    package = "fake-lpips"
    version = "test"
    device = "cpu"
    dtype = "float32"

    def __call__(self, prediction, target):
        return 0.0


def test_benchmark_cli_does_not_depend_on_training_or_rendering_modules():
    source = inspect.getsource(benchmark_module)
    assert "bts_nvs.training" not in source
    assert "bts_nvs.rendering" not in source


def _write_rgb(path: Path, value: int = 20):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((16, 16, 3), value, dtype=np.uint8)).save(path)


def _patch_manifests(monkeypatch, selected):
    import bts_nvs.evaluation.run_benchmark as module

    intrinsics = CameraIntrinsics(16, 16, 10.0, 10.0, 8.0, 8.0)
    manifests = {
        scene_id: SimpleNamespace(
            scene_id=scene_id,
            test_output_names=(f"{scene_id}.png",),
            test_intrinsics=(intrinsics,),
        )
        for scene_id in selected
    }
    monkeypatch.setattr(module, "validate_scene_pool", lambda *args: selected)
    monkeypatch.setattr(
        module,
        "load_scene_manifest",
        lambda path, scene_root: manifests[Path(path).parent.name],
    )


def test_local_benchmark_reads_existing_output_and_explicit_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    selected = ("HCM0644", "HCM0421")
    _patch_manifests(monkeypatch, selected)
    outputs = tmp_path / "outputs"
    references = tmp_path / "references"
    for scene_id in selected:
        _write_rgb(outputs / scene_id / f"{scene_id}.png")
        _write_rgb(references / scene_id / f"{scene_id}.png")
    report_path = tmp_path / "benchmark.json"

    report = run_local_benchmark(
        outputs_root=outputs,
        reference_root=references,
        manifests_root=tmp_path / "manifests",
        scenes_root=tmp_path / "scenes",
        scene_ids=selected,
        config=MetricConfig(psnr_max=40.0),
        lpips_backend=ConstantLpips(),
        report_path=report_path,
    )

    assert report["final_score"] == pytest.approx(1.0)
    assert list(report["scenes"]) == sorted(selected)
    assert json.loads(report_path.read_text())["metadata"]["psnr_max"] == 40.0


def test_local_benchmark_rejects_top_level_or_image_schema_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    selected = ("HCM0644",)
    _patch_manifests(monkeypatch, selected)
    outputs = tmp_path / "outputs"
    references = tmp_path / "references"
    _write_rgb(outputs / "HCM0644" / "HCM0644.png")
    _write_rgb(references / "HCM0644" / "HCM0644.png")
    (outputs / "EXTRA").mkdir()

    with pytest.raises(ValueError, match="scene directories"):
        run_local_benchmark(
            outputs_root=outputs,
            reference_root=references,
            manifests_root=tmp_path / "manifests",
            scenes_root=tmp_path / "scenes",
            scene_ids=selected,
            config=MetricConfig(psnr_max=40.0),
            lpips_backend=ConstantLpips(),
            report_path=tmp_path / "benchmark.json",
        )


def test_benchmark_cli_requires_metric_identity_and_parses_scene_selection(
    tmp_path: Path,
):
    args = parse_args(
        [
            "--outputs_root",
            str(tmp_path / "outputs"),
            "--reference_root",
            str(tmp_path / "references"),
            "--manifests_root",
            str(tmp_path / "manifests"),
            "--scenes_root",
            str(tmp_path / "scenes"),
            "--scene_ids",
            "HCM0644",
            "HCM0421",
            "--psnr_max",
            "40",
            "--lpips_backbone",
            "alex",
            "--report_path",
            str(tmp_path / "report.json"),
        ]
    )
    assert args.psnr_max == 40.0
    assert args.scene_ids == ["HCM0644", "HCM0421"]
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--outputs_root",
                "outputs",
                "--reference_root",
                "references",
                "--manifests_root",
                "manifests",
                "--scenes_root",
                "scenes",
                "--report_path",
                "report.json",
            ]
        )
