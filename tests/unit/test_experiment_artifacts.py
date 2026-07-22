from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from bts_nvs.evaluation.experiment_report import build_experiment_report
from bts_nvs.experiments.artifacts import append_failure, validate_run_artifacts
from bts_nvs.experiments.experiment import Experiment, ExperimentStage
from bts_nvs.training.checkpoint import save_checkpoint
from bts_nvs.training.trainer import compute_config_sha256


MANIFEST_SHA256 = "a" * 64
HOLDOUT_SHA256 = "b" * 64
IMAGE_NAMES = ("a.JPG", "b.JPG", "c.JPG")


def _experiment(stage: ExperimentStage) -> Experiment:
    if stage is ExperimentStage.REFERENCE:
        return Experiment(stage, "HCM0644", "B0-reference")
    if stage is ExperimentStage.CONFIRM:
        return Experiment(
            stage,
            "HCM0644",
            "E1-density-absgrad-t04-v1",
            authorized_scene_winner="E1-density-absgrad-t04-v1",
        )
    if stage is ExperimentStage.PRODUCTION:
        return Experiment(
            stage,
            "HCM0644",
            "E1-density-absgrad-t04-v1",
            authorized_cohort_candidate="E1-density-absgrad-t04-v1",
        )
    return Experiment(stage, "HCM0644", "E1-density-absgrad-t04-v1")


def _write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _reports(experiment: Experiment, step: int) -> tuple[dict, dict, dict, dict]:
    images = {
        name: {"psnr_db": 24.0, "ssim": 0.8, "lpips": 0.2}
        for name in IMAGE_NAMES
    }
    qualification = {
        "schema_version": 1,
        "scene_id": experiment.scene_id,
        "candidate_id": experiment.candidate_id,
        "step": step,
        "image_count": len(images),
        "psnr_db_mean": 24.0,
        "ssim_mean": 0.8,
        "lpips_mean": 0.2,
        "valid_fraction_mean": 1.0,
        "peak_gaussians": 10,
        "max_vram_mb": 1024.0,
        "total_time_seconds": 20.0,
        "config_sha256": "",
        "holdout_sha256": HOLDOUT_SHA256,
        "images": images,
    }
    detail = {
        "schema_version": 1,
        "scene_id": experiment.scene_id,
        "image_count": len(images),
        "hf_l1_mean": 0.1,
        "missing_edge_mean": 0.1,
        "spurious_edge_mean": 0.1,
        "symmetric_edge_distance_mean": 0.1,
        "images": {
            name: {
                "hf_l1": 0.1,
                "missing_edge": 0.1,
                "spurious_edge": 0.1,
                "symmetric_edge_distance": 0.1,
            }
            for name in IMAGE_NAMES
        },
    }
    strata = ("easy", "medium", "hard")
    pose = {
        "schema_version": 1,
        "scene_id": experiment.scene_id,
        "algorithm": "nearest-pose-tertiles-v1",
        "holdout_algorithm": "pose-fps-v1",
        "holdout_manifest_sha256": HOLDOUT_SHA256,
        "image_count": len(images),
        "images": {
            name: {
                "nearest_train_image_name": "train.JPG",
                "pose_distance": float(index + 1),
                "center_distance": float(index + 1),
                "rotation_angle_deg": float(index + 1),
                "stratum": strata[index],
            }
            for index, name in enumerate(IMAGE_NAMES)
        },
    }
    return qualification, detail, pose, images


def _write_run(
    root: Path,
    experiment: Experiment,
    *,
    step: int | None = None,
) -> tuple[Path, str]:
    target_step = experiment.horizon if step is None else step
    run = root / "run"
    run.mkdir(parents=True)
    config = {
        "scene_id": experiment.scene_id,
        "candidate_id": experiment.candidate_id,
        "experiment_stage": experiment.stage.value,
        "max_steps": experiment.horizon,
        "internal_holdout": experiment.stage is not ExperimentStage.PRODUCTION,
    }
    config_sha256 = compute_config_sha256(config)
    (run / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    _write_json(run / "environment.json", {"device": "test", "cuda": False})
    _write_json(run / "manifest_hash.json", {"manifest_hash": MANIFEST_SHA256})
    _write_json(
        run / "summary.json",
        {
            "total_steps": target_step,
            "total_time_seconds": 20.0,
            "final_loss": 1.0,
            "final_num_gaussians": 10,
            "max_vram_mb": 1024.0,
        },
    )
    (run / "metrics.jsonl").write_text(
        "".join(
            json.dumps({"step": index, "loss": 1.0, "num_gaussians": 10})
            + "\n"
            for index in range(1, target_step + 1)
        ),
        encoding="utf-8",
    )

    if experiment.stage is not ExperimentStage.PRODUCTION:
        report_root = (
            run / "snapshots" / "step_000015000"
            if experiment.stage is ExperimentStage.CONFIRM and target_step == 15_000
            else run
        )
        qualification, detail, pose, full_images = _reports(experiment, target_step)
        qualification["config_sha256"] = config_sha256
        resources = {
            "total_time_seconds": 20.0,
            "max_vram_mb": 1024.0,
            "peak_gaussians": 10,
            "final_num_gaussians": 10,
        }
        experiment_report = build_experiment_report(
            scene_id=experiment.scene_id,
            candidate_id=experiment.candidate_id,
            step=target_step,
            config_sha256=config_sha256,
            manifest_sha256=MANIFEST_SHA256,
            holdout_sha256=HOLDOUT_SHA256,
            full_frame_report={
                "image_count": len(full_images),
                "images": full_images,
            },
            detail_report=detail,
            pose_strata_report=pose,
            resource_summary=resources,
        )
        for name, record in (
            ("qualification_report.json", qualification),
            ("detail_metrics.json", detail),
            ("pose_strata.json", pose),
            ("experiment_report.json", experiment_report),
        ):
            _write_json(report_root / name, record)
        renders = report_root / "validation_renders"
        renders.mkdir(parents=True)
        for name in IMAGE_NAMES:
            (renders / Path(name).with_suffix(".png").name).write_bytes(b"")

    if experiment.stage in (ExperimentStage.CONFIRM, ExperimentStage.PRODUCTION):
        save_checkpoint(
            run / "checkpoints" / "recovery.pt",
            target_step,
            {},
            {},
            {},
            {},
            0,
            MANIFEST_SHA256,
            config_sha256,
            precision_state={"enabled": False},
        )
    return run, config_sha256


def _validate(
    run: Path,
    experiment: Experiment,
    config_sha256: str,
    *,
    step: int | None = None,
    b0_report: dict | None = None,
):
    return validate_run_artifacts(
        run,
        experiment,
        manifest_sha256=MANIFEST_SHA256,
        config_sha256=config_sha256,
        holdout_sha256=(
            None
            if experiment.stage is ExperimentStage.PRODUCTION
            else HOLDOUT_SHA256
        ),
        expected_image_names=(
            () if experiment.stage is ExperimentStage.PRODUCTION else IMAGE_NAMES
        ),
        step=step,
        b0_report=b0_report,
    )


def test_internal_holdout_validation_uses_reports_and_exact_render_names(
    tmp_path: Path,
) -> None:
    experiment = _experiment(ExperimentStage.SCREEN)
    run, config_sha256 = _write_run(tmp_path, experiment)

    result = _validate(run, experiment, config_sha256)

    assert result.experiment_report is not None
    assert result.experiment_report["image_count"] == 3
    assert result.integrity_passed is True
    assert result.primitive_growth_controlled is True
    assert result.paired_wall_time_ratio is None
    with pytest.raises(FrozenInstanceError):
        result.integrity_passed = False


def test_production_has_an_explicit_minimal_contract_without_holdout_reports(
    tmp_path: Path,
) -> None:
    experiment = _experiment(ExperimentStage.PRODUCTION)
    run, config_sha256 = _write_run(tmp_path, experiment)

    result = _validate(run, experiment, config_sha256)

    assert result.experiment_report is None
    assert not (run / "qualification_report.json").exists()
    assert not (run / "validation_renders").exists()


def test_production_rejects_stale_internal_holdout_artifacts(tmp_path: Path) -> None:
    experiment = _experiment(ExperimentStage.PRODUCTION)
    run, config_sha256 = _write_run(tmp_path, experiment)
    _write_json(run / "qualification_report.json", {"stale": True})

    with pytest.raises(ValueError, match="production.*holdout"):
        _validate(run, experiment, config_sha256)


@pytest.mark.parametrize(
    ("relative_path", "match"),
    [
        ("qualification_report.json", "qualification_report"),
        ("detail_metrics.json", "detail_metrics"),
        ("pose_strata.json", "pose_strata"),
        ("experiment_report.json", "experiment_report"),
        ("summary.json", "summary"),
        ("metrics.jsonl", "metrics"),
        ("config.yaml", "config"),
        ("environment.json", "environment"),
        ("manifest_hash.json", "manifest_hash"),
    ],
)
def test_validation_rejects_partial_run_directories(
    tmp_path: Path, relative_path: str, match: str
) -> None:
    experiment = _experiment(ExperimentStage.SCREEN)
    run, config_sha256 = _write_run(tmp_path, experiment)
    (run / relative_path).unlink()

    with pytest.raises((FileNotFoundError, ValueError), match=match):
        _validate(run, experiment, config_sha256)


def test_validation_rejects_missing_extra_or_colliding_render_names(
    tmp_path: Path,
) -> None:
    experiment = _experiment(ExperimentStage.SCREEN)
    run, config_sha256 = _write_run(tmp_path, experiment)
    renders = run / "validation_renders"
    (renders / "a.png").unlink()
    (renders / "extra.png").write_bytes(b"")

    with pytest.raises(ValueError, match="render filenames"):
        _validate(run, experiment, config_sha256)

    with pytest.raises(ValueError, match="collide"):
        validate_run_artifacts(
            run,
            experiment,
            manifest_sha256=MANIFEST_SHA256,
            config_sha256=config_sha256,
            holdout_sha256=HOLDOUT_SHA256,
            expected_image_names=("camera.JPG", "camera.png"),
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("scene_id", "HCM0674", "scene"),
        ("candidate_id", "E1-density-scale005-v1", "candidate"),
        ("step", 6_999, "step"),
        ("manifest_sha256", "c" * 64, "manifest"),
        ("config_sha256", "c" * 64, "config"),
        ("holdout_sha256", "c" * 64, "holdout"),
    ],
)
def test_validation_rejects_stale_report_identity(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    experiment = _experiment(ExperimentStage.SCREEN)
    run, config_sha256 = _write_run(tmp_path, experiment)
    path = run / "experiment_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report[field] = value
    _write_json(path, report)

    with pytest.raises(ValueError, match=match):
        _validate(run, experiment, config_sha256)


def test_validation_rejects_nonfinite_metrics_failures_growth_and_temp_state(
    tmp_path: Path,
) -> None:
    experiment = _experiment(ExperimentStage.SCREEN)
    run, config_sha256 = _write_run(tmp_path, experiment)
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["primitive_growth_controlled"] = False
    _write_json(summary_path, summary)

    with pytest.raises(ValueError, match="primitive growth"):
        _validate(run, experiment, config_sha256)

    summary["primitive_growth_controlled"] = True
    _write_json(summary_path, summary)
    _write_json(run / "failure.json", {"reason": "out_of_memory"})
    with pytest.raises(ValueError, match="failure record"):
        _validate(run, experiment, config_sha256)

    (run / "failure.json").unlink()
    (run / ".summary.json.tmp").write_text("partial", encoding="utf-8")
    with pytest.raises(ValueError, match="partial"):
        _validate(run, experiment, config_sha256)


def test_validation_rejects_nonfinite_or_incomplete_metric_stream(tmp_path: Path) -> None:
    experiment = _experiment(ExperimentStage.SCREEN)
    run, config_sha256 = _write_run(tmp_path, experiment)
    metrics = run / "metrics.jsonl"
    records = metrics.read_text(encoding="utf-8").splitlines()
    records[-1] = '{"step": 7000, "loss": NaN, "num_gaussians": 10}'
    metrics.write_text("\n".join(records) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="finite JSON|non-finite loss"):
        _validate(run, experiment, config_sha256)

    metrics.write_text("\n".join(records[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="6999|7000"):
        _validate(run, experiment, config_sha256)


def test_qualification_resources_must_match_common_run_artifacts(tmp_path: Path) -> None:
    experiment = _experiment(ExperimentStage.SCREEN)
    run, config_sha256 = _write_run(tmp_path, experiment)
    qualification_path = run / "qualification_report.json"
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["total_time_seconds"] = 21.0
    _write_json(qualification_path, qualification)

    with pytest.raises(ValueError, match="qualification.*total_time_seconds"):
        _validate(run, experiment, config_sha256)


def test_finite_signed_full_frame_metrics_remain_valid(tmp_path: Path) -> None:
    experiment = _experiment(ExperimentStage.SCREEN)
    run, config_sha256 = _write_run(tmp_path, experiment)
    qualification_path = run / "qualification_report.json"
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    qualification["ssim_mean"] = -0.1
    for image in qualification["images"].values():
        image["ssim"] = -0.1
    _write_json(qualification_path, qualification)

    detail = json.loads((run / "detail_metrics.json").read_text(encoding="utf-8"))
    pose = json.loads((run / "pose_strata.json").read_text(encoding="utf-8"))
    original = json.loads((run / "experiment_report.json").read_text(encoding="utf-8"))
    rebuilt = build_experiment_report(
        scene_id=experiment.scene_id,
        candidate_id=experiment.candidate_id,
        step=7_000,
        config_sha256=config_sha256,
        manifest_sha256=MANIFEST_SHA256,
        holdout_sha256=HOLDOUT_SHA256,
        full_frame_report=qualification,
        detail_report=detail,
        pose_strata_report=pose,
        resource_summary=original["resources"],
    )
    _write_json(run / "experiment_report.json", rebuilt)

    assert _validate(run, experiment, config_sha256).integrity_passed is True


def test_checkpoint_policy_and_15k_snapshot_are_enforced(tmp_path: Path) -> None:
    screen = _experiment(ExperimentStage.SCREEN)
    screen_run, screen_hash = _write_run(tmp_path / "screen", screen)
    forbidden = screen_run / "nested" / "model.pth"
    forbidden.parent.mkdir()
    forbidden.write_bytes(b"model")
    with pytest.raises(ValueError, match="checkpoint"):
        _validate(screen_run, screen, screen_hash)

    confirm = _experiment(ExperimentStage.CONFIRM)
    confirm_run, confirm_hash = _write_run(tmp_path / "confirm", confirm, step=15_000)
    snapshot_checkpoint = (
        confirm_run / "snapshots" / "step_000015000" / "model.pt"
    )
    snapshot_checkpoint.write_bytes(b"model")
    with pytest.raises(ValueError, match="checkpoint"):
        _validate(confirm_run, confirm, confirm_hash, step=15_000)

    snapshot_checkpoint.unlink()
    (confirm_run / "checkpoints" / "extra.pt").write_bytes(b"model")
    with pytest.raises(ValueError, match="recovery.pt"):
        _validate(confirm_run, confirm, confirm_hash, step=15_000)


def test_resource_limits_and_paired_time_ratio_use_locked_validators(
    tmp_path: Path,
) -> None:
    experiment = _experiment(ExperimentStage.SCREEN)
    run, config_sha256 = _write_run(tmp_path, experiment)
    report_path = run / "experiment_report.json"
    b0 = json.loads(report_path.read_text(encoding="utf-8"))
    b0["candidate_id"] = "B0-reference"
    b0["resources"]["total_time_seconds"] = 10.0

    with pytest.raises(ValueError, match="paired_wall_time_ratio"):
        _validate(run, experiment, config_sha256, b0_report=b0)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["resources"]["max_vram_mb"] = 23 * 1024
    _write_json(report_path, report)
    with pytest.raises(ValueError, match="peak_vram_mb"):
        _validate(run, experiment, config_sha256)


def test_append_failure_is_atomic_finite_and_preserves_previous_records(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "failures.jsonl"
    experiment = _experiment(ExperimentStage.SCREEN)

    first = append_failure(
        ledger,
        experiment=experiment,
        command_argv=["python", "train.py", "--scene", "HCM0644"],
        reason="process exited with code 1",
        provenance={"git_commit": "1" * 40},
    )
    before = ledger.read_bytes()
    second = append_failure(
        ledger,
        experiment=experiment,
        command_argv=["python", "train.py", "--resume"],
        reason="invalid-state recovery",
        provenance={"manifest_sha256": MANIFEST_SHA256},
    )

    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert records == [first, second]
    assert ledger.read_bytes().startswith(before)
    assert records[0]["stage"] == "screen"
    assert records[0]["scene_id"] == "HCM0644"
    assert records[0]["candidate_id"] == "E1-density-absgrad-t04-v1"
    assert records[0]["command_argv"] == [
        "python",
        "train.py",
        "--scene",
        "HCM0644",
    ]
    assert not ledger.with_name(f".{ledger.name}.tmp").exists()

    stable = ledger.read_bytes()
    with pytest.raises(ValueError, match="finite"):
        append_failure(
            ledger,
            experiment=experiment,
            command_argv=["python"],
            reason="bad provenance",
            provenance={"loss": float("nan")},
        )
    assert ledger.read_bytes() == stable


@pytest.mark.parametrize(
    "malformed",
    [b"not-json\n", b"[]\n", b"{}\n", b'{"loss":NaN}\n'],
)
def test_append_failure_rejects_malformed_ledger_without_overwriting(
    tmp_path: Path, malformed: bytes
) -> None:
    ledger = tmp_path / "failures.jsonl"
    ledger.write_bytes(malformed)
    experiment = _experiment(ExperimentStage.REFERENCE)

    with pytest.raises(ValueError, match="ledger"):
        append_failure(
            ledger,
            experiment=experiment,
            command_argv=["python", "train.py"],
            reason="failed",
            provenance={"git_commit": "1" * 40},
        )

    assert ledger.read_bytes() == malformed
