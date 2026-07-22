from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from bts_nvs.evaluation.experiment_report import build_experiment_report
from bts_nvs.experiments.experiment import (
    Experiment,
    ExperimentStage,
    validate_paired_wall_time_ratio,
    validate_peak_vram_mb,
)
from bts_nvs.experiments.provenance import (
    canonical_json_sha256,
    load_json_artifact,
)
from bts_nvs.training.full_training import _validate_metrics as _validate_30k_metrics
from bts_nvs.training.run_training import validate_recovery_checkpoint
from bts_nvs.training.trainer import compute_config_sha256


_HOLDOUT_REPORT_NAMES = (
    "qualification_report.json",
    "detail_metrics.json",
    "pose_strata.json",
    "experiment_report.json",
)
_MODEL_SUFFIXES = frozenset((".pt", ".pth"))
_RECOVERY_PATH = Path("checkpoints") / "recovery.pt"
_OPTIONAL_REPORT_FIELDS = frozenset(
    ("integrity_passed", "primitive_growth_controlled")
)


@dataclass(frozen=True)
class ArtifactValidationResult:
    experiment_report: dict[str, object] | None
    paired_wall_time_ratio: float | None
    integrity_passed: bool
    primitive_growth_controlled: bool


def validate_run_artifacts(
    run_dir: str | Path,
    experiment: Experiment,
    *,
    manifest_sha256: str,
    config_sha256: str,
    expected_image_names: Sequence[str],
    holdout_sha256: str | None = None,
    step: int | None = None,
    b0_report: Mapping[str, object] | None = None,
) -> ArtifactValidationResult:
    """Validate one completed generic experiment without rewriting its artifacts."""
    if not isinstance(experiment, Experiment):
        raise ValueError("experiment must be an Experiment")
    run = Path(run_dir)
    if not run.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run}")
    _reject_partial_state(run)
    _reject_failure_record(run)

    target_step = experiment.horizon if step is None else step
    _validate_target_step(experiment, target_step)
    names = _expected_names(expected_image_names, experiment.stage)

    config = _validate_config(
        run / "config.yaml",
        experiment,
        config_sha256,
    )
    del config
    _validate_provenance(run, manifest_sha256)
    summary = _load_summary(run / "summary.json", target_step)
    peak_gaussians, final_gaussians = _validate_metric_stream(
        run / "metrics.jsonl", target_step
    )
    if final_gaussians != summary["final_num_gaussians"]:
        raise ValueError("metrics final Gaussian count does not match summary")
    _validate_checkpoint_policy(
        run,
        experiment.stage,
        target_step,
        manifest_sha256,
        config_sha256,
    )

    report = None
    paired_ratio = None
    if experiment.stage is not ExperimentStage.PRODUCTION:
        if holdout_sha256 is None:
            raise ValueError("holdout_sha256 is required for internal-holdout stages")
        report_root = _report_root(run, experiment.stage, target_step)
        report = _validate_holdout_artifacts(
            report_root,
            experiment,
            target_step,
            manifest_sha256,
            config_sha256,
            holdout_sha256,
            names,
            summary,
            peak_gaussians,
            final_gaussians,
        )
        if b0_report is not None:
            paired_ratio = _paired_time_ratio(report, b0_report)
    else:
        if names:
            raise ValueError("production does not accept internal-holdout image names")
        _reject_production_holdout_artifacts(run)

    growth_controlled = _recorded_gate(
        "primitive_growth_controlled", summary, report
    )
    integrity_passed = _recorded_gate("integrity_passed", summary, report)
    return ArtifactValidationResult(
        experiment_report=report,
        paired_wall_time_ratio=paired_ratio,
        integrity_passed=integrity_passed,
        primitive_growth_controlled=growth_controlled,
    )


def append_failure(
    ledger_path: str | Path,
    *,
    experiment: Experiment,
    command_argv: Sequence[str],
    reason: str,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Atomically append one finite failure record while preserving prior JSONL."""
    if not isinstance(experiment, Experiment):
        raise ValueError("experiment must be an Experiment")
    if (
        isinstance(command_argv, (str, bytes))
        or not isinstance(command_argv, Sequence)
        or not command_argv
        or any(not isinstance(value, str) or not value for value in command_argv)
    ):
        raise ValueError("command_argv must be a non-empty sequence of strings")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a JSON object")

    record: dict[str, object] = {
        "schema_version": 1,
        "stage": experiment.stage.value,
        "scene_id": experiment.scene_id,
        "candidate_id": experiment.candidate_id,
        "command_argv": list(command_argv),
        "reason": reason,
        "provenance": dict(provenance),
    }
    try:
        canonical_json_sha256(record)
        line = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError("failure record must contain finite JSON values") from error

    output = Path(ledger_path)
    previous = b""
    if output.exists():
        if not output.is_file():
            raise ValueError("existing failure ledger is not a file")
        previous = output.read_bytes()
        _validate_existing_ledger(previous)
        if not previous.endswith(b"\n"):
            previous += b"\n"

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(previous)
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return record


def _validate_target_step(experiment: Experiment, step: int) -> None:
    if isinstance(step, bool) or not isinstance(step, int):
        raise ValueError("step must be an integer")
    if experiment.stage is ExperimentStage.CONFIRM:
        if step not in (15_000, 30_000):
            raise ValueError("confirm step must be 15000 or 30000")
    elif step != experiment.horizon:
        raise ValueError(f"{experiment.stage.value} step must be {experiment.horizon}")


def _expected_names(
    expected_image_names: Sequence[str], stage: ExperimentStage
) -> tuple[str, ...]:
    if isinstance(expected_image_names, (str, bytes)) or not isinstance(
        expected_image_names, Sequence
    ):
        raise ValueError("expected_image_names must be a sequence")
    names = tuple(expected_image_names)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("expected image names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("expected image names must be unique")
    if stage is not ExperimentStage.PRODUCTION and not names:
        raise ValueError("internal-holdout stages require expected image names")
    render_names = tuple(Path(name).with_suffix(".png").name for name in names)
    if len({name.casefold() for name in render_names}) != len(render_names):
        raise ValueError("expected validation render names collide")
    return names


def _validate_config(
    path: Path,
    experiment: Experiment,
    expected_sha256: str,
) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"required config does not exist: {path}")
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"config is invalid: {path}") from error
    if not isinstance(parsed, dict):
        raise ValueError("config must contain a mapping")
    config = dict(parsed)
    if compute_config_sha256(config) != expected_sha256:
        raise ValueError("config hash does not match expected config_sha256")
    expected = {
        "scene_id": experiment.scene_id,
        "candidate_id": experiment.candidate_id,
        "experiment_stage": experiment.stage.value,
        "max_steps": experiment.horizon,
        "internal_holdout": experiment.stage is not ExperimentStage.PRODUCTION,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"config {field} does not match experiment")
    return config


def _validate_provenance(run: Path, manifest_sha256: str) -> None:
    environment_path = run / "environment.json"
    if not environment_path.is_file():
        raise FileNotFoundError(
            f"required environment provenance does not exist: {environment_path}"
        )
    load_json_artifact(environment_path)
    hash_path = run / "manifest_hash.json"
    if not hash_path.is_file():
        raise FileNotFoundError(
            f"required manifest_hash provenance does not exist: {hash_path}"
        )
    if load_json_artifact(hash_path) != {"manifest_hash": manifest_sha256}:
        raise ValueError("manifest hash does not match expected manifest_sha256")


def _load_summary(path: Path, expected_step: int) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"required summary does not exist: {path}")
    summary = load_json_artifact(path)
    if summary.get("total_steps") != expected_step:
        raise ValueError("summary total_steps does not match expected step")
    for field in ("total_time_seconds", "final_loss", "max_vram_mb"):
        _finite_nonnegative(summary.get(field), f"summary {field}")
    validate_peak_vram_mb(float(summary["max_vram_mb"]))
    final_count = summary.get("final_num_gaussians")
    if isinstance(final_count, bool) or not isinstance(final_count, int) or final_count <= 0:
        raise ValueError("summary final_num_gaussians must be a positive integer")
    return summary


def _validate_metric_stream(path: Path, expected_step: int) -> tuple[int, int]:
    if not path.is_file():
        raise FileNotFoundError(f"required metrics do not exist: {path}")
    if expected_step == 30_000:
        _validate_30k_metrics(path)

    count = 0
    peak_gaussians = 0
    final_gaussians = 0
    with path.open("r", encoding="utf-8") as handle:
        for count, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"metrics contain a blank record at step {count}")
            try:
                record = json.loads(line, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"metrics must contain finite JSON at step {count}") from error
            if not isinstance(record, dict):
                raise ValueError(f"metrics record {count} must contain an object")
            if record.get("step") != count:
                raise ValueError(f"metrics are not ordered at step {count}")
            _finite_nonnegative(record.get("loss"), f"metrics non-finite loss at step {count}")
            gaussians = record.get("num_gaussians")
            if isinstance(gaussians, bool) or not isinstance(gaussians, int) or gaussians <= 0:
                raise ValueError(
                    f"metrics num_gaussians must be a positive integer at step {count}"
                )
            peak_gaussians = max(peak_gaussians, gaussians)
            final_gaussians = gaussians
    if count != expected_step:
        raise ValueError(f"metrics contain {count} records, expected {expected_step}")
    return peak_gaussians, final_gaussians


def _validate_checkpoint_policy(
    run: Path,
    stage: ExperimentStage,
    step: int,
    manifest_sha256: str,
    config_sha256: str,
) -> None:
    model_paths = {
        path.relative_to(run)
        for path in run.rglob("*")
        if path.is_file() and path.suffix.lower() in _MODEL_SUFFIXES
    }
    if stage in (ExperimentStage.REFERENCE, ExperimentStage.SCREEN):
        if model_paths:
            raise ValueError(f"{stage.value} must not contain model checkpoints")
        return
    if model_paths != {_RECOVERY_PATH}:
        raise ValueError(
            "confirm/production must contain only checkpoints/recovery.pt"
        )
    validate_recovery_checkpoint(
        run / _RECOVERY_PATH,
        manifest_sha256,
        config_sha256,
        step,
        require_precision_state=True,
    )


def _report_root(run: Path, stage: ExperimentStage, step: int) -> Path:
    if stage is ExperimentStage.CONFIRM and step == 15_000:
        return run / "snapshots" / "step_000015000"
    return run


def _validate_holdout_artifacts(
    root: Path,
    experiment: Experiment,
    step: int,
    manifest_sha256: str,
    config_sha256: str,
    holdout_sha256: str,
    expected_names: tuple[str, ...],
    summary: Mapping[str, object],
    peak_gaussians: int,
    final_gaussians: int,
) -> dict[str, object]:
    if not root.is_dir():
        raise FileNotFoundError(f"required report directory does not exist: {root}")
    records: dict[str, dict[str, object]] = {}
    for name in _HOLDOUT_REPORT_NAMES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"required {name} does not exist: {path}")
        records[name] = load_json_artifact(path)

    qualification = records["qualification_report.json"]
    detail = records["detail_metrics.json"]
    pose = records["pose_strata.json"]
    report = records["experiment_report.json"]
    expected_resources = {
        "total_time_seconds": float(summary["total_time_seconds"]),
        "max_vram_mb": float(summary["max_vram_mb"]),
        "peak_gaussians": peak_gaussians,
        "final_num_gaussians": final_gaussians,
    }
    _validate_qualification(
        qualification,
        experiment,
        step,
        config_sha256,
        holdout_sha256,
        expected_names,
        expected_resources,
    )
    _validate_render_names(root / "validation_renders", expected_names)
    _validate_report_identity(
        report,
        experiment,
        step,
        manifest_sha256,
        config_sha256,
        holdout_sha256,
    )
    resources = report.get("resources")
    if not isinstance(resources, Mapping):
        raise ValueError("experiment report resources must be a mapping")
    validate_peak_vram_mb(
        _finite_nonnegative(resources.get("max_vram_mb"), "peak_vram_mb")
    )
    for field, expected in expected_resources.items():
        actual = resources.get(field)
        if isinstance(expected, int):
            if actual != expected:
                raise ValueError(f"experiment report {field} does not match run artifacts")
        elif not math.isclose(
            _finite_nonnegative(actual, f"experiment report {field}"),
            expected,
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError(f"experiment report {field} does not match run artifacts")

    rebuilt = build_experiment_report(
        scene_id=experiment.scene_id,
        candidate_id=experiment.candidate_id,
        step=step,
        config_sha256=config_sha256,
        manifest_sha256=manifest_sha256,
        holdout_sha256=holdout_sha256,
        full_frame_report=qualification,
        detail_report=detail,
        pose_strata_report=pose,
        resource_summary=expected_resources,
    )
    unknown = set(report).difference(rebuilt, _OPTIONAL_REPORT_FIELDS)
    if unknown or any(report.get(key) != value for key, value in rebuilt.items()):
        raise ValueError("experiment report does not match validated source reports")
    return report


def _validate_qualification(
    report: Mapping[str, object],
    experiment: Experiment,
    step: int,
    config_sha256: str,
    holdout_sha256: str,
    expected_names: tuple[str, ...],
    expected_resources: Mapping[str, float | int],
) -> None:
    expected = {
        "schema_version": 1,
        "scene_id": experiment.scene_id,
        "candidate_id": experiment.candidate_id,
        "step": step,
        "config_sha256": config_sha256,
        "holdout_sha256": holdout_sha256,
        "image_count": len(expected_names),
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise ValueError(f"qualification report {field} does not match")
    images = report.get("images")
    if not isinstance(images, Mapping) or set(images) != set(expected_names):
        raise ValueError("qualification report image names do not match expected names")
    _finite_number(report.get("psnr_db_mean"), "qualification report psnr_db_mean")
    ssim = _finite_number(report.get("ssim_mean"), "qualification report ssim_mean")
    lpips = _finite_nonnegative(
        report.get("lpips_mean"), "qualification report lpips_mean"
    )
    valid_fraction = _finite_nonnegative(
        report.get("valid_fraction_mean"),
        "qualification report valid_fraction_mean",
    )
    if not -1.0 <= ssim <= 1.0:
        raise ValueError("qualification report ssim_mean is out of range")
    if lpips > 1.0:
        raise ValueError("qualification report lpips_mean is out of range")
    if valid_fraction > 1.0:
        raise ValueError("qualification report valid_fraction_mean is out of range")
    for field in ("total_time_seconds", "max_vram_mb"):
        actual = _finite_nonnegative(
            report.get(field), f"qualification report {field}"
        )
        if not math.isclose(
            actual,
            float(expected_resources[field]),
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ValueError(
                f"qualification report {field} does not match run artifacts"
            )
    peak = report.get("peak_gaussians")
    if isinstance(peak, bool) or not isinstance(peak, int) or peak <= 0:
        raise ValueError("qualification report peak_gaussians must be positive")
    if peak != expected_resources["peak_gaussians"]:
        raise ValueError(
            "qualification report peak_gaussians does not match run artifacts"
        )


def _validate_report_identity(
    report: Mapping[str, object],
    experiment: Experiment,
    step: int,
    manifest_sha256: str,
    config_sha256: str,
    holdout_sha256: str,
) -> None:
    expected = {
        "schema_version": 1,
        "scene_id": experiment.scene_id,
        "candidate_id": experiment.candidate_id,
        "step": step,
        "manifest_sha256": manifest_sha256,
        "config_sha256": config_sha256,
        "holdout_sha256": holdout_sha256,
    }
    for field, value in expected.items():
        if report.get(field) != value:
            raise ValueError(f"experiment report {field} does not match")


def _validate_render_names(path: Path, expected_image_names: tuple[str, ...]) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"required validation renders do not exist: {path}")
    expected = {Path(name).with_suffix(".png").name for name in expected_image_names}
    entries = tuple(path.iterdir())
    if any(not entry.is_file() for entry in entries):
        raise ValueError("validation render directory contains non-file entries")
    actual = {entry.name for entry in entries}
    if actual != expected:
        raise ValueError(
            "validation render filenames mismatch; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _paired_time_ratio(
    report: Mapping[str, object], b0_report: Mapping[str, object]
) -> float:
    if not isinstance(b0_report, Mapping):
        raise ValueError("b0_report must be a mapping")
    for field in ("scene_id", "step", "manifest_sha256", "holdout_sha256"):
        if b0_report.get(field) != report.get(field):
            raise ValueError(f"B0 report {field} does not match candidate report")
    if b0_report.get("candidate_id") != "B0-reference":
        raise ValueError("B0 report candidate_id must be B0-reference")
    current_resources = report.get("resources")
    b0_resources = b0_report.get("resources")
    if not isinstance(current_resources, Mapping) or not isinstance(b0_resources, Mapping):
        raise ValueError("paired reports must contain resource mappings")
    current_time = _finite_nonnegative(
        current_resources.get("total_time_seconds"), "candidate total_time_seconds"
    )
    b0_time = _finite_nonnegative(
        b0_resources.get("total_time_seconds"), "B0 total_time_seconds"
    )
    if b0_time <= 0.0:
        raise ValueError("B0 total_time_seconds must be positive")
    ratio = current_time / b0_time
    validate_paired_wall_time_ratio(ratio)
    return ratio


def _recorded_gate(
    field: str,
    summary: Mapping[str, object],
    report: Mapping[str, object] | None,
) -> bool:
    values = [record[field] for record in (summary, report) if record and field in record]
    if any(value is not True for value in values):
        label = field.replace("_", " ")
        raise ValueError(f"run records uncontrolled or invalid {label}")
    return True


def _reject_partial_state(run: Path) -> None:
    partial = next(
        (
            path
            for path in run.rglob("*")
            if path.name.endswith((".tmp", ".partial"))
        ),
        None,
    )
    if partial is not None:
        raise ValueError(f"run contains stale partial artifact: {partial}")


def _reject_failure_record(run: Path) -> None:
    path = run / "failure.json"
    if path.exists():
        if not path.is_file():
            raise ValueError("run failure record is not a file")
        load_json_artifact(path)
        raise ValueError("completed run contains a failure record")


def _reject_production_holdout_artifacts(run: Path) -> None:
    stale = [run / name for name in _HOLDOUT_REPORT_NAMES]
    stale.extend((run / "validation_renders", run / "snapshots"))
    present = next((path for path in stale if path.exists()), None)
    if present is not None:
        raise ValueError(
            f"production run contains stale internal-holdout artifact: {present}"
        )


def _finite_number(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be finite")
    return float(value)


def _finite_nonnegative(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{field} must be a finite nonnegative number")
    return float(value)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _validate_existing_ledger(payload: bytes) -> None:
    if not payload:
        raise ValueError("existing failure ledger is empty")
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise ValueError("existing failure ledger is not UTF-8") from error
    lines = text.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("existing failure ledger contains a blank record")
    for index, line in enumerate(lines, start=1):
        try:
            record = json.loads(line, parse_constant=_reject_json_constant)
            canonical_json_sha256(record)
            _validate_failure_record(record)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(
                f"existing failure ledger has malformed record {index}"
            ) from error


def _validate_failure_record(record: object) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("failure ledger record must be an object")
    if record.get("schema_version") != 1:
        raise ValueError("failure ledger record has an invalid schema_version")
    if record.get("stage") not in {stage.value for stage in ExperimentStage}:
        raise ValueError("failure ledger record has an invalid stage")
    for field in ("scene_id", "candidate_id", "reason"):
        value = record.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"failure ledger record has an invalid {field}")
    command = record.get("command_argv")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(value, str) or not value for value in command)
    ):
        raise ValueError("failure ledger record has invalid command_argv")
    if not isinstance(record.get("provenance"), Mapping):
        raise ValueError("failure ledger record has invalid provenance")
