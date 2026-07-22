from __future__ import annotations

import sys
from pathlib import Path

from bts_nvs.experiments.candidates import candidate_settings
from bts_nvs.experiments.experiment import Experiment, ExperimentStage


_OPTIMIZER_BACKENDS = frozenset(("adam", "adam-fused"))
_PRECISIONS = frozenset(("fp32", "amp-fp16"))
_TRAINING_SCRIPT = Path("src") / "bts_nvs" / "training" / "run_training.py"
_RECOVERY_PATH = Path("checkpoints") / "recovery.pt"


def build_training_command(
    *,
    repo_root: str | Path,
    scene_dir: str | Path,
    manifest_dir: str | Path,
    output_dir: str | Path,
    optimizer_backend: str,
    precision: str,
    experiment: Experiment,
    stop_step: int,
    resume_path: str | Path | None = None,
) -> list[str]:
    """Build one fair generic-training argument vector without launching it."""
    if not isinstance(experiment, Experiment):
        raise ValueError("experiment must be an Experiment")
    _validate_runtime(optimizer_backend, precision)
    _validate_stop_step(stop_step)

    output = Path(output_dir)
    resume = None if resume_path is None else Path(resume_path)
    _validate_stage_run(experiment.stage, stop_step, resume, output)
    candidate_id = candidate_settings(experiment.candidate_id).candidate_id

    command = [
        sys.executable,
        str(Path(repo_root) / _TRAINING_SCRIPT),
        "--scene_dir",
        str(scene_dir),
        "--manifest_dir",
        str(manifest_dir),
        "--output_dir",
        str(output),
        "--seed",
        "0",
        "--resize_factor",
        "1",
        "--max_steps",
        str(experiment.horizon),
        "--stop_step",
        str(stop_step),
        "--candidate_id",
        candidate_id,
        "--experiment_stage",
        experiment.stage.value,
        "--cache_images",
        "--pinned_transfer",
    ]
    if experiment.stage is not ExperimentStage.PRODUCTION:
        command.append("--internal_holdout")
    if experiment.stage in (ExperimentStage.CONFIRM, ExperimentStage.PRODUCTION):
        command.extend(
            (
                "--authorized_candidate_id",
                candidate_id,
                "--checkpoint_every",
                "3000",
                "--rolling_checkpoint",
            )
        )
    if resume is not None:
        command.extend(("--resume", str(resume)))
    command.extend(
        (
            "--optimizer_backend",
            optimizer_backend,
            "--precision",
            precision,
        )
    )
    return command


def _validate_runtime(optimizer_backend: str, precision: str) -> None:
    if optimizer_backend not in _OPTIMIZER_BACKENDS:
        raise ValueError("optimizer_backend is unsupported")
    if precision not in _PRECISIONS:
        raise ValueError("precision is unsupported")
    if precision == "amp-fp16" and optimizer_backend != "adam-fused":
        raise ValueError("amp-fp16 requires adam-fused")


def _validate_stop_step(stop_step: int) -> None:
    if isinstance(stop_step, bool) or not isinstance(stop_step, int):
        raise ValueError("stop_step must be an integer")


def _validate_stage_run(
    stage: ExperimentStage,
    stop_step: int,
    resume_path: Path | None,
    output_dir: Path,
) -> None:
    if stage in (ExperimentStage.REFERENCE, ExperimentStage.SCREEN):
        if stop_step != 7_000 or resume_path is not None:
            raise ValueError(f"{stage.value} requires a fresh 7000-step run")
        return

    expected_recovery = output_dir / _RECOVERY_PATH
    if resume_path is not None and resume_path != expected_recovery:
        raise ValueError("resume_path must be output_dir/checkpoints/recovery.pt")

    if stage is ExperimentStage.CONFIRM:
        if stop_step not in (15_000, 30_000):
            raise ValueError("confirm requires stop_step 15000 or 30000")
        if resume_path is not None and stop_step != 30_000:
            raise ValueError("confirm resume requires stop_step 30000")
        return

    if stop_step != 30_000:
        raise ValueError("production requires stop_step 30000")
