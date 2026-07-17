from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml
from PIL import Image

from bts_nvs.training.backend_qualification import (
    compare_backend_profiles,
    load_backend_profile,
    write_backend_comparison,
)
from bts_nvs.training.checkpoint import load_checkpoint
from bts_nvs.training.trainer import compute_config_sha256, compute_manifest_sha256


CANONICAL_SCENES = (
    "hcm0031",
    "hcm0034",
    "HCM0181",
    "HCM0193",
    "HCM0204",
    "HCM0249",
    "HCM0254",
    "HCM0276",
    "HCM0421",
    "HCM0539",
    "HCM0540",
    "HCM0644",
    "HCM0674",
    "HCM1439",
    "HNI0131",
    "HNI0265",
    "HNI0366",
    "HNI0437",
)


@dataclass(frozen=True)
class BackendDecision:
    optimizer_backend: str
    precision: str
    report_sha256: str


@dataclass(frozen=True)
class TrainedRun:
    completed_step: int
    config_sha256: str
    manifest_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_or_create_backend_decision(backend_root: Path) -> BackendDecision:
    root = Path(backend_root)
    expected = compare_backend_profiles(
        load_backend_profile(root / "reference" / "backend_profile.json"),
        load_backend_profile(root / "fused" / "backend_profile.json"),
        load_backend_profile(root / "amp" / "backend_profile.json"),
    )
    report_path = root / "backend_qualification.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report != expected:
            raise ValueError("backend qualification report does not match backend profiles")
    else:
        write_backend_comparison(report_path, expected)
        report = expected
    if report.get("accepted") is not True:
        raise ValueError("backend qualification report was not accepted")
    pair = (
        report.get("selected_optimizer_backend"),
        report.get("selected_precision"),
    )
    if pair not in {
        ("adam", "fp32"),
        ("adam-fused", "fp32"),
        ("adam-fused", "amp-fp16"),
    }:
        raise ValueError("backend qualification selected an unsupported backend")
    return BackendDecision(pair[0], pair[1], _sha256(report_path))


def _directory_names(root: Path) -> set[str]:
    if not Path(root).is_dir():
        raise FileNotFoundError(f"required directory does not exist: {root}")
    return {path.name for path in Path(root).iterdir() if path.is_dir()}


def validate_scene_pool(
    scenes_root: Path,
    manifests_root: Path,
) -> tuple[str, ...]:
    expected = set(CANONICAL_SCENES)
    scene_names = _directory_names(Path(scenes_root))
    manifest_names = _directory_names(Path(manifests_root))
    if scene_names != expected:
        raise ValueError("scene root does not match the canonical 18-scene pool")
    if manifest_names != expected:
        raise ValueError("manifest root does not match the canonical 18-scene pool")

    for scene_id in CANONICAL_SCENES:
        manifest_path = Path(manifests_root) / scene_id / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("scene_id") != scene_id:
            raise ValueError(f"manifest scene identity mismatch: {manifest_path}")
        arrays_name = payload.get("arrays_file")
        if (
            not isinstance(arrays_name, str)
            or Path(arrays_name).is_absolute()
            or Path(arrays_name).name != arrays_name
        ):
            raise ValueError(f"invalid manifest arrays_file: {manifest_path}")
        arrays_path = manifest_path.parent / arrays_name
        if not arrays_path.is_file():
            raise FileNotFoundError(f"manifest arrays do not exist: {arrays_path}")
    return CANONICAL_SCENES


_STATUSES = {"pending", "running", "trained", "failed"}


def _atomic_json(path: Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, output)


def _new_ledger(
    decision: BackendDecision,
    scene_ids: tuple[str, ...],
) -> dict:
    return {
        "schema_version": 1,
        "optimizer_backend": decision.optimizer_backend,
        "precision": decision.precision,
        "qualification_report_sha256": decision.report_sha256,
        "scenes": [
            {
                "scene_id": scene_id,
                "status": "pending",
                "run_dir": f"scenes/{scene_id}",
                "completed_step": 0,
                "config_sha256": None,
                "manifest_sha256": None,
                "error_type": None,
                "error_message": None,
            }
            for scene_id in scene_ids
        ],
    }


def _validate_ledger_identity(
    ledger: dict,
    decision: BackendDecision,
    scene_ids: tuple[str, ...],
) -> None:
    identity_matches = (
        ledger.get("schema_version") == 1
        and ledger.get("optimizer_backend") == decision.optimizer_backend
        and ledger.get("precision") == decision.precision
        and ledger.get("qualification_report_sha256") == decision.report_sha256
        and [record.get("scene_id") for record in ledger.get("scenes", [])]
        == list(scene_ids)
    )
    if not identity_matches:
        raise ValueError("existing full-training ledger identity does not match")
    for record in ledger["scenes"]:
        if record.get("status") not in _STATUSES:
            raise ValueError("ledger contains an invalid scene status")


def load_or_create_ledger(
    output_root: Path,
    decision: BackendDecision,
    scene_ids: tuple[str, ...],
) -> dict:
    path = Path(output_root) / "ledger.json"
    if path.is_file():
        ledger = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(ledger, dict):
            raise ValueError("full-training ledger must contain an object")
        _validate_ledger_identity(ledger, decision, scene_ids)
        return ledger
    ledger = _new_ledger(decision, scene_ids)
    _atomic_json(path, ledger)
    return ledger


def set_scene_status(
    ledger_path: Path,
    ledger: dict,
    scene_id: str,
    status: str,
    *,
    completed_step: int = 0,
    config_sha256: str | None = None,
    manifest_sha256: str | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> None:
    if status not in _STATUSES:
        raise ValueError("invalid full-training scene status")
    record = next(
        (item for item in ledger.get("scenes", []) if item.get("scene_id") == scene_id),
        None,
    )
    if record is None:
        raise ValueError(f"scene is not present in ledger: {scene_id}")
    record.update(
        {
            "status": status,
            "completed_step": completed_step,
            "config_sha256": config_sha256,
            "manifest_sha256": manifest_sha256,
            "error_type": error_type,
            "error_message": error_message,
        }
    )
    _atomic_json(Path(ledger_path), ledger)


def _load_locked_config(
    path: Path,
    scene_id: str,
    decision: BackendDecision,
) -> dict:
    if not Path(path).is_file():
        raise FileNotFoundError(f"training config does not exist: {path}")
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {
        "scene_id": scene_id,
        "resize_factor": 1,
        "max_steps": 30_000,
        "seed": 0,
        "cache_images": True,
        "pinned_transfer": True,
        "optimizer_backend": decision.optimizer_backend,
        "precision": decision.precision,
        "rolling_checkpoint": True,
        "internal_holdout": False,
    }
    if not isinstance(config, dict) or any(
        config.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"training config violates the production contract: {path}")
    return config


def _validate_metrics(path: Path) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"metrics do not exist: {path}")
    count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for count, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid metrics JSON at step {count}") from error
            loss = record.get("loss")
            if record.get("step") != count:
                raise ValueError(f"metrics are not ordered at step {count}")
            if (
                isinstance(loss, bool)
                or not isinstance(loss, (int, float))
                or not math.isfinite(float(loss))
            ):
                raise ValueError(f"metrics contain non-finite loss at step {count}")
    if count != 30_000:
        raise ValueError(f"metrics contain {count} records, expected 30000")


def _current_run_hashes(config: dict, manifest_path: Path) -> tuple[str, str]:
    return compute_config_sha256(config), compute_manifest_sha256(manifest_path)


def _validate_recovery(
    run_dir: Path,
    scene_id: str,
    manifest_path: Path,
    decision: BackendDecision,
) -> tuple[dict, dict, str, str]:
    config = _load_locked_config(run_dir / "config.yaml", scene_id, decision)
    config_hash, manifest_hash = _current_run_hashes(config, manifest_path)
    recorded_hash = json.loads(
        (run_dir / "manifest_hash.json").read_text(encoding="utf-8")
    )
    if recorded_hash != {"manifest_hash": manifest_hash}:
        raise ValueError("run manifest hash does not match the current manifest")
    checkpoint = load_checkpoint(
        run_dir / "checkpoints" / "recovery.pt",
        expected_manifest_hash=manifest_hash,
        expected_config_hash=config_hash,
    )
    step = checkpoint.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or not 0 < step <= 30_000:
        raise ValueError("recovery checkpoint has an invalid step")
    return config, checkpoint, config_hash, manifest_hash


def validate_trained_run(
    run_dir: Path,
    scene_id: str,
    manifest_path: Path,
    decision: BackendDecision,
) -> TrainedRun:
    run = Path(run_dir)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if summary.get("total_steps") != 30_000:
        raise ValueError("training summary did not reach 30000 steps")
    convergence = json.loads(
        (run / "convergence.json").read_text(encoding="utf-8")
    )
    if convergence.get("final_render_non_blank") is not True:
        raise ValueError("final train preview is blank")
    _validate_metrics(run / "metrics.jsonl")
    _, checkpoint, config_hash, manifest_hash = _validate_recovery(
        run, scene_id, manifest_path, decision
    )
    if checkpoint["step"] != 30_000:
        raise ValueError("final recovery checkpoint is not at step 30000")
    preview_path = run / "train_previews" / "step_000030000.png"
    with Image.open(preview_path) as image:
        image.load()
        if image.mode != "RGB" or image.width <= 0 or image.height <= 0:
            raise ValueError("final train preview must be non-empty RGB")
    return TrainedRun(30_000, config_hash, manifest_hash)


def inspect_scene_run(
    run_dir: Path,
    scene_id: str,
    manifest_path: Path,
    decision: BackendDecision,
) -> str:
    run = Path(run_dir)
    if not run.exists() or (run.is_dir() and not any(run.iterdir())):
        return "fresh"
    if not run.is_dir():
        raise ValueError(f"scene run path is not a directory: {run}")
    try:
        validate_trained_run(run, scene_id, manifest_path, decision)
        return "trained"
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        recovery = run / "checkpoints" / "recovery.pt"
        if recovery.is_file():
            _validate_recovery(run, scene_id, manifest_path, decision)
            return "resume"
        raise ValueError(
            f"scene run directory is non-empty without valid recovery: {run}"
        )


def build_scene_command(
    *,
    repo_root: Path,
    scenes_root: Path,
    manifests_root: Path,
    output_root: Path,
    scene_id: str,
    decision: BackendDecision,
    python_bin: str,
    resume_path: Path | None = None,
) -> list[str]:
    command = [
        python_bin,
        str(Path(repo_root) / "src" / "bts_nvs" / "training" / "run_training.py"),
        "--scene_dir",
        str(Path(scenes_root) / scene_id),
        "--manifest_dir",
        str(Path(manifests_root) / scene_id),
        "--output_dir",
        str(Path(output_root) / "scenes" / scene_id),
        "--resize_factor",
        "1",
        "--max_steps",
        "30000",
        "--checkpoint_every",
        "3000",
        "--seed",
        "0",
        "--cache_images",
        "--pinned_transfer",
        "--optimizer_backend",
        decision.optimizer_backend,
        "--precision",
        decision.precision,
        "--rolling_checkpoint",
    ]
    if resume_path is not None:
        command.extend(("--resume", str(resume_path)))
    return command


def run_full_training(
    *,
    repo_root: Path,
    scenes_root: Path,
    manifests_root: Path,
    backend_root: Path,
    output_root: Path,
    python_bin: str,
    run_process: Callable[..., object] = subprocess.run,
) -> None:
    decision = load_or_create_backend_decision(backend_root)
    scene_ids = validate_scene_pool(scenes_root, manifests_root)
    ledger = load_or_create_ledger(output_root, decision, scene_ids)
    ledger_path = Path(output_root) / "ledger.json"

    for scene_id in scene_ids:
        run_dir = Path(output_root) / "scenes" / scene_id
        manifest_path = Path(manifests_root) / scene_id / "manifest.json"
        try:
            state = inspect_scene_run(run_dir, scene_id, manifest_path, decision)
        except (FileNotFoundError, OSError, ValueError) as error:
            set_scene_status(
                ledger_path,
                ledger,
                scene_id,
                "failed",
                error_type=type(error).__name__,
                error_message=str(error),
            )
            raise
        if state == "trained":
            trained = validate_trained_run(
                run_dir, scene_id, manifest_path, decision
            )
            set_scene_status(
                ledger_path,
                ledger,
                scene_id,
                "trained",
                completed_step=trained.completed_step,
                config_sha256=trained.config_sha256,
                manifest_sha256=trained.manifest_sha256,
            )
            continue

        set_scene_status(ledger_path, ledger, scene_id, "running")
        resume_path = (
            run_dir / "checkpoints" / "recovery.pt" if state == "resume" else None
        )
        command = build_scene_command(
            repo_root=repo_root,
            scenes_root=scenes_root,
            manifests_root=manifests_root,
            output_root=output_root,
            scene_id=scene_id,
            decision=decision,
            python_bin=python_bin,
            resume_path=resume_path,
        )
        try:
            result = run_process(command, check=False)
        except OSError as error:
            set_scene_status(
                ledger_path,
                ledger,
                scene_id,
                "failed",
                error_type=type(error).__name__,
                error_message=str(error),
            )
            raise

        returncode = getattr(result, "returncode", None)
        if returncode != 0:
            message = f"training process exited with code {returncode}"
            set_scene_status(
                ledger_path,
                ledger,
                scene_id,
                "failed",
                error_type="ProcessExit",
                error_message=message,
            )
            raise RuntimeError(f"full training failed for {scene_id}: {message}")

        try:
            trained = validate_trained_run(
                run_dir, scene_id, manifest_path, decision
            )
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
            set_scene_status(
                ledger_path,
                ledger,
                scene_id,
                "failed",
                error_type=type(error).__name__,
                error_message=str(error),
            )
            raise
        set_scene_status(
            ledger_path,
            ledger,
            scene_id,
            "trained",
            completed_step=trained.completed_step,
            config_sha256=trained.config_sha256,
            manifest_sha256=trained.manifest_sha256,
        )
