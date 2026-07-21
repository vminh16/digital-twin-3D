from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Callable

from bts_nvs.training.c1_phase_c import (
    LOCKED_CANDIDATE,
    PHASE_C_SCENE,
    _final_validation,
    _is_sha256,
    build_phase_c_command,
    build_phase_c_decision,
    classify_run_directory,
    compute_phase_c_config_sha256,
    validate_phase_b_authorization,
    validate_phase_c_baseline,
    validate_phase_c_baseline_artifacts,
    validate_phase_c_candidate,
    validate_phase_c_pair,
)


def _load_json(path: Path) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"required JSON artifact does not exist: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {source}")
    return payload


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run_phase_c(
    *,
    repo_root: Path,
    scenes_root: Path,
    manifests_root: Path,
    backend_root: Path,
    baseline_root: Path,
    phase_b_root: Path,
    output_root: Path,
    python_bin: str,
    run_process: Callable[..., object] = subprocess.run,
    load_backend_fn: Callable[[Path], object] | None = None,
    hash_renders_fn: Callable[[Path, tuple[str, ...]], dict[str, str]] | None = None,
    diagnostics_fn: Callable[..., dict] | None = None,
    validate_recovery_fn: Callable[[Path, str, str, int], None] | None = None,
    compute_manifest_hash_fn: Callable[[Path], str] | None = None,
    validate_baseline_artifacts_fn: Callable[[object, dict[str, str]], None]
    | None = None,
) -> dict:
    import yaml

    output = Path(output_root)
    ledger_path = output / "phase_c_ledger.json"
    ledger = {
        "schema_version": 1,
        "phase": "C1-phase-C",
        "scene_id": PHASE_C_SCENE,
        "candidate_id": LOCKED_CANDIDATE,
        "status": "pending",
        "error_type": None,
        "error_message": None,
    }
    _atomic_json(ledger_path, ledger)
    try:
        if load_backend_fn is None:
            from bts_nvs.training.full_training import (
                load_or_create_backend_decision,
            )

            load_backend_fn = load_or_create_backend_decision
        if hash_renders_fn is None:
            from bts_nvs.training.qualification import hash_validation_renders

            hash_renders_fn = hash_validation_renders
        if diagnostics_fn is None:
            from bts_nvs.training.c1_screening import diagnostics_for_run

            diagnostics_fn = diagnostics_for_run
        if validate_recovery_fn is None:
            from bts_nvs.training.run_training import validate_recovery_checkpoint

            validate_recovery_fn = validate_recovery_checkpoint
        if compute_manifest_hash_fn is None:
            from bts_nvs.training.trainer import compute_manifest_sha256

            compute_manifest_hash_fn = compute_manifest_sha256
        if validate_baseline_artifacts_fn is None:
            validate_baseline_artifacts_fn = validate_phase_c_baseline_artifacts

        phase_b = _load_json(Path(phase_b_root) / "phase_b_decision.json")
        validate_phase_b_authorization(phase_b)
        backend = load_backend_fn(Path(backend_root))
        ledger.update(
            {
                "optimizer_backend": getattr(backend, "optimizer_backend", None),
                "precision": getattr(backend, "precision", None),
                "backend_report_sha256": getattr(backend, "report_sha256", None),
            }
        )

        baseline_report_path = Path(baseline_root) / "full_length_report.json"
        baseline_config_path = Path(baseline_root) / "config.yaml"
        baseline_report = _load_json(baseline_report_path)
        baseline_config = yaml.safe_load(
            baseline_config_path.read_text(encoding="utf-8")
        )
        if not isinstance(baseline_config, dict):
            raise ValueError("Phase C baseline config must contain a mapping")
        validate_phase_c_baseline(baseline_report, baseline_config, backend)
        baseline_manifest_record = _load_json(
            Path(baseline_root) / "manifest_hash.json"
        )
        baseline_manifest_hash = baseline_manifest_record.get("manifest_hash")

        scene_root = Path(scenes_root) / PHASE_C_SCENE
        manifest_root = Path(manifests_root) / PHASE_C_SCENE
        required = (
            scene_root,
            manifest_root / "manifest.json",
            manifest_root / "holdout.json",
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "missing Phase C input:\n" + "\n".join(f"- {path}" for path in missing)
            )
        current_manifest_hash = compute_manifest_hash_fn(
            manifest_root / "manifest.json"
        )
        if current_manifest_hash != baseline_manifest_hash:
            raise ValueError(
                "Phase C input manifest does not match the pinned B0 manifest"
            )
        baseline_validation = _final_validation(baseline_report)
        baseline_render_hashes = hash_renders_fn(
            Path(baseline_root) / "validation_renders",
            tuple(baseline_validation["images"]),
        )
        validate_baseline_artifacts_fn(
            baseline_manifest_hash,
            baseline_render_hashes,
        )

        diagnostic_args = {
            "scene_id": PHASE_C_SCENE,
            "scene_root": scene_root,
            "manifest_root": manifest_root,
        }
        baseline_diagnostic = diagnostics_fn(
            candidate_id="B0-reference",
            render_dir=Path(baseline_root) / "validation_renders",
            **diagnostic_args,
        )
        _atomic_json(output / "baseline_high_frequency.json", baseline_diagnostic)

        run_dir = output / PHASE_C_SCENE
        state = classify_run_directory(run_dir)
        ledger["status"] = "resuming" if state == "resume" else "running"
        if state == "complete":
            ledger["status"] = "validating"
        _atomic_json(ledger_path, ledger)
        if state != "complete":
            recovery = (
                run_dir / "checkpoints" / "recovery.pt"
                if state == "resume"
                else None
            )
            command = build_phase_c_command(
                repo_root=repo_root,
                scenes_root=scenes_root,
                manifests_root=manifests_root,
                output_root=output,
                decision=backend,
                python_bin=python_bin,
                resume_path=recovery,
            )
            result = run_process(command, check=False)
            returncode = getattr(result, "returncode", None)
            if returncode != 0:
                raise RuntimeError(f"Phase C training exited with code {returncode}")

        candidate_report_path = run_dir / "full_length_report.json"
        candidate_config_path = run_dir / "config.yaml"
        candidate_report = _load_json(candidate_report_path)
        candidate_config = yaml.safe_load(
            candidate_config_path.read_text(encoding="utf-8")
        )
        if not isinstance(candidate_config, dict):
            raise ValueError("Phase C candidate config must contain a mapping")
        validate_phase_c_pair(
            baseline_report, candidate_report, baseline_config, candidate_config
        )
        if (
            candidate_config.get("optimizer_backend")
            != getattr(backend, "optimizer_backend", None)
            or candidate_config.get("precision")
            != getattr(backend, "precision", None)
        ):
            raise ValueError("Phase C candidate does not match the accepted backend")
        manifest_record = _load_json(run_dir / "manifest_hash.json")
        manifest_hash = manifest_record.get("manifest_hash")
        if not _is_sha256(manifest_hash):
            raise ValueError("Phase C manifest hash record is invalid")
        config_hash = compute_phase_c_config_sha256(candidate_config)
        validate_recovery_fn(
            run_dir / "checkpoints" / "recovery.pt",
            manifest_hash,
            config_hash,
            30_000,
        )
        candidate_render_hashes = hash_renders_fn(
            run_dir / "validation_renders",
            tuple(candidate_report["final_validation"]["images"]),
        )
        validate_phase_c_candidate(
            candidate_report,
            candidate_config,
            manifest_hash,
            candidate_render_hashes,
        )

        candidate_diagnostic = diagnostics_fn(
            candidate_id=LOCKED_CANDIDATE,
            render_dir=run_dir / "validation_renders",
            **diagnostic_args,
        )
        _atomic_json(run_dir / "high_frequency.json", candidate_diagnostic)
        provenance = {
            "baseline_root": str(Path(baseline_root).resolve()),
            "candidate_root": str(run_dir.resolve()),
            "baseline_config_sha256": _sha256(baseline_config_path),
            "baseline_report_sha256": _sha256(baseline_report_path),
            "candidate_config_sha256": config_hash,
            "baseline_manifest_sha256": baseline_manifest_hash,
            "candidate_manifest_sha256": manifest_hash,
            "candidate_report_sha256": _sha256(candidate_report_path),
            "backend_report_sha256": getattr(backend, "report_sha256", None),
            "baseline_render_sha256": baseline_render_hashes,
            "candidate_render_sha256": candidate_render_hashes,
        }
        decision = build_phase_c_decision(
            baseline_report,
            candidate_report,
            baseline_diagnostic,
            candidate_diagnostic,
            integrity_passed=True,
            provenance=provenance,
        )
        _atomic_json(output / "phase_c_decision.json", decision)
        ledger["status"] = "passed" if decision["phase_c_passed"] else "rejected"
        ledger["failed_gates"] = decision["failed_gates"]
        _atomic_json(ledger_path, ledger)
        return decision
    except Exception as error:
        ledger.update(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        _atomic_json(ledger_path, ledger)
        raise
