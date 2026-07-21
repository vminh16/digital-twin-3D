
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


LOCKED_CANDIDATE = "C1-absgrad-t08-revopacity-v1"
PHASE_C_SCENE = "HCM0181"
MAX_VRAM_MB = 23 * 1024
LEGACY_B0_GIT_COMMIT = "aa55a21704f6d40375662d669dbf52ec1684a528"
LEGACY_B0_HOLDOUT_SHA256 = (
    "0d39a9f2ec144f06640470cc3fca10cf355b37c9c04c619ec756991d2d36aec5"
)
LEGACY_B0_MANIFEST_SHA256 = (
    "24d02f63ff6530c152a8f7a9dc09a01ee63d10debb8e61638aa7512d65a01ea6"
)
LEGACY_B0_RENDER_SET_SHA256 = (
    "d557dfbf740f3215f06dbc8b4d6a5a96e3ab1e3a597e19a44d121395094e26c6"
)
LEGACY_B0_CONFIG = {
    "cache_images": True,
    "data_range": 1.0,
    "full_length_qualification": True,
    "grow_grad2d": 0.0002,
    "grow_scale3d": 0.01,
    "guard_count": 46,
    "holdout_algorithm": "pose_fps_guard2_v1",
    "holdout_manifest_sha256": LEGACY_B0_HOLDOUT_SHA256,
    "internal_holdout": True,
    "internal_train_count": 169,
    "lambda_dssim": 0.2,
    "max_steps": 30_000,
    "pinned_transfer": True,
    "profile_input": False,
    "prune_opa": 0.005,
    "refine_every": 100,
    "refine_start_step": 500,
    "refine_stop_step": 15_000,
    "reset_every": 3_000,
    "resize_factor": 1,
    "resize_height": 989,
    "resize_width": 1320,
    "scene_id": PHASE_C_SCENE,
    "seed": 0,
    "undistort": True,
    "validation_count": 25,
}


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _score50(metrics: dict) -> float:
    return (
        40.0
        - 40.0 * _finite(metrics.get("lpips_mean"), "lpips_mean")
        + 30.0 * _finite(metrics.get("ssim_mean"), "ssim_mean")
        + 0.6 * _finite(metrics.get("psnr_db_mean"), "psnr_db_mean")
    )


def _final_validation(report: dict) -> dict:
    if (
        report.get("schema_version") != 1
        or report.get("scene_id") != PHASE_C_SCENE
        or report.get("step") != 30_000
    ):
        raise ValueError("full-length report identity mismatch")
    validation = report.get("final_validation")
    if not isinstance(validation, dict):
        raise ValueError("full-length report has no final validation")
    _score50(validation)
    images = validation.get("images")
    if (
        not isinstance(images, dict)
        or not images
        or validation.get("image_count") != len(images)
    ):
        raise ValueError("full-length validation image set is incomplete")
    for image_name, metrics in images.items():
        if not isinstance(image_name, str) or not isinstance(metrics, dict):
            raise ValueError("full-length per-image metrics are invalid")
        for key in ("psnr_db", "ssim", "lpips"):
            _finite(metrics.get(key), f"{image_name} {key}")
    return validation


def compute_phase_c_config_sha256(config: dict) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_render_set_sha256(render_hashes: dict[str, str]) -> str:
    payload = json.dumps(
        render_hashes,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def build_phase_c_command(
    *,
    repo_root: Path,
    scenes_root: Path,
    manifests_root: Path,
    output_root: Path,
    decision: object,
    python_bin: str,
    resume_path: Path | None = None,
) -> list[str]:
    command = [
        python_bin,
        str(Path(repo_root) / "src" / "bts_nvs" / "training" / "run_training.py"),
        "--scene_dir",
        str(Path(scenes_root) / PHASE_C_SCENE),
        "--manifest_dir",
        str(Path(manifests_root) / PHASE_C_SCENE),
        "--output_dir",
        str(Path(output_root) / PHASE_C_SCENE),
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
        str(getattr(decision, "optimizer_backend")),
        "--precision",
        str(getattr(decision, "precision")),
        "--full_length_candidate",
        LOCKED_CANDIDATE,
    ]
    if resume_path is not None:
        expected = Path(output_root) / PHASE_C_SCENE / "checkpoints" / "recovery.pt"
        if Path(resume_path).resolve() != expected.resolve():
            raise ValueError("Phase C resume must use its colocated recovery.pt")
        command.extend(("--resume", str(resume_path)))
    return command


def classify_run_directory(run_dir: Path) -> str:
    run = Path(run_dir)
    if not run.exists() or (run.is_dir() and not any(run.iterdir())):
        return "fresh"
    if not run.is_dir():
        raise ValueError(f"Phase C run path is not a directory: {run}")
    report = run / "full_length_report.json"
    recovery = run / "checkpoints" / "recovery.pt"
    if report.is_file():
        if not recovery.is_file():
            raise ValueError("complete Phase C run has no rolling recovery checkpoint")
        return "complete"
    if recovery.is_file():
        return "resume"
    raise ValueError(f"Phase C run directory is non-empty without recovery: {run}")


def validate_phase_b_authorization(decision: dict) -> None:
    if (
        decision.get("phase_b_passed") is not True
        or decision.get("selected_candidate") != LOCKED_CANDIDATE
    ):
        raise ValueError("Phase B did not authorize the locked Phase C candidate")


def validate_phase_c_baseline(
    baseline_report: dict, baseline_config: dict, backend: object
) -> None:
    validation = _final_validation(baseline_report)
    if baseline_config != LEGACY_B0_CONFIG:
        raise ValueError("Phase C baseline is not the pinned legacy B0 config")
    if baseline_report.get("git_commit") != LEGACY_B0_GIT_COMMIT:
        raise ValueError("Phase C baseline source commit is not pinned")
    expected = {
        "psnr_db_mean": 22.691555943105403,
        "ssim_mean": 0.8053435290993863,
        "lpips_mean": 0.11125879257917404,
    }
    if any(
        not math.isclose(
            _finite(validation.get(key), f"baseline {key}"),
            value,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for key, value in expected.items()
    ):
        raise ValueError("Phase C baseline metrics are not the pinned B0 artifact")
    if (
        validation.get("image_count") != 25
        or baseline_report.get("peak_gaussians") != 6_861_805
        or not math.isclose(
            _finite(baseline_report.get("max_vram_mb"), "baseline max_vram_mb"),
            11_789.9814453125,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("Phase C baseline resource identity is not pinned")
    if (
        getattr(backend, "optimizer_backend", None) != "adam"
        or getattr(backend, "precision", None) != "fp32"
    ):
        raise ValueError(
            "historical B0 requires the accepted backend to remain legacy Adam/FP32"
        )


def validate_phase_c_baseline_artifacts(
    manifest_hash: object,
    render_hashes: dict[str, str],
) -> None:
    if manifest_hash != LEGACY_B0_MANIFEST_SHA256:
        raise ValueError("Phase C baseline manifest is not the pinned B0 artifact")
    if (
        len(render_hashes) != 25
        or any(not _is_sha256(value) for value in render_hashes.values())
        or compute_render_set_sha256(render_hashes)
        != LEGACY_B0_RENDER_SET_SHA256
    ):
        raise ValueError("Phase C baseline renders are not the pinned B0 artifact")


def validate_phase_c_pair(
    baseline_report: dict,
    candidate_report: dict,
    baseline_config: dict,
    candidate_config: dict,
) -> None:
    baseline_validation = _final_validation(baseline_report)
    candidate_validation = _final_validation(candidate_report)
    if baseline_config.get("holdout_manifest_sha256") != candidate_config.get(
        "holdout_manifest_sha256"
    ):
        raise ValueError("Phase C baseline and candidate holdout hashes differ")
    if set(baseline_validation.get("images", {})) != set(
        candidate_validation.get("images", {})
    ):
        raise ValueError("Phase C baseline and candidate validation images differ")
    expected_candidate = {
        "scene_id": PHASE_C_SCENE,
        "resize_factor": 1,
        "max_steps": 30_000,
        "seed": 0,
        "cache_images": True,
        "pinned_transfer": True,
        "internal_holdout": True,
        "rolling_checkpoint": True,
        "full_length_candidate": LOCKED_CANDIDATE,
        "grow_grad2d": 0.0008,
        "absgrad": True,
        "revised_opacity": True,
    }
    if any(
        candidate_config.get(key) != value
        for key, value in expected_candidate.items()
    ):
        raise ValueError("Phase C candidate config violates the locked contract")


def validate_phase_c_candidate(
    candidate_report: dict,
    candidate_config: dict,
    manifest_hash: str,
    render_hashes: dict[str, str],
) -> None:
    validation = _final_validation(candidate_report)
    if candidate_report.get("candidate_id") != LOCKED_CANDIDATE:
        raise ValueError("Phase C candidate report identity mismatch")
    if candidate_report.get("config_sha256") != compute_phase_c_config_sha256(
        candidate_config
    ):
        raise ValueError("Phase C candidate report config hash mismatch")
    if (
        manifest_hash != LEGACY_B0_MANIFEST_SHA256
        or candidate_report.get("manifest_sha256") != manifest_hash
    ):
        raise ValueError("Phase C candidate report manifest hash mismatch")
    expected_hashes = candidate_report.get("validation_render_sha256")
    if (
        not isinstance(expected_hashes, dict)
        or expected_hashes != render_hashes
        or set(render_hashes) != set(validation["images"])
        or any(not _is_sha256(value) for value in render_hashes.values())
    ):
        raise ValueError("Phase C validation render hashes mismatch")
    if (
        candidate_report.get("timing_record_count") != 30_000
        or not isinstance(candidate_report.get("peak_gaussians"), int)
        or candidate_report["peak_gaussians"] <= 0
        or not isinstance(candidate_report.get("final_num_gaussians"), int)
        or candidate_report["final_num_gaussians"] <= 0
        or _finite(
            candidate_report.get("total_time_seconds"), "total_time_seconds"
        )
        <= 0.0
    ):
        raise ValueError("Phase C candidate integrity fields are invalid")


def build_phase_c_decision(
    baseline_report: dict,
    candidate_report: dict,
    baseline_diagnostic: dict,
    candidate_diagnostic: dict,
    *,
    integrity_passed: bool,
    provenance: dict,
) -> dict:
    baseline = _final_validation(baseline_report)
    candidate = _final_validation(candidate_report)
    baseline_score = _score50(baseline)
    candidate_score = _score50(candidate)
    missing_delta = _finite(
        candidate_diagnostic.get("missing_edge_mean"), "candidate missing_edge_mean"
    ) - _finite(
        baseline_diagnostic.get("missing_edge_mean"), "baseline missing_edge_mean"
    )
    spurious_delta = _finite(
        candidate_diagnostic.get("spurious_edge_mean"), "candidate spurious_edge_mean"
    ) - _finite(
        baseline_diagnostic.get("spurious_edge_mean"), "baseline spurious_edge_mean"
    )
    gates = {
        "score_improved": candidate_score > baseline_score,
        "lpips_not_worse": _finite(candidate.get("lpips_mean"), "candidate lpips")
        <= _finite(baseline.get("lpips_mean"), "baseline lpips"),
        "edge_errors_not_both_worse": not (
            missing_delta > 0.0 and spurious_delta > 0.0
        ),
        "candidate_integrity_passed": integrity_passed,
        "peak_vram_below_23gb": _finite(
            candidate_report.get("max_vram_mb"), "max_vram_mb"
        )
        < MAX_VRAM_MB,
    }
    passed = all(gates.values())
    failed_gates = [name for name, value in gates.items() if not value]
    return {
        "schema_version": 1,
        "phase": "C1-phase-C",
        "scene_id": PHASE_C_SCENE,
        "candidate_id": LOCKED_CANDIDATE,
        "selected_candidate": LOCKED_CANDIDATE if passed else None,
        "phase_c_passed": passed,
        "failed_gates": failed_gates,
        "baseline_score50": baseline_score,
        "candidate_score50": candidate_score,
        "delta_score50": candidate_score - baseline_score,
        "baseline_psnr_db": baseline["psnr_db_mean"],
        "candidate_psnr_db": candidate["psnr_db_mean"],
        "baseline_ssim": baseline["ssim_mean"],
        "candidate_ssim": candidate["ssim_mean"],
        "baseline_lpips": baseline["lpips_mean"],
        "candidate_lpips": candidate["lpips_mean"],
        "missing_edge_delta": missing_delta,
        "spurious_edge_delta": spurious_delta,
        "max_vram_mb": candidate_report["max_vram_mb"],
        "total_time_seconds": candidate_report.get("total_time_seconds"),
        "peak_gaussians": candidate_report.get("peak_gaussians"),
        "final_num_gaussians": candidate_report.get("final_num_gaussians"),
        "baseline_high_frequency": {
            key: baseline_diagnostic.get(key)
            for key in ("hf_l1_mean", "missing_edge_mean", "spurious_edge_mean")
        },
        "candidate_high_frequency": {
            key: candidate_diagnostic.get(key)
            for key in ("hf_l1_mean", "missing_edge_mean", "spurious_edge_mean")
        },
        "provenance": provenance,
        "gates": gates,
    }
