from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from bts_nvs.training.c1_screening import (
    build_screening_command,
    load_completed_run,
    score50,
)
from bts_nvs.training.trainer import compute_config_sha256


CANDIDATE = "C1-absgrad-t08-revopacity-v1"


def _report(scene_id: str, config_sha256: str) -> dict:
    return {
        "schema_version": 1,
        "scene_id": scene_id,
        "candidate_id": CANDIDATE,
        "step": 7000,
        "image_count": 8,
        "psnr_db_mean": 21.0,
        "ssim_mean": 0.8,
        "lpips_mean": 0.2,
        "peak_gaussians": 1_000_000,
        "max_vram_mb": 8_000.0,
        "total_time_seconds": 900.0,
        "config_sha256": config_sha256,
    }


def _config(scene_id: str) -> dict:
    return {
        "scene_id": scene_id,
        "qualification_candidate": CANDIDATE,
        "resize_factor": 1,
        "max_steps": 7000,
        "seed": 0,
        "cache_images": True,
        "pinned_transfer": True,
        "internal_holdout": True,
        "optimizer_backend": "adam",
        "precision": "fp32",
        "rolling_checkpoint": False,
        "grow_grad2d": 0.0008,
        "absgrad": True,
        "revised_opacity": True,
    }


def test_screening_command_uses_explicit_flat_run_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "phase_b" / "HCM0181"

    command = build_screening_command(
        repo_root=tmp_path / "repo",
        scenes_root=tmp_path / "scenes",
        manifests_root=tmp_path / "manifests",
        run_dir=run_dir,
        scene_id="HCM0181",
        candidate_id=CANDIDATE,
        decision=SimpleNamespace(optimizer_backend="adam", precision="fp32"),
        python_bin="python",
    )

    assert command[command.index("--output_dir") + 1] == str(run_dir)
    assert "--max_steps 7000" in " ".join(command)
    assert "--qualification_candidate C1-absgrad-t08-revopacity-v1" in " ".join(
        command
    )
    assert "--rolling_checkpoint" not in command
    assert "--resume" not in command


def test_score50_matches_formula_and_rejects_nonfinite() -> None:
    assert score50(
        {"psnr_db_mean": 25.0, "ssim_mean": 0.8, "lpips_mean": 0.2}
    ) == pytest.approx(71.0)
    with pytest.raises(ValueError, match="finite"):
        score50(
            {"psnr_db_mean": float("nan"), "ssim_mean": 0.8, "lpips_mean": 0.2}
        )


def test_completed_run_validates_locked_candidate_contract(tmp_path: Path) -> None:
    config = _config("HCM0181")
    run_dir = tmp_path / "HCM0181"
    (run_dir / "validation_renders").mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run_dir / "qualification_report.json").write_text(
        json.dumps(_report("HCM0181", compute_config_sha256(config))),
        encoding="utf-8",
    )

    report = load_completed_run(
        run_dir,
        "HCM0181",
        CANDIDATE,
        SimpleNamespace(optimizer_backend="adam", precision="fp32"),
    )

    assert report["candidate_id"] == CANDIDATE


def test_completed_run_rejects_model_checkpoint(tmp_path: Path) -> None:
    config = _config("HCM0181")
    run_dir = tmp_path / "HCM0181"
    (run_dir / "validation_renders").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "recovery.pt").write_bytes(b"checkpoint")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run_dir / "qualification_report.json").write_text(
        json.dumps(_report("HCM0181", compute_config_sha256(config))),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not contain model checkpoints"):
        load_completed_run(
            run_dir,
            "HCM0181",
            CANDIDATE,
            SimpleNamespace(optimizer_backend="adam", precision="fp32"),
        )
