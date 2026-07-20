from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import bts_nvs.training.c1_phase_b_runner as runner
from bts_nvs.training.c1_phase_a import PHASE_A_SCENES
from bts_nvs.training.c1_phase_b import (
    LOCKED_CANDIDATE,
    PHASE_B_SCENES,
    SCREENING_SCENES,
)
from bts_nvs.training.trainer import compute_config_sha256


def _report(scene: str, candidate: str, psnr: float, config_hash: str = "") -> dict:
    report = {
        "schema_version": 1,
        "scene_id": scene,
        "candidate_id": candidate,
        "step": 7000,
        "image_count": 8,
        "psnr_db_mean": psnr,
        "ssim_mean": 0.8,
        "lpips_mean": 0.2,
        "peak_gaussians": 1_000_000,
        "max_vram_mb": 8_000.0,
        "total_time_seconds": 900.0,
        "holdout_sha256": f"holdout-{scene}",
    }
    if config_hash:
        report["config_sha256"] = config_hash
    return report


def _diagnostic(scene: str, candidate: str) -> dict:
    value = 0.1 if candidate == "B0-reference" else 0.09
    return {
        "schema_version": 1,
        "scene_id": scene,
        "candidate_id": candidate,
        "image_count": 8,
        "hf_l1_mean": value,
        "missing_edge_mean": value,
        "spurious_edge_mean": value,
        "images": {},
    }


def _config(scene: str) -> dict:
    return {
        "scene_id": scene,
        "qualification_candidate": LOCKED_CANDIDATE,
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


def _write_candidate_run(run_dir: Path, scene: str) -> None:
    config = _config(scene)
    (run_dir / "validation_renders").mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run_dir / "qualification_report.json").write_text(
        json.dumps(
            _report(scene, LOCKED_CANDIDATE, 21.0, compute_config_sha256(config))
        ),
        encoding="utf-8",
    )


def _roots(tmp_path: Path) -> dict[str, Path]:
    roots = {
        name: tmp_path / name
        for name in (
            "repo",
            "scenes",
            "manifests",
            "backend",
            "baseline",
            "phase_a",
            "output",
        )
    }
    entrypoint = roots["repo"] / "src" / "bts_nvs" / "training" / "run_training.py"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.touch()
    for scene in SCREENING_SCENES:
        (roots["scenes"] / scene).mkdir(parents=True)
        manifest_root = roots["manifests"] / scene
        manifest_root.mkdir(parents=True)
        (manifest_root / "manifest.json").write_text("{}", encoding="utf-8")
        (manifest_root / "holdout.json").write_text("{}", encoding="utf-8")
        baseline = roots["baseline"] / scene / "B0-reference"
        (baseline / "validation_renders").mkdir(parents=True)
        (baseline / "qualification_report.json").write_text(
            json.dumps(_report(scene, "B0-reference", 20.0)), encoding="utf-8"
        )
    (roots["phase_a"] / "phase_a_decision.json").parent.mkdir(parents=True)
    (roots["phase_a"] / "phase_a_decision.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phase": "C1-phase-A",
                "phase_a_passed": True,
                "selected_candidate": LOCKED_CANDIDATE,
            }
        ),
        encoding="utf-8",
    )
    for scene in PHASE_A_SCENES:
        run_dir = roots["phase_a"] / scene / LOCKED_CANDIDATE
        _write_candidate_run(run_dir, scene)
        (run_dir / "high_frequency.json").write_text(
            json.dumps(_diagnostic(scene, LOCKED_CANDIDATE)), encoding="utf-8"
        )
        baseline_diagnostic = roots["phase_a"] / "baseline_diagnostics" / f"{scene}.json"
        baseline_diagnostic.parent.mkdir(parents=True, exist_ok=True)
        baseline_diagnostic.write_text(
            json.dumps(_diagnostic(scene, "B0-reference")), encoding="utf-8"
        )
    return roots


def test_phase_b_command_writes_directly_to_scene_directory(tmp_path: Path) -> None:
    command = runner.build_phase_b_command(
        repo_root=tmp_path / "repo",
        scenes_root=tmp_path / "scenes",
        manifests_root=tmp_path / "manifests",
        output_root=tmp_path / "phase_b",
        scene_id="HCM0181",
        decision=SimpleNamespace(optimizer_backend="adam", precision="fp32"),
        python_bin="python",
    )

    assert command[command.index("--output_dir") + 1] == str(
        tmp_path / "phase_b" / "HCM0181"
    )
    assert "--qualification_candidate C1-absgrad-t08-revopacity-v1" in " ".join(
        command
    )


def test_phase_b_rejects_unapproved_or_wrong_phase_a_lock(tmp_path: Path) -> None:
    decision = tmp_path / "phase_a_decision.json"
    decision.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phase": "C1-phase-A",
                "phase_a_passed": False,
                "selected_candidate": None,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="did not pass"):
        runner.load_phase_a_lock(decision)

    decision.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "phase": "C1-phase-A",
                "phase_a_passed": True,
                "selected_candidate": "C1-absgrad-t08-v1",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="winner mismatch"):
        runner.load_phase_a_lock(decision)


def test_phase_b_runner_executes_four_flat_runs_and_reuses_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    backend = SimpleNamespace(optimizer_backend="adam", precision="fp32")
    monkeypatch.setattr(runner, "load_or_create_backend_decision", lambda _: backend)
    monkeypatch.setattr(
        runner,
        "diagnostics_for_run",
        lambda **kwargs: _diagnostic(kwargs["scene_id"], kwargs["candidate_id"]),
    )
    observed: list[str] = []

    def fake_process(command, check=False):
        scene = Path(command[command.index("--scene_dir") + 1]).name
        run_dir = Path(command[command.index("--output_dir") + 1])
        _write_candidate_run(run_dir, scene)
        observed.append(scene)
        return SimpleNamespace(returncode=0)

    decision = runner.run_phase_b(
        repo_root=roots["repo"],
        scenes_root=roots["scenes"],
        manifests_root=roots["manifests"],
        backend_root=roots["backend"],
        baseline_root=roots["baseline"],
        phase_a_root=roots["phase_a"],
        output_root=roots["output"],
        python_bin="python",
        run_process=fake_process,
    )
    runner.run_phase_b(
        repo_root=roots["repo"],
        scenes_root=roots["scenes"],
        manifests_root=roots["manifests"],
        backend_root=roots["backend"],
        baseline_root=roots["baseline"],
        phase_a_root=roots["phase_a"],
        output_root=roots["output"],
        python_bin="python",
        run_process=fake_process,
    )

    assert observed == list(PHASE_B_SCENES)
    assert decision["phase_b_passed"] is True
    assert (roots["output"] / "phase_b_decision.json").is_file()
    assert all((roots["output"] / scene / "qualification_report.json").is_file() for scene in PHASE_B_SCENES)
    assert not (roots["output"] / "HCM0181" / LOCKED_CANDIDATE).exists()


def test_phase_b_preflight_reports_missing_inputs_before_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    missing = roots["baseline"] / "HNI0265" / "B0-reference" / "qualification_report.json"
    missing.unlink()
    backend_loaded = False

    def fail_if_loaded(_):
        nonlocal backend_loaded
        backend_loaded = True
        raise AssertionError("backend must not load before preflight")

    monkeypatch.setattr(runner, "load_or_create_backend_decision", fail_if_loaded)

    with pytest.raises(FileNotFoundError, match="HNI0265"):
        runner.run_phase_b(
            repo_root=roots["repo"],
            scenes_root=roots["scenes"],
            manifests_root=roots["manifests"],
            backend_root=roots["backend"],
            baseline_root=roots["baseline"],
            phase_a_root=roots["phase_a"],
            output_root=roots["output"],
            python_bin="python",
        )
    assert backend_loaded is False


def test_phase_b_rejects_partial_run_and_failed_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots = _roots(tmp_path)
    backend = SimpleNamespace(optimizer_backend="adam", precision="fp32")
    monkeypatch.setattr(runner, "load_or_create_backend_decision", lambda _: backend)
    partial = roots["output"] / PHASE_B_SCENES[0]
    partial.mkdir(parents=True)
    (partial / "metrics.jsonl").write_text("partial", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        runner.run_phase_b(
            repo_root=roots["repo"],
            scenes_root=roots["scenes"],
            manifests_root=roots["manifests"],
            backend_root=roots["backend"],
            baseline_root=roots["baseline"],
            phase_a_root=roots["phase_a"],
            output_root=roots["output"],
            python_bin="python",
        )

    partial.rename(roots["output"] / "partial")
    with pytest.raises(RuntimeError, match="exited with code 7"):
        runner.run_phase_b(
            repo_root=roots["repo"],
            scenes_root=roots["scenes"],
            manifests_root=roots["manifests"],
            backend_root=roots["backend"],
            baseline_root=roots["baseline"],
            phase_a_root=roots["phase_a"],
            output_root=roots["output"],
            python_bin="python",
            run_process=lambda *args, **kwargs: SimpleNamespace(returncode=7),
        )
