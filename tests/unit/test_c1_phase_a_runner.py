from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import bts_nvs.training.c1_phase_a_runner as runner
from bts_nvs.training.c1_candidates import C1_CANDIDATES
from bts_nvs.training.c1_phase_a import PHASE_A_SCENES


def _report(scene: str, candidate: str, psnr: float) -> dict:
    return {
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
    }


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


def _roots(tmp_path: Path):
    roots = {
        name: tmp_path / name
        for name in (
            "repo",
            "scenes",
            "manifests",
            "backend",
            "baseline",
            "output",
        )
    }
    (roots["repo"] / "src" / "bts_nvs" / "training").mkdir(parents=True)
    (roots["repo"] / "src" / "bts_nvs" / "training" / "run_training.py").touch()
    for scene in PHASE_A_SCENES:
        (roots["scenes"] / scene).mkdir(parents=True)
        (roots["manifests"] / scene).mkdir(parents=True)
        (roots["manifests"] / scene / "manifest.json").write_text("{}")
        (roots["manifests"] / scene / "holdout.json").write_text("{}")
        baseline = roots["baseline"] / scene / "B0-reference"
        (baseline / "validation_renders").mkdir(parents=True)
        (baseline / "qualification_report.json").write_text(
            json.dumps(_report(scene, "B0-reference", 20.0))
        )
    return roots


def test_phase_a_command_is_locked_and_checkpoint_free(tmp_path) -> None:
    command = runner.build_phase_a_command(
        repo_root=tmp_path / "repo",
        scenes_root=tmp_path / "scenes",
        manifests_root=tmp_path / "manifests",
        output_root=tmp_path / "output",
        scene_id="HCM0421",
        candidate_id=C1_CANDIDATES[0],
        decision=SimpleNamespace(optimizer_backend="adam-fused", precision="amp-fp16"),
        python_bin="python",
    )

    joined = " ".join(command)
    assert "--resize_factor 1" in joined
    assert "--max_steps 7000" in joined
    assert "--seed 0" in joined
    assert "--cache_images" in command
    assert "--pinned_transfer" in command
    assert "--qualification_candidate C1-absgrad-t08-v1" in joined
    assert "--optimizer_backend adam-fused" in joined
    assert "--precision amp-fp16" in joined
    assert "--rolling_checkpoint" not in command
    assert "--resume" not in command


def test_runner_executes_exact_four_pairs_and_reuses_completed_reports(
    tmp_path, monkeypatch
) -> None:
    roots = _roots(tmp_path)
    decision = SimpleNamespace(optimizer_backend="adam", precision="fp32")
    monkeypatch.setattr(runner, "load_or_create_backend_decision", lambda _: decision)
    monkeypatch.setattr(
        runner,
        "_diagnostics_for_run",
        lambda **kwargs: _diagnostic(kwargs["scene_id"], kwargs["candidate_id"]),
    )
    observed = []

    def fake_process(command, check=False):
        scene = Path(command[command.index("--scene_dir") + 1]).name
        candidate = command[command.index("--qualification_candidate") + 1]
        run_dir = Path(command[command.index("--output_dir") + 1])
        run_dir.mkdir(parents=True)
        (run_dir / "validation_renders").mkdir()
        (run_dir / "qualification_report.json").write_text(
            json.dumps(_report(scene, candidate, 21.0))
        )
        observed.append((scene, candidate))
        return SimpleNamespace(returncode=0)

    result = runner.run_phase_a(
        repo_root=roots["repo"],
        scenes_root=roots["scenes"],
        manifests_root=roots["manifests"],
        backend_root=roots["backend"],
        baseline_root=roots["baseline"],
        output_root=roots["output"],
        python_bin="python",
        run_process=fake_process,
    )
    runner.run_phase_a(
        repo_root=roots["repo"],
        scenes_root=roots["scenes"],
        manifests_root=roots["manifests"],
        backend_root=roots["backend"],
        baseline_root=roots["baseline"],
        output_root=roots["output"],
        python_bin="python",
        run_process=fake_process,
    )

    assert observed == [
        (scene, candidate)
        for scene in PHASE_A_SCENES
        for candidate in C1_CANDIDATES
    ]
    assert result["phase_a_passed"] is True
    assert (roots["output"] / "phase_a_decision.json").is_file()


def test_runner_rejects_nonempty_incomplete_run_directory(tmp_path, monkeypatch) -> None:
    roots = _roots(tmp_path)
    monkeypatch.setattr(
        runner,
        "load_or_create_backend_decision",
        lambda _: SimpleNamespace(optimizer_backend="adam", precision="fp32"),
    )
    partial = roots["output"] / PHASE_A_SCENES[0] / C1_CANDIDATES[0]
    partial.mkdir(parents=True)
    (partial / "metrics.jsonl").write_text("partial")

    with pytest.raises(ValueError, match="non-empty"):
        runner.run_phase_a(
            repo_root=roots["repo"],
            scenes_root=roots["scenes"],
            manifests_root=roots["manifests"],
            backend_root=roots["backend"],
            baseline_root=roots["baseline"],
            output_root=roots["output"],
            python_bin="python",
        )


def test_runner_stops_on_failed_training_process(tmp_path, monkeypatch) -> None:
    roots = _roots(tmp_path)
    monkeypatch.setattr(
        runner,
        "load_or_create_backend_decision",
        lambda _: SimpleNamespace(optimizer_backend="adam", precision="fp32"),
    )

    with pytest.raises(RuntimeError, match="exited with code 7"):
        runner.run_phase_a(
            repo_root=roots["repo"],
            scenes_root=roots["scenes"],
            manifests_root=roots["manifests"],
            backend_root=roots["backend"],
            baseline_root=roots["baseline"],
            output_root=roots["output"],
            python_bin="python",
            run_process=lambda *args, **kwargs: SimpleNamespace(returncode=7),
        )
