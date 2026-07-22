from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import json
import numpy as np
import pytest
import torch

import bts_nvs.training.run_training as run_training


def _args(**changes):
    values = {
        "scene_dir": "scene",
        "output_dir": "run",
        "manifest_dir": None,
        "resize_factor": 4,
        "max_steps": 500,
        "checkpoint_every": 100,
        "seed": 7,
        "resume": None,
        "cache_images": False,
        "pinned_transfer": False,
        "profile_input": False,
        "internal_holdout": False,
        "qualification_candidate": None,
        "full_length_qualification": False,
        "backend_qualification": False,
        "optimizer_backend": "adam",
        "precision": "fp32",
        "rolling_checkpoint": False,
        "candidate_id": None,
        "experiment_stage": None,
        "authorized_candidate_id": None,
        "stop_step": None,
    }
    values.update(changes)
    return Namespace(**values)


def test_cuda_preflight_rejects_cpu_only_torch(monkeypatch):
    monkeypatch.setattr(run_training.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        run_training.run_cuda_preflight("adam-fused", "amp-fp16")


def test_preflight_requires_a_finite_gradient():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))

    with pytest.raises(RuntimeError, match="finite parameter gradient"):
        run_training.validate_preflight_gradients([parameter])

    (parameter * 2.0).sum().backward()
    run_training.validate_preflight_gradients([parameter])


@pytest.mark.parametrize("factor", [0, -1, True])
def test_resize_factor_must_be_positive_integer(factor):
    with pytest.raises(ValueError, match="resize_factor"):
        run_training.validate_resize_factor(factor)


def test_training_config_binds_preprocessing_identity():
    manifest = SimpleNamespace(scene_id="HCM0181")
    config = run_training.build_training_config(
        _args(resize_factor=2, seed=19),
        manifest,
        resize=(660, 494),
    )

    assert config["scene_id"] == "HCM0181"
    assert config["resize_factor"] == 2
    assert config["resize_width"] == 660
    assert config["resize_height"] == 494
    assert config["undistort"] is True
    assert config["seed"] == 19
    assert config["cache_images"] is False
    assert config["pinned_transfer"] is False
    assert config["optimizer_backend"] == "adam"
    assert config["precision"] == "fp32"
    assert config["rolling_checkpoint"] is False


@pytest.mark.parametrize(
    "backend,precision",
    [
        ("adam", "fp32"),
        ("adam-fused", "fp32"),
        ("adam-fused", "amp-fp16"),
    ],
)
def test_training_backend_contract_accepts_supported_pairs(backend, precision):
    run_training.validate_training_backend(
        _args(optimizer_backend=backend, precision=precision)
    )


def test_training_backend_contract_rejects_amp_with_unfused_adam():
    with pytest.raises(ValueError, match="amp-fp16 requires adam-fused"):
        run_training.validate_training_backend(
            _args(optimizer_backend="adam", precision="amp-fp16")
        )


def test_rolling_checkpoint_is_allowed_only_for_ordinary_or_full_length_runs():
    run_training.validate_rolling_checkpoint_args(_args(rolling_checkpoint=True))
    run_training.validate_rolling_checkpoint_args(
        _args(rolling_checkpoint=True, full_length_qualification=True)
    )

    for change in (
        {"profile_input": True},
        {"backend_qualification": True},
        {"qualification_candidate": "B0-reference"},
    ):
        with pytest.raises(ValueError, match="rolling checkpoint"):
            run_training.validate_rolling_checkpoint_args(
                _args(rolling_checkpoint=True, **change)
            )


def test_profile_mode_requires_fresh_exact_550_step_run():
    run_training.validate_profile_args(_args(profile_input=True, max_steps=550))

    with pytest.raises(ValueError, match="550"):
        run_training.validate_profile_args(_args(profile_input=True, max_steps=549))
    with pytest.raises(ValueError, match="fresh"):
        run_training.validate_profile_args(
            _args(profile_input=True, max_steps=550, resume="checkpoint.pt")
        )


def test_profile_implies_internal_holdout():
    assert run_training.internal_holdout_enabled(
        _args(profile_input=True, max_steps=550)
    )
    assert run_training.internal_holdout_enabled(_args(internal_holdout=True))
    assert not run_training.internal_holdout_enabled(_args())


def test_backend_qualification_locks_1000_step_l4_contract():
    valid = _args(
        backend_qualification=True,
        resize_factor=1,
        max_steps=1000,
        checkpoint_every=1000,
        seed=0,
        cache_images=True,
        pinned_transfer=True,
        optimizer_backend="adam-fused",
        precision="amp-fp16",
    )
    run_training.validate_backend_qualification_args(valid)
    assert run_training.internal_holdout_enabled(valid)
    assert run_training.should_save_checkpoints(valid) is False

    for change in (
        {"max_steps": 999},
        {"resize_factor": 2},
        {"seed": 1},
        {"cache_images": False},
        {"pinned_transfer": False},
        {"resume": "checkpoint.pt"},
        {"full_length_qualification": True},
    ):
        with pytest.raises(ValueError, match="backend qualification"):
            run_training.validate_backend_qualification_args(
                _args(**(vars(valid) | change))
            )


def test_qualification_candidate_locks_run_contract():
    valid = _args(
        qualification_candidate="B0-compact",
        resize_factor=1,
        max_steps=7000,
        seed=0,
        cache_images=True,
        pinned_transfer=True,
    )
    run_training.validate_qualification_args(valid)
    assert run_training.internal_holdout_enabled(valid)

    for change in (
        {"max_steps": 6999},
        {"resize_factor": 2},
        {"seed": 1},
        {"cache_images": False},
        {"pinned_transfer": False},
    ):
        with pytest.raises(ValueError, match="qualification"):
            run_training.validate_qualification_args(_args(**(vars(valid) | change)))


def test_full_length_qualification_locks_30k_contract(tmp_path):
    output = tmp_path / "run"
    valid = _args(
        output_dir=str(output),
        full_length_qualification=True,
        resize_factor=1,
        max_steps=30_000,
        checkpoint_every=3_000,
        seed=0,
        cache_images=True,
        pinned_transfer=True,
    )

    run_training.validate_full_length_args(valid)
    assert run_training.internal_holdout_enabled(valid)
    assert run_training.should_save_checkpoints(valid)
    run_training.validate_full_length_scene("HCM0181", valid)

    for change in (
        {"max_steps": 7_000},
        {"resize_factor": 2},
        {"checkpoint_every": 6_000},
        {"seed": 1},
        {"cache_images": False},
        {"pinned_transfer": False},
        {"qualification_candidate": "B0-reference"},
    ):
        with pytest.raises(ValueError, match="full-length"):
            run_training.validate_full_length_args(_args(**(vars(valid) | change)))

    with pytest.raises(ValueError, match="HCM0181"):
        run_training.validate_full_length_scene("HCM0421", valid)


def test_full_length_resume_is_fixed_to_rolling_checkpoint(tmp_path):
    output = tmp_path / "run"
    recovery = output / "checkpoints" / "recovery.pt"
    recovery.parent.mkdir(parents=True)
    recovery.write_bytes(b"checkpoint")
    valid = _args(
        output_dir=str(output),
        full_length_qualification=True,
        resize_factor=1,
        max_steps=30_000,
        checkpoint_every=3_000,
        seed=0,
        cache_images=True,
        pinned_transfer=True,
        resume=str(recovery),
    )

    run_training.validate_full_length_args(valid)

    valid.resume = str(output / "checkpoints" / "other.pt")
    with pytest.raises(ValueError, match="recovery.pt"):
        run_training.validate_full_length_args(valid)


def test_recovery_checkpoint_must_be_complete_and_at_expected_step(
    monkeypatch, tmp_path
):
    complete = {
        "step": 30_000,
        "gaussians": {},
        "optimizers": {},
        "scheduler": {},
        "strategy_state": {},
        "active_sh_degree": 3,
        "rng_states": {},
        "manifest_hash": "manifest",
        "config_hash": "config",
    }
    monkeypatch.setattr(
        run_training,
        "load_checkpoint",
        lambda *args, **kwargs: complete,
    )

    run_training.validate_recovery_checkpoint(
        tmp_path / "recovery.pt", "manifest", "config", 30_000
    )

    monkeypatch.setattr(
        run_training,
        "load_checkpoint",
        lambda *args, **kwargs: {
            key: value for key, value in complete.items() if key != "rng_states"
        },
    )
    with pytest.raises(ValueError, match="rng_states"):
        run_training.validate_recovery_checkpoint(
            tmp_path / "recovery.pt", "manifest", "config", 30_000
        )


def test_completed_resume_skips_optimization_but_allows_finalization():
    assert run_training.optimization_required(27_000, 30_000) is True
    assert run_training.optimization_required(30_000, 30_000) is False

    with pytest.raises(ValueError, match="beyond"):
        run_training.optimization_required(30_001, 30_000)


def test_clean_git_commit_rejects_tracked_changes(monkeypatch, tmp_path):
    responses = iter(("abc123\n", " M src/file.py\n"))
    monkeypatch.setattr(
        run_training.subprocess,
        "check_output",
        lambda *args, **kwargs: next(responses),
    )

    with pytest.raises(RuntimeError, match="dirty"):
        run_training.read_clean_git_commit(tmp_path)


def test_full_length_config_uses_reference_threshold():
    args = _args(full_length_qualification=True)
    config = run_training.build_training_config(
        args, SimpleNamespace(scene_id="HCM0181"), resize=(1320, 989)
    )

    assert config["full_length_qualification"] is True
    assert config["grow_grad2d"] == pytest.approx(0.0002)


def test_training_config_applies_locked_candidate_threshold():
    manifest = SimpleNamespace(scene_id="HCM0181")
    config = run_training.build_training_config(
        _args(qualification_candidate="B0-compact"),
        manifest,
        resize=(8, 6),
    )

    assert config["qualification_candidate"] == "B0-compact"
    assert config["grow_grad2d"] == pytest.approx(0.0003)
    assert run_training.should_save_checkpoints(_args()) is True
    assert run_training.should_save_checkpoints(
        _args(qualification_candidate="B0-compact")
    ) is False


def test_internal_holdout_requires_colocated_artifact(tmp_path, monkeypatch):
    manifest = SimpleNamespace(scene_id="HCM0181")
    with pytest.raises(FileNotFoundError, match="holdout.json"):
        run_training.load_internal_holdout(tmp_path, manifest, enabled=True)
    assert run_training.load_internal_holdout(tmp_path, manifest, enabled=False) is None

    path = tmp_path / "holdout.json"
    path.write_text("{}")
    expected = SimpleNamespace(train_image_names=("a.JPG",))
    monkeypatch.setattr(
        run_training,
        "load_holdout_split",
        lambda source, value: expected,
    )
    assert run_training.load_internal_holdout(tmp_path, manifest, enabled=True) is expected


def test_training_config_records_holdout_identity():
    manifest = SimpleNamespace(scene_id="HCM0181")
    split = SimpleNamespace(
        algorithm="pose_fps_guard2_v1",
        manifest_sha256="abc",
        train_image_names=("a", "b"),
        guard_image_names=("c",),
        validation_image_names=("d",),
    )

    config = run_training.build_training_config(
        _args(), manifest, resize=(8, 6), split=split
    )

    assert config["holdout_algorithm"] == "pose_fps_guard2_v1"
    assert config["holdout_manifest_sha256"] == "abc"
    assert config["internal_train_count"] == 2
    assert config["guard_count"] == 1
    assert config["validation_count"] == 1


def test_internal_holdout_replaces_only_sparse_initialization(monkeypatch):
    @dataclass(frozen=True)
    class Manifest:
        sparse_points: np.ndarray
        sparse_colors: np.ndarray

    manifest = Manifest(
        np.zeros((1, 3), dtype=np.float64),
        np.zeros((1, 3), dtype=np.uint8),
    )
    split = SimpleNamespace()
    sparse = SimpleNamespace(
        points=np.ones((2, 3), dtype=np.float64),
        colors=np.full((2, 3), 7, dtype=np.uint8),
    )
    monkeypatch.setattr(
        run_training,
        "build_split_sparse_initialization",
        lambda value, root, holdout: sparse,
    )

    result = run_training.build_initialization_manifest(manifest, Path("scene"), split)

    np.testing.assert_array_equal(result.sparse_points, sparse.points)
    np.testing.assert_array_equal(result.sparse_colors, sparse.colors)
    np.testing.assert_array_equal(manifest.sparse_points, np.zeros((1, 3)))


def test_host_resource_preflight_rejects_swap_and_low_cache_ram():
    run_training.validate_host_resources(cache_bytes=0, memory_status=None)
    with pytest.raises(RuntimeError, match="swap"):
        run_training.validate_host_resources(
            cache_bytes=0,
            memory_status=(8 * 1024**3, 1),
        )
    with pytest.raises(MemoryError, match="host RAM"):
        run_training.validate_host_resources(
            cache_bytes=5 * 1024**3,
            memory_status=(8 * 1024**3, 0),
        )


def test_non_resume_run_rejects_non_empty_output(tmp_path: Path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "metrics.jsonl").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--resume"):
        run_training.validate_output_directory(output, resume=None)


def test_resume_requires_existing_checkpoint(tmp_path: Path):
    missing = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError, match="resume checkpoint"):
        run_training.validate_output_directory(tmp_path / "run", resume=missing)


def test_convergence_report_records_deltas_and_non_blank_render(tmp_path: Path):
    initial = {
        "psnr_db": 10.0,
        "psnr_is_infinite": False,
        "ssim": 0.20,
        "alpha_coverage": 0.4,
        "rgb_std": 0.1,
        "valid_fraction": 0.99,
    }
    final = {**initial, "psnr_db": 12.5, "ssim": 0.28}

    report = run_training.write_convergence_report(
        tmp_path / "convergence.json", initial, final
    )

    assert report["psnr_delta_db"] == pytest.approx(2.5)
    assert report["ssim_delta"] == pytest.approx(0.08)
    assert report["quality_improved"] is True
    assert report["final_render_non_blank"] is True
    assert json.loads((tmp_path / "convergence.json").read_text()) == report


def test_initial_metrics_round_trip_for_resume(tmp_path: Path):
    path = tmp_path / "initial_metrics.json"
    metrics = {"psnr_db": 10.0, "ssim": 0.2}

    run_training.write_json_record(path, metrics)

    assert run_training.read_json_record(path) == metrics


def test_fresh_generic_confirmation_writes_initial_validation(
    monkeypatch, tmp_path
):
    output = tmp_path / "confirm"
    args = _generic_args(
        "confirm",
        "E1-density-absgrad-t04-v1",
        output_dir=str(output),
    )
    trainer = SimpleNamespace(device=torch.device("cpu"))
    validation_dataset = object()
    backend = object()
    validation = {"image_count": 3, "lpips_mean": 0.25}
    observed = []

    monkeypatch.setattr(run_training, "LpipsBackend", lambda **kwargs: backend)

    def evaluate(trainer_arg, dataset_arg, backend_arg, render_dir):
        observed.append((trainer_arg, dataset_arg, backend_arg, render_dir))
        return validation

    monkeypatch.setattr(run_training, "evaluate_internal_validation", evaluate)

    actual_backend, actual_validation = run_training.prepare_initial_validation(
        args,
        trainer,
        validation_dataset,
    )

    assert actual_backend is backend
    assert actual_validation == validation
    assert observed == [(trainer, validation_dataset, backend, None)]
    assert run_training.read_json_record(output / "initial_validation.json") == validation


def test_resumed_generic_confirmation_reuses_initial_validation_without_recompute(
    monkeypatch, tmp_path
):
    output = tmp_path / "confirm"
    initial_validation = {"image_count": 3, "lpips_mean": 0.25}
    run_training.write_json_record(
        output / "initial_validation.json",
        initial_validation,
    )
    args = _generic_args(
        "confirm",
        "E1-density-absgrad-t04-v1",
        output_dir=str(output),
        resume=str(output / "checkpoints" / "recovery.pt"),
    )
    backend = object()
    monkeypatch.setattr(run_training, "LpipsBackend", lambda **kwargs: backend)

    def fail(*args, **kwargs):
        raise AssertionError("initial validation must not be recomputed on resume")

    monkeypatch.setattr(run_training, "evaluate_internal_validation", fail)

    actual_backend, actual_validation = run_training.prepare_initial_validation(
        args,
        SimpleNamespace(device=torch.device("cpu")),
        object(),
    )

    assert actual_backend is backend
    assert actual_validation == initial_validation


def _generic_args(stage: str, candidate_id: str, **changes):
    values = {
        "experiment_stage": stage,
        "candidate_id": candidate_id,
        "authorized_candidate_id": (
            candidate_id if stage in ("confirm", "production") else None
        ),
        "resize_factor": 1,
        "max_steps": 7_000,
        "seed": 0,
        "cache_images": True,
        "pinned_transfer": True,
        "internal_holdout": True,
    }
    values.update(changes)
    return _args(**values)


def test_parse_args_accepts_generic_identity_authorization_and_stop_step(monkeypatch):
    monkeypatch.setattr(
        run_training.sys,
        "argv",
        [
            "run_training.py",
            "--scene_dir",
            "scene",
            "--output_dir",
            "run",
            "--candidate_id",
            "E1-density-absgrad-t04-v1",
            "--experiment_stage",
            "confirm",
            "--authorized_candidate_id",
            "E1-density-absgrad-t04-v1",
            "--stop_step",
            "15000",
        ],
    )

    args = run_training.parse_args()

    assert args.candidate_id == "E1-density-absgrad-t04-v1"
    assert args.experiment_stage == "confirm"
    assert args.authorized_candidate_id == "E1-density-absgrad-t04-v1"
    assert args.stop_step == 15_000


def test_generic_identity_is_all_or_none_and_uses_stage_candidate_contract():
    assert run_training.validate_generic_experiment_args(_args()) is None

    with pytest.raises(ValueError, match="generic experiment"):
        run_training.validate_generic_experiment_args(_args(stop_step=100))
    with pytest.raises(ValueError, match="generic experiment"):
        run_training.validate_generic_experiment_args(
            _args(authorized_candidate_id="E1-density-absgrad-t04-v1")
        )
    with pytest.raises(ValueError, match="together"):
        run_training.validate_generic_experiment_args(
            _args(candidate_id="B0-reference")
        )
    with pytest.raises(ValueError, match="legacy"):
        run_training.validate_generic_experiment_args(
            _generic_args(
                "reference", "B0-reference", qualification_candidate="B0-reference"
            )
        )
    with pytest.raises(ValueError, match="reference"):
        run_training.validate_generic_experiment_args(
            _generic_args("reference", "E1-density-absgrad-t04-v1")
        )
    with pytest.raises(ValueError, match="screen"):
        run_training.validate_generic_experiment_args(
            _generic_args("screen", "B0-reference")
        )


@pytest.mark.parametrize("stage", ("reference", "screen"))
def test_generic_early_stages_reject_candidate_authorization(stage):
    candidate_id = (
        "B0-reference" if stage == "reference" else "E1-density-absgrad-t04-v1"
    )

    with pytest.raises(ValueError, match="authorization.*absent"):
        run_training.validate_generic_experiment_args(
            _generic_args(
                stage,
                candidate_id,
                authorized_candidate_id=candidate_id,
            )
        )


@pytest.mark.parametrize("stage", ("confirm", "production"))
@pytest.mark.parametrize(
    "authorized_candidate_id",
    (None, "E1-density-scale005-v1"),
)
def test_generic_late_stages_require_explicit_matching_candidate_authorization(
    stage, authorized_candidate_id
):
    candidate_id = "E1-density-absgrad-t04-v1"

    with pytest.raises(ValueError, match="authorization.*match candidate_id"):
        run_training.validate_generic_experiment_args(
            _generic_args(
                stage,
                candidate_id,
                max_steps=30_000,
                stop_step=30_000,
                checkpoint_every=3_000,
                rolling_checkpoint=True,
                internal_holdout=stage == "confirm",
                authorized_candidate_id=authorized_candidate_id,
            )
        )


def test_generic_reference_and_screen_lock_fresh_7k_holdout_contract():
    reference = _generic_args("reference", "B0-reference")
    screen = _generic_args("screen", "E1-density-absgrad-t04-v1")

    run_training.validate_generic_experiment_args(reference)
    run_training.validate_generic_experiment_args(screen)
    assert run_training.training_target_step(reference) == 7_000
    assert run_training.should_save_checkpoints(reference) is False
    assert run_training.should_save_checkpoints(screen) is False

    for change in (
        {"max_steps": 6_999},
        {"stop_step": 6_999},
        {"resize_factor": 2},
        {"seed": 1},
        {"cache_images": False},
        {"pinned_transfer": False},
        {"internal_holdout": False},
        {"resume": "recovery.pt"},
        {"rolling_checkpoint": True},
    ):
        with pytest.raises(ValueError, match="reference"):
            run_training.validate_generic_experiment_args(
                _args(**(vars(reference) | change))
            )


def test_generic_confirmation_locks_30k_schedule_and_rolling_recovery(tmp_path):
    output = tmp_path / "confirm"
    valid = _generic_args(
        "confirm",
        "E1-density-absgrad-t04-v1",
        output_dir=str(output),
        max_steps=30_000,
        stop_step=15_000,
        checkpoint_every=3_000,
        rolling_checkpoint=True,
    )

    run_training.validate_generic_experiment_args(valid)
    assert run_training.training_target_step(valid) == 15_000
    assert run_training.should_save_checkpoints(valid) is True

    recovery = output / "checkpoints" / "recovery.pt"
    resumed = _args(**(vars(valid) | {"stop_step": 30_000, "resume": str(recovery)}))
    run_training.validate_generic_experiment_args(resumed)

    with pytest.raises(ValueError, match="resume.*30000"):
        run_training.validate_generic_experiment_args(
            _args(**(vars(valid) | {"resume": str(recovery)}))
        )

    for change in (
        {"max_steps": 15_000},
        {"stop_step": None},
        {"stop_step": 20_000},
        {"checkpoint_every": 6_000},
        {"rolling_checkpoint": False},
        {"internal_holdout": False},
        {"resume": str(output / "checkpoints" / "other.pt")},
    ):
        with pytest.raises(ValueError, match="confirm"):
            run_training.validate_generic_experiment_args(
                _args(**(vars(valid) | change))
            )


def test_confirmation_resume_requires_complete_15k_recovery(
    monkeypatch, tmp_path
):
    output = tmp_path / "confirm"
    recovery = output / "checkpoints" / "recovery.pt"
    args = _generic_args(
        "confirm",
        "E1-density-absgrad-t04-v1",
        output_dir=str(output),
        max_steps=30_000,
        stop_step=30_000,
        checkpoint_every=3_000,
        rolling_checkpoint=True,
        resume=str(recovery),
    )
    observed = {}

    def validate(
        path,
        manifest_hash,
        config_hash,
        expected_step,
        *,
        require_precision_state,
    ):
        observed.update(
            path=path,
            manifest_hash=manifest_hash,
            config_hash=config_hash,
            expected_step=expected_step,
            require_precision_state=require_precision_state,
        )

    monkeypatch.setattr(run_training, "validate_recovery_checkpoint", validate)

    run_training.validate_confirmation_resume(
        args,
        manifest_hash="manifest",
        config_hash="config",
    )

    assert observed == {
        "path": recovery,
        "manifest_hash": "manifest",
        "config_hash": "config",
        "expected_step": 15_000,
        "require_precision_state": True,
    }


def test_confirmation_rejects_missing_precision_state_but_legacy_allows_it(
    monkeypatch, tmp_path
):
    output = tmp_path / "confirm"
    recovery = output / "checkpoints" / "recovery.pt"
    complete_legacy_state = {
        "step": 15_000,
        "gaussians": {},
        "optimizers": {},
        "scheduler": {},
        "strategy_state": {},
        "active_sh_degree": 3,
        "rng_states": {},
        "manifest_hash": "manifest",
        "config_hash": "config",
    }
    monkeypatch.setattr(
        run_training,
        "load_checkpoint",
        lambda *args, **kwargs: complete_legacy_state,
    )

    run_training.validate_recovery_checkpoint(
        recovery,
        "manifest",
        "config",
        15_000,
    )

    args = _generic_args(
        "confirm",
        "E1-density-absgrad-t04-v1",
        output_dir=str(output),
        max_steps=30_000,
        stop_step=30_000,
        checkpoint_every=3_000,
        rolling_checkpoint=True,
        resume=str(recovery),
    )
    with pytest.raises(ValueError, match="precision_state"):
        run_training.validate_confirmation_resume(
            args,
            manifest_hash="manifest",
            config_hash="config",
        )


def test_final_confirmation_recovery_requires_current_step_and_precision_state(
    monkeypatch, tmp_path
):
    output = tmp_path / "confirm"
    args = _generic_args(
        "confirm",
        "E1-density-absgrad-t04-v1",
        output_dir=str(output),
        max_steps=30_000,
        stop_step=30_000,
        checkpoint_every=3_000,
        rolling_checkpoint=True,
    )
    observed = {}

    def validate(
        path,
        manifest_hash,
        config_hash,
        expected_step,
        *,
        require_precision_state,
    ):
        observed.update(
            path=path,
            manifest_hash=manifest_hash,
            config_hash=config_hash,
            expected_step=expected_step,
            require_precision_state=require_precision_state,
        )

    monkeypatch.setattr(run_training, "validate_recovery_checkpoint", validate)

    run_training.validate_confirmation_recovery(
        args,
        manifest_hash="manifest",
        config_hash="config",
        expected_step=30_000,
    )

    assert observed == {
        "path": output / "checkpoints" / "recovery.pt",
        "manifest_hash": "manifest",
        "config_hash": "config",
        "expected_step": 30_000,
        "require_precision_state": True,
    }


def test_non_confirmation_or_fresh_run_skips_confirmation_recovery_validation(
    monkeypatch,
):
    def fail(*args, **kwargs):
        raise AssertionError("recovery validation should not run")

    monkeypatch.setattr(run_training, "validate_recovery_checkpoint", fail)

    run_training.validate_confirmation_resume(
        _generic_args(
            "confirm",
            "E1-density-absgrad-t04-v1",
            max_steps=30_000,
            stop_step=15_000,
            checkpoint_every=3_000,
            rolling_checkpoint=True,
        ),
        manifest_hash="manifest",
        config_hash="config",
    )
    run_training.validate_confirmation_resume(
        _generic_args("screen", "E1-density-absgrad-t04-v1"),
        manifest_hash="manifest",
        config_hash="config",
    )


def test_generic_production_locks_full_30k_without_holdout():
    valid = _generic_args(
        "production",
        "E1-density-scale005-v1",
        max_steps=30_000,
        stop_step=30_000,
        checkpoint_every=3_000,
        rolling_checkpoint=True,
        internal_holdout=False,
    )

    run_training.validate_generic_experiment_args(valid)
    assert run_training.training_target_step(valid) == 30_000
    assert run_training.internal_holdout_enabled(valid) is False
    assert run_training.should_save_checkpoints(valid) is True

    for change in (
        {"stop_step": None},
        {"stop_step": 15_000},
        {"internal_holdout": True},
    ):
        with pytest.raises(ValueError, match="production"):
            run_training.validate_generic_experiment_args(
                _args(**(vars(valid) | change))
            )


def test_generic_candidate_overrides_do_not_include_stop_step():
    manifest = SimpleNamespace(scene_id="HCM0539")
    first = _generic_args(
        "confirm",
        "E1-density-absgrad-t04-v1",
        max_steps=30_000,
        stop_step=15_000,
        checkpoint_every=3_000,
        rolling_checkpoint=True,
    )
    second = _args(**(vars(first) | {"stop_step": 30_000}))

    config_15k = run_training.build_training_config(
        first, manifest, resize=(1320, 989)
    )
    config_30k = run_training.build_training_config(
        second, manifest, resize=(1320, 989)
    )

    assert config_15k == config_30k
    assert config_15k["candidate_id"] == "E1-density-absgrad-t04-v1"
    assert config_15k["experiment_stage"] == "confirm"
    assert config_15k["absgrad"] is True
    assert config_15k["grow_grad2d"] == pytest.approx(0.0004)
    assert "stop_step" not in config_15k


def test_experiment_resource_summary_uses_summary_and_metric_records(tmp_path):
    output = tmp_path / "run"
    run_training.write_json_record(
        output / "summary.json",
        {
            "total_time_seconds": 123.5,
            "max_vram_mb": 4096.0,
            "final_num_gaussians": 11,
        },
    )
    (output / "metrics.jsonl").write_text(
        "\n".join(
            json.dumps({"step": step, "num_gaussians": count})
            for step, count in ((1, 5), (2, 13), (3, 11))
        )
        + "\n",
        encoding="utf-8",
    )

    assert run_training.build_experiment_resource_summary(output) == {
        "total_time_seconds": 123.5,
        "max_vram_mb": 4096.0,
        "peak_gaussians": 13,
        "final_num_gaussians": 11,
    }


def test_resumed_training_summary_merge_accumulates_resources_only():
    prior = {
        "total_steps": 15_000,
        "total_time_seconds": 120.5,
        "max_vram_mb": 4096.0,
        "final_num_gaussians": 10,
    }
    current = {
        "total_steps": 30_000,
        "total_time_seconds": 180.25,
        "max_vram_mb": 3584.0,
        "final_num_gaussians": 12,
        "current_segment_field": "authoritative",
    }

    merged = run_training.merge_resumed_training_summaries(prior, current)

    assert merged == {
        **current,
        "total_time_seconds": 300.75,
        "max_vram_mb": 4096.0,
    }
    assert prior["total_time_seconds"] == 120.5
    assert current["total_time_seconds"] == 180.25


@pytest.mark.parametrize(
    "source, field, value",
    (
        ("prior", "total_time_seconds", True),
        ("prior", "max_vram_mb", -1.0),
        ("current", "total_time_seconds", "180.0"),
        ("current", "max_vram_mb", float("inf")),
        ("current", "total_time_seconds", float("nan")),
    ),
)
def test_resumed_training_summary_merge_rejects_invalid_resources(
    source, field, value
):
    prior = {"total_time_seconds": 120.0, "max_vram_mb": 4096.0}
    current = {"total_time_seconds": 180.0, "max_vram_mb": 3584.0}
    (prior if source == "prior" else current)[field] = value

    with pytest.raises(ValueError, match=f"{source} {field}.*finite nonnegative"):
        run_training.merge_resumed_training_summaries(prior, current)


def test_finalize_generic_resume_summary_feeds_merged_experiment_resources(tmp_path):
    output = tmp_path / "confirm"
    prior = {
        "total_steps": 15_000,
        "total_time_seconds": 120.0,
        "max_vram_mb": 4096.0,
        "final_num_gaussians": 10,
    }
    current = {
        "total_steps": 30_000,
        "total_time_seconds": 180.0,
        "max_vram_mb": 3584.0,
        "final_num_gaussians": 12,
    }
    run_training.write_json_record(output / "summary.json", current)
    (output / "metrics.jsonl").write_text(
        "\n".join(
            json.dumps({"step": step, "num_gaussians": count})
            for step, count in ((15_000, 10), (30_000, 12))
        )
        + "\n",
        encoding="utf-8",
    )

    run_training.finalize_generic_resume_summary(
        output,
        prior,
        optimization_ran=True,
    )

    assert run_training.build_experiment_resource_summary(output) == {
        "total_time_seconds": 300.0,
        "max_vram_mb": 4096.0,
        "peak_gaussians": 12,
        "final_num_gaussians": 12,
    }
    assert run_training.read_json_record(output / "summary.json")["total_steps"] == 30_000


def test_finalize_generic_resume_summary_does_not_double_count_without_optimization(
    tmp_path,
):
    output = tmp_path / "confirm"
    prior = {
        "total_steps": 30_000,
        "total_time_seconds": 300.0,
        "max_vram_mb": 4096.0,
        "final_num_gaussians": 12,
    }
    run_training.write_json_record(output / "summary.json", prior)

    run_training.finalize_generic_resume_summary(
        output,
        prior,
        optimization_ran=False,
    )

    assert run_training.read_json_record(output / "summary.json") == prior


def test_generic_internal_holdout_writes_all_module_one_reports(
    monkeypatch, tmp_path
):
    output = tmp_path / "screen"
    args = _generic_args(
        "screen",
        "E1-density-absgrad-t04-v1",
        output_dir=str(output),
    )
    manifest = SimpleNamespace(scene_id="HCM0539")
    split = SimpleNamespace(manifest_sha256="a" * 64)
    validation_dataset = object()
    trainer = SimpleNamespace(
        device=torch.device("cpu"),
        config_hash="b" * 64,
        manifest_hash="c" * 64,
    )
    full_frame = {
        "image_count": 3,
        "psnr_db_mean": 24.0,
        "ssim_mean": 0.8,
        "lpips_mean": 0.2,
        "valid_fraction_mean": 0.95,
        "images": {name: {} for name in ("a.JPG", "b.JPG", "c.JPG")},
    }
    detail = {"schema_version": 1, "scene_id": "HCM0539", "image_count": 3}
    pose = {
        "schema_version": 1,
        "scene_id": "HCM0539",
        "holdout_manifest_sha256": "a" * 64,
        "image_count": 3,
    }
    backend = object()
    observed = {}

    def evaluate(trainer_arg, dataset_arg, backend_arg, render_dir):
        observed["evaluation"] = (
            trainer_arg,
            dataset_arg,
            backend_arg,
            render_dir,
        )
        return full_frame

    def build_report(**arguments):
        observed["experiment"] = arguments
        return {"schema_version": 1, "candidate_id": arguments["candidate_id"]}

    monkeypatch.setattr(run_training, "LpipsBackend", lambda **kwargs: backend)
    monkeypatch.setattr(run_training, "evaluate_internal_validation", evaluate)
    monkeypatch.setattr(
        run_training, "evaluate_detail_directory", lambda dataset, path: detail
    )
    monkeypatch.setattr(run_training, "build_pose_strata", lambda *_: pose)
    monkeypatch.setattr(run_training, "build_experiment_report", build_report)

    run_training.write_json_record(
        output / "summary.json",
        {
            "total_time_seconds": 20.0,
            "max_vram_mb": 1024.0,
            "final_num_gaussians": 9,
        },
    )
    (output / "metrics.jsonl").write_text(
        json.dumps({"step": 7_000, "num_gaussians": 12}) + "\n",
        encoding="utf-8",
    )

    run_training.generate_generic_experiment_reports(
        args=args,
        trainer=trainer,
        manifest=manifest,
        split=split,
        validation_dataset=validation_dataset,
    )

    assert observed["evaluation"] == (
        trainer,
        validation_dataset,
        backend,
        output / "validation_renders",
    )
    assert observed["experiment"] == {
        "scene_id": "HCM0539",
        "candidate_id": "E1-density-absgrad-t04-v1",
        "step": 7_000,
        "config_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
        "holdout_sha256": "a" * 64,
        "full_frame_report": full_frame,
        "detail_report": detail,
        "pose_strata_report": pose,
        "resource_summary": {
            "total_time_seconds": 20.0,
            "max_vram_mb": 1024.0,
            "peak_gaussians": 12,
            "final_num_gaussians": 9,
        },
    }
    qualification = json.loads(
        (output / "qualification_report.json").read_text(encoding="utf-8")
    )
    assert qualification["candidate_id"] == "E1-density-absgrad-t04-v1"
    assert qualification["step"] == 7_000
    assert json.loads((output / "detail_metrics.json").read_text()) == detail
    assert json.loads((output / "pose_strata.json").read_text()) == pose
    assert json.loads((output / "experiment_report.json").read_text()) == {
        "schema_version": 1,
        "candidate_id": "E1-density-absgrad-t04-v1",
    }


def test_15k_confirmation_snapshot_atomically_preserves_auditable_reports(
    tmp_path,
):
    output = tmp_path / "confirm"
    report_names = (
        "qualification_report.json",
        "detail_metrics.json",
        "pose_strata.json",
        "experiment_report.json",
    )
    for index, name in enumerate(report_names):
        run_training.write_json_record(output / name, {"version": index})
    render = output / "validation_renders" / "camera.JPG"
    render.parent.mkdir(parents=True)
    render.write_bytes(b"render-15k")
    recovery = output / "checkpoints" / "recovery.pt"
    recovery.parent.mkdir(parents=True)
    recovery.write_bytes(b"model-state")
    args = _generic_args(
        "confirm",
        "E1-density-absgrad-t04-v1",
        output_dir=str(output),
        max_steps=30_000,
        stop_step=15_000,
        checkpoint_every=3_000,
        rolling_checkpoint=True,
    )

    snapshot = run_training.preserve_confirmation_snapshot(args)

    assert snapshot == output / "snapshots" / "step_000015000"
    assert sorted(path.name for path in snapshot.iterdir()) == [
        "detail_metrics.json",
        "experiment_report.json",
        "pose_strata.json",
        "qualification_report.json",
        "validation_renders",
    ]
    for index, name in enumerate(report_names):
        assert run_training.read_json_record(snapshot / name) == {"version": index}
    assert (snapshot / "validation_renders" / "camera.JPG").read_bytes() == b"render-15k"
    assert not list(snapshot.rglob("*.pt"))

    preserved_files = {
        path.relative_to(snapshot).as_posix(): path.read_bytes()
        for path in snapshot.rglob("*")
        if path.is_file()
    }
    for index, name in enumerate(report_names, start=30):
        run_training.write_json_record(output / name, {"version": index})
    render.write_bytes(b"render-30k")
    resumed_args = _args(**(vars(args) | {"stop_step": 30_000}))

    assert run_training.preserve_confirmation_snapshot(resumed_args) is None
    assert {
        path.relative_to(snapshot).as_posix(): path.read_bytes()
        for path in snapshot.rglob("*")
        if path.is_file()
    } == preserved_files


@pytest.mark.parametrize(
    "stage,stop_step",
    (("confirm", 30_000), ("screen", 7_000)),
)
def test_non_15k_confirmation_keeps_reports_at_run_root_only(
    tmp_path, stage, stop_step
):
    output = tmp_path / stage
    output.mkdir()
    args = _generic_args(
        stage,
        (
            "E1-density-absgrad-t04-v1"
            if stage != "reference"
            else "B0-reference"
        ),
        output_dir=str(output),
        max_steps=30_000 if stage == "confirm" else 7_000,
        stop_step=stop_step,
        checkpoint_every=3_000,
        rolling_checkpoint=stage == "confirm",
    )

    assert run_training.preserve_confirmation_snapshot(args) is None
    assert not (output / "snapshots").exists()
