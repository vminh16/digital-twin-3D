# C1 AbsGrad × Revised Opacity Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and verify the four-run, two-scene 7k Phase A screen for AbsGrad and revised opacity without changing the closed B0 or production 30k contracts.

**Architecture:** Reuse the existing qualification training path and add only candidate configuration plus a Python Phase A orchestrator. Keep model flags, high-frequency diagnostics, and Phase A decision logic in focused modules; do not add a shell runner or checkpoint persistence for 7k runs.

**Tech Stack:** Python 3.11, PyTorch, gsplat 1.4.0, NumPy, OpenCV, Pillow, pytest.

## Global Constraints

- `B0-submission-q99-v1` remains CLOSED and immutable.
- Phase A scenes are exactly `HCM0421` and `HCM1439`.
- Candidates are exactly `C1-absgrad-t08-v1` and `C1-absgrad-t08-revopacity-v1`.
- Every Phase A run is fresh, factor 1, seed 0, 7,000 steps, internal holdout, cached images, pinned transfer, and no model checkpoint.
- Both candidates use `absgrad=true` and `grow_grad2d=0.0008`; only the combined candidate uses `revised_opacity=true`.
- Optimizer, precision, loss, SH schedule, remaining density settings, renderer mode, and dependencies remain unchanged.
- The accepted backend decision is reused; AMP must unscale both signed projected gradients and AbsGrad projected gradients.
- Full validation metrics and stored-render high-frequency diagnostics are durable; no official test data is used.
- No new bash script is added.

---

### Task 1: Candidate configuration and gsplat propagation

**Files:**
- Create: `src/bts_nvs/training/c1_candidates.py`
- Modify: `src/bts_nvs/rendering/gsplat_renderer.py`
- Modify: `src/bts_nvs/rendering/density_strategy.py`
- Modify: `src/bts_nvs/training/precision.py`
- Modify: `src/bts_nvs/training/trainer.py`
- Modify: `src/bts_nvs/training/run_training.py`
- Test: `tests/unit/test_c1_candidates.py`
- Test: `tests/unit/test_renderer.py`
- Test: `tests/unit/test_strategy.py`
- Test: `tests/unit/test_training_precision.py`
- Test: `tests/unit/test_run_training.py`

**Interfaces:**
- Produces: `C1_CANDIDATES`, `QUALIFICATION_CANDIDATES`, and `candidate_settings(candidate_id: str | None) -> CandidateSettings`.
- Produces: `render_gaussians(..., absgrad: bool = False) -> RenderResult`.
- Produces: `GsplatStrategy(..., absgrad: bool = False, revised_opacity: bool = False)`.
- Preserves: B0 and ordinary defaults `grow_grad2d=0.0002`, `absgrad=False`, `revised_opacity=False`.

- [x] **Step 1: Write failing candidate/configuration tests**

```python
def test_c1_candidate_settings_are_exact():
    plain = candidate_settings("C1-absgrad-t08-v1")
    revised = candidate_settings("C1-absgrad-t08-revopacity-v1")
    assert (plain.grow_grad2d, plain.absgrad, plain.revised_opacity) == (0.0008, True, False)
    assert (revised.grow_grad2d, revised.absgrad, revised.revised_opacity) == (0.0008, True, True)

def test_training_config_records_c1_density_settings():
    config = run_training.build_training_config(
        _args(qualification_candidate="C1-absgrad-t08-revopacity-v1"),
        SimpleNamespace(scene_id="HCM0421"),
        resize=(8, 6),
    )
    assert config["grow_grad2d"] == pytest.approx(0.0008)
    assert config["absgrad"] is True
    assert config["revised_opacity"] is True
```

- [x] **Step 2: Run the focused tests and confirm RED**

Run: `python -m pytest tests/unit/test_c1_candidates.py tests/unit/test_run_training.py -q`

Expected: FAIL because `c1_candidates` and C1 choices/settings do not exist.

- [x] **Step 3: Implement the immutable candidate mapping and config lookup**

```python
@dataclass(frozen=True)
class CandidateSettings:
    grow_grad2d: float
    absgrad: bool
    revised_opacity: bool

C1_CANDIDATES = ("C1-absgrad-t08-v1", "C1-absgrad-t08-revopacity-v1")
QUALIFICATION_CANDIDATES = ("B0-reference", "B0-compact", *C1_CANDIDATES)
```

Use this mapping in argparse choices and `build_training_config`; keep the existing qualification validation so C1 inherits the locked 7k/no-resume contract.

- [x] **Step 4: Write failing renderer, strategy, trainer, preflight, and AMP tests**

```python
def test_renderer_forwards_absgrad(monkeypatch):
    # fake rasterization captures kwargs
    render_gaussians(..., absgrad=True)
    assert captured["absgrad"] is True

def test_strategy_forwards_absgrad_and_revised_opacity(monkeypatch):
    strategy = GsplatStrategy(..., grow_grad2d=0.0008, absgrad=True, revised_opacity=True)
    assert strategy.backend.config["absgrad"] is True
    assert strategy.backend.config["revised_opacity"] is True

def test_amp_unscales_projected_absgrad():
    projected = torch.tensor([1.0], requires_grad=True)
    projected.absgrad = torch.tensor([1024.0])
    # fake scaler scale is 1024
    precision.backward_and_unscale(loss, optimizers, projected)
    assert projected.absgrad.item() == pytest.approx(1.0)
```

- [x] **Step 5: Run the propagation tests and confirm RED**

Run: `python -m pytest tests/unit/test_renderer.py tests/unit/test_strategy.py tests/unit/test_training_precision.py tests/unit/test_trainer_loop.py tests/unit/test_run_training.py -q`

Expected: FAIL because the new flags are not accepted or propagated and AMP does not unscale `.absgrad`.

- [x] **Step 6: Implement minimal propagation and AbsGrad unscale**

Pass `config["absgrad"]` to training rasterization and both flags to `GsplatStrategy`. Pass the same settings through CUDA preflight. In AMP mode only, divide `projected_means.absgrad` by the active loss scale when that tensor attribute exists; leave FP32 unchanged.

- [x] **Step 7: Run Task 1 tests and commit**

Run: `python -m pytest tests/unit/test_c1_candidates.py tests/unit/test_renderer.py tests/unit/test_strategy.py tests/unit/test_training_precision.py tests/unit/test_trainer_loop.py tests/unit/test_run_training.py -q`

Expected: PASS.

Commit: `feat: add C1 AbsGrad candidate configuration`

---

### Task 2: Stored-render high-frequency diagnostics

**Files:**
- Create: `src/bts_nvs/evaluation/high_frequency.py`
- Test: `tests/unit/test_high_frequency.py`

**Interfaces:**
- Produces: `high_frequency_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]`.
- Produces: `evaluate_render_directory(dataset, render_dir: Path) -> dict`.
- Consumes: undistorted `SceneDataset` validation samples and the existing PNG validation renders.

- [x] **Step 1: Write failing mathematical metric tests**

```python
def test_identical_images_have_zero_high_frequency_error():
    image = np.zeros((32, 32, 3), dtype=np.float32)
    assert high_frequency_metrics(image, image) == {
        "hf_l1": 0.0, "missing_edge": 0.0, "spurious_edge": 0.0
    }

def test_blurred_edge_increases_missing_edge():
    target = vertical_step_image()
    blurred = cv2.GaussianBlur(target, (9, 9), 2.0)
    assert high_frequency_metrics(blurred, target)["missing_edge"] > 0.0

def test_noise_in_flat_region_increases_spurious_edge():
    target = vertical_step_image()
    noisy = target.copy(); noisy[:8, :8] = checkerboard()
    assert high_frequency_metrics(noisy, target)["spurious_edge"] > 0.0
```

- [x] **Step 2: Run metric tests and confirm RED**

Run: `python -m pytest tests/unit/test_high_frequency.py -q`

Expected: FAIL because the module does not exist.

- [x] **Step 3: Implement the fixed diagnostic equations**

Convert RGB to `Y=0.299R+0.587G+0.114B`. Compute Sobel magnitudes, use the target 90th percentile as edge mask and 50th percentile as flat mask, and normalize missing/spurious excess by `max(mean(target_gradient[edge_mask]), 1e-12)`. Define `HF-L1` as the mean absolute difference between target and prediction Laplacians. Validate equal shape, RGB layout, finite values, and `[0,1]` range.

- [x] **Step 4: Write failing stored-render tests**

```python
def test_render_directory_matches_dataset_names_and_masks_invalid_pixels(tmp_path):
    # one CameraSample with an invalid border and one matching PNG
    result = evaluate_render_directory(dataset, tmp_path)
    assert result["image_count"] == 1
    assert set(result["images"]) == {"validation.JPG"}

def test_render_directory_rejects_missing_or_colliding_png_names(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        evaluate_render_directory(dataset, tmp_path)
```

- [x] **Step 5: Implement stored-render evaluation and verify Task 2**

Load each PNG as RGB float `[0,1]`, require exact target resolution, replace invalid-mask pixels with target pixels, compute per-image metrics, and return finite means plus image records.

Run: `python -m pytest tests/unit/test_high_frequency.py -q`

Expected: PASS.

Commit: `feat: add high-frequency render diagnostics`

---

### Task 3: Phase A paired decision

**Files:**
- Create: `src/bts_nvs/training/c1_phase_a.py`
- Test: `tests/unit/test_c1_phase_a.py`

**Interfaces:**
- Produces: `score50(report: Mapping[str, object]) -> float`.
- Produces: `build_phase_a_decision(baseline_reports, candidate_reports, diagnostics) -> dict`.
- Produces: `save_phase_a_decision(decision: dict, path: Path) -> None`.

- [x] **Step 1: Write failing decision tests**

```python
def test_score50_matches_locked_formula():
    report = {"psnr_db_mean": 25.0, "ssim_mean": 0.8, "lpips_mean": 0.2}
    assert score50(report) == pytest.approx(71.0)

def test_candidate_must_win_both_scenes_and_not_worsen_both_edge_errors():
    decision = build_phase_a_decision(...)
    assert decision["selected_candidate"] == "C1-absgrad-t08-v1"
    assert decision["phase_a_passed"] is True

def test_incomplete_duplicate_or_nonfinite_matrix_is_rejected():
    with pytest.raises(ValueError):
        build_phase_a_decision(...)
```

- [x] **Step 2: Run decision tests and confirm RED**

Run: `python -m pytest tests/unit/test_c1_phase_a.py -q`

Expected: FAIL because Phase A decision logic does not exist.

- [x] **Step 3: Implement exact paired gates and deterministic JSON**

Require B0-reference plus both C1 candidates for both locked scenes. Compute per-scene `delta_score50`; candidate eligibility requires positive deltas on both scenes and forbids both missing-edge and spurious-edge means worsening on either scene. Select largest mean delta, then lower mean HF-L1, lower mean peak Gaussian count, then the AbsGrad-only candidate. Save with sorted keys, finite JSON, atomic replacement.

- [x] **Step 4: Verify Task 3 and commit**

Run: `python -m pytest tests/unit/test_c1_phase_a.py -q`

Expected: PASS.

Commit: `feat: add deterministic C1 Phase A decision`

---

### Task 4: Python Phase A orchestrator

**Files:**
- Create: `src/bts_nvs/training/c1_phase_a_runner.py`
- Create: `src/bts_nvs/training/run_c1_phase_a.py`
- Test: `tests/unit/test_c1_phase_a_runner.py`

**Interfaces:**
- Produces: `build_phase_a_command(...) -> list[str]` using the existing `run_training.py` entry point.
- Produces: `run_phase_a(...) -> dict` that returns and writes the Phase A decision.
- CLI requires repo, scenes, manifests, backend qualification, existing B0 qualification, and new output roots.

- [x] **Step 1: Write failing command and orchestration tests**

```python
def test_phase_a_command_is_locked_and_checkpoint_free():
    command = build_phase_a_command(...)
    assert command contains factor 1, max steps 7000, seed 0, cache, pinned transfer
    assert "--rolling_checkpoint" not in command and "--resume" not in command

def test_runner_executes_exact_four_pairs_and_reuses_completed_reports():
    run_phase_a(..., run_process=fake_process)
    assert observed_pairs == [
        ("HCM0421", "C1-absgrad-t08-v1"),
        ("HCM0421", "C1-absgrad-t08-revopacity-v1"),
        ("HCM1439", "C1-absgrad-t08-v1"),
        ("HCM1439", "C1-absgrad-t08-revopacity-v1"),
    ]

def test_runner_rejects_nonempty_incomplete_run_directory():
    with pytest.raises(ValueError, match="non-empty"):
        run_phase_a(...)
```

- [x] **Step 2: Run runner tests and confirm RED**

Run: `python -m pytest tests/unit/test_c1_phase_a_runner.py -q`

Expected: FAIL because the runner modules do not exist.

- [x] **Step 3: Implement the minimal Python runner**

Reuse `load_or_create_backend_decision`, `load_scene_manifest`, `load_holdout_split`, `SceneDataset`, existing qualification reports/renders, and `run_training.py`. Run sequentially, skip only a valid complete report, reject partial directories, compute stored-render diagnostics for B0 and C1 with the same path, then write `phase_a_decision.json`. Do not prepare manifests, run inference, modify B0 artifacts, or launch Phase B.

- [x] **Step 4: Verify Task 4 and all Phase A tests**

Run: `python -m pytest tests/unit/test_c1_phase_a_runner.py tests/unit/test_c1_phase_a.py tests/unit/test_high_frequency.py tests/unit/test_c1_candidates.py tests/unit/test_renderer.py tests/unit/test_strategy.py tests/unit/test_training_precision.py tests/unit/test_trainer_loop.py tests/unit/test_run_training.py -q`

Expected: PASS.

- [x] **Step 5: Run full regression and compile checks**

Run: `python -m compileall -q src tests`

Expected: exit 0.

Run: `python -m pytest -q`

Expected: all available tests pass; CUDA-only tests may remain skipped on the local CPU environment.

Commit: `feat: orchestrate C1 Phase A screen`

---

## Self-review

- Spec coverage: candidate identity, 7k fairness, no checkpoints, same backend, high-frequency metrics, paired gates, and four-run orchestration each map to a task above.
- Deliberately excluded: Phase B six-scene gate, Phase C 30k mode, Phase D production retraining, inference, codec, dependency changes, and new bash scripts.
- Type consistency: candidate IDs/settings feed config, config feeds renderer/strategy/preflight, stored-render diagnostics feed the Phase A decision, and the runner writes that decision.
- Placeholder scan: no deferred implementation marker is present.
