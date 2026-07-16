# Phase 4.3 Holdout and Profile Gate Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Phase 4 profiling use leakage-controlled internal-train data, prepare manifest/holdout artifacts in batch, and compare cached/uncached optimization only before the first topology mutation.

**Architecture:** Add one batch preparation CLI around existing manifest and holdout APIs. Extend the existing training CLI with one `--internal_holdout` switch; profile mode implies it, loads the colocated holdout, subsets `SceneDataset`, and builds split-specific sparse initialization. Keep the 500-step profiler unchanged for timing but add an explicit pre-refinement equivalence prefix to its report/comparator.

**Tech Stack:** Python 3.12, NumPy 1.26.4, PyTorch, pycolmap 4.1.0, pytest.

## Global Constraints

- Guard and validation cameras never enter the internal qualification sampler.
- Normal training without `--internal_holdout` continues to use all physical train images.
- `--profile_input` requires a valid colocated `holdout.json` and implies internal holdout.
- Sparse support and colors in internal mode use internal-train observations only.
- Performance still measures exactly 50 warm-up plus 500 measured steps.
- Sample index sequences match for all 500 measured steps; loss/count equivalence stops before `refine_start_step`.
- Do not change optimizer, loss, renderer, density strategy, batch size, dependencies, or manifest schema.
- Commit directly to `main`; preserve the user's notebook and profile archives.

---

### Task 1: Batch manifest and holdout preparation

**Files:**
- Create: `src/bts_nvs/data/prepare_phase4_artifacts.py`
- Create: `tests/unit/test_prepare_phase4_artifacts.py`

**Interfaces:**
- Produces `prepare_scene_artifacts(scene_root: Path, artifact_dir: Path) -> tuple[Path, Path]`.
- CLI consumes `--scenes_root`, `--manifests_root`, optional `--expected_scenes`, and `--require_expected`.

- [ ] Write RED tests that monkeypatch production builders/loaders and assert sorted scene traversal, missing manifest creation, existing artifact validation, canonical `holdout.json`, and strict expected-count failure before partial generation.
- [ ] Run `pytest tests/unit/test_prepare_phase4_artifacts.py -q`; expect import failure because the module does not exist.
- [ ] Implement the minimal scanner and per-scene function. A scene is a direct child containing `train/images`; its artifact directory is `<manifests_root>/<scene_name>`. Existing manifests and holdouts are loaded/validated, never silently overwritten.
- [ ] Run `pytest tests/unit/test_prepare_phase4_artifacts.py tests/unit/test_holdout.py tests/unit/test_manifest_serialization.py -q`; expect all selected tests to pass.

### Task 2: Internal holdout training integration

**Files:**
- Modify: `src/bts_nvs/training/run_training.py`
- Modify: `tests/unit/test_run_training.py`

**Interfaces:**
- Add CLI `--internal_holdout`.
- Add `load_internal_holdout(manifest_dir, manifest, enabled) -> HoldoutSplit | None`.
- Extend `build_training_config(..., split: HoldoutSplit | None = None)` with immutable holdout identity/count fields.

- [ ] Write RED tests proving profile mode enables holdout, missing/stale holdout fails before training, config records algorithm/hash/counts, and normal mode remains full-data.
- [ ] Run `pytest tests/unit/test_run_training.py -q`; expect failures for the missing interface and config fields.
- [ ] Load `<manifest_dir>/holdout.json` whenever `args.internal_holdout or args.profile_input`. Pass `split.train_image_names` to `SceneDataset`; calculate cache preflight only for those selected indices.
- [ ] In internal mode call `build_split_sparse_initialization`, use `dataclasses.replace` only for Gaussian initialization points/colors, and keep the dataset/manifest artifact validation on the original manifest.
- [ ] Run `pytest tests/unit/test_run_training.py tests/unit/test_dataset.py tests/unit/test_sparse_subset.py tests/unit/test_trainer_loop.py -q`; expect all selected tests to pass.

### Task 3: Pre-refinement profile equivalence

**Files:**
- Modify: `src/bts_nvs/training/trainer.py`
- Modify: `src/bts_nvs/training/profiling.py`
- Modify: `tests/unit/test_training_profiling.py`
- Modify: `tests/unit/test_trainer_loop.py`

**Interfaces:**
- Profile JSON adds `equivalence_steps: int` and `final_gaussian_count_delta` appears in comparison output.
- `equivalence_steps = clamp(refine_start_step - (warmup_steps + 1), 1, measured_steps)`.

- [ ] Write RED comparator tests using the real failure pattern: identical 500 indices, allclose pre-refinement losses/counts, and a one-Gaussian divergence at refinement must pass trace equivalence while reporting the final delta. A difference before the boundary must fail.
- [ ] Run `pytest tests/unit/test_training_profiling.py -q`; expect schema/interface failures.
- [ ] Record `equivalence_steps` from trainer config. Validate both profiles use the same positive boundary. Compare all sample indices, but slice losses/counts to that prefix; retain `rtol=1e-4, atol=1e-6`.
- [ ] Run `pytest tests/unit/test_training_profiling.py tests/unit/test_trainer_loop.py -q`; expect all selected tests to pass.

### Task 4: Real-scene regression and handoff

**Files:**
- Modify only if required by failing tests from Tasks 1–3.

- [ ] Run artifact preparation on the currently available scene into a temporary workspace path; verify HCM0181 produces a valid manifest and holdout with 169 train, 46 guard, and 25 validation images.
- [ ] Run `pytest -q`, `python -m compileall -q src`, both changed CLI `--help` commands, and `git diff --check`.
- [ ] Red-review that profile mode cannot sample guard/validation, normal mode still supports full-data production retrain, sparse colors are split-specific, and the comparator cannot hide pre-refinement divergence.
- [ ] Stage only Phase 4.3 fix files and commit `fix: enforce phase 4 holdout profiling`.
- [ ] Provide exact batch preparation, uncached profile, cached profile, and comparison commands for the L4 VM. Do not claim the L4 gate passes until rerun artifacts return `accepted: true`.

## Plan self-review

- Every approved design requirement maps to a task and test.
- Batch preparation reuses production manifest/holdout APIs and adds no sparse serialization schema.
- Guard remains qualification-only; full-data production mode is explicit and unchanged.
- The profiler still times density refinement, but semantic equivalence cannot be invalidated solely by its nondeterministic topology threshold.
- No placeholders, alternate parser, or unbounded configuration surface remains.
