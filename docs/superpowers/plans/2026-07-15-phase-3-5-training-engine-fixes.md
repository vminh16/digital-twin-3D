# Phase 3.5 Training Engine Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Phase 3.5 training loop mathematically consistent, reproducible, finite-safe, and honestly tested.

**Architecture:** Keep the Phase 2 manifest in raw COLMAP coordinates and normalize each sampled camera only at the trainer boundary, where Gaussian means are already normalized. Keep one training horizon in config, enforce undistorted pinhole samples, and bind checkpoints to the complete manifest artifact and optimization config.

**Tech Stack:** Python 3.10, NumPy, PyTorch, pytest, gsplat 1.4.0.

## Global Constraints

- Work directly on `main` as explicitly requested by the user.
- Keep Phase 3.5 minimal; do not add unrelated data-loader, preview, or validation features.
- Use test-first red/green cycles for every behavior change.
- Run tests with `.venv\Scripts\python.exe -m pytest`.

---

### Task 1: Camera-domain and device invariants

**Files:**
- Modify: `src/bts_nvs/training/trainer.py`
- Test: `tests/unit/test_trainer_loop.py`

**Interfaces:**
- Consumes: `SceneDataset.manifest.normalization_transform`, raw `CameraSample.world_to_camera`.
- Produces: normalized rigid W2C passed to `render_gaussians`; device-local `JointLoss`; pinhole-only training samples.

- [x] Add tests proving projection-preserving normalized W2C, rejection of distorted samples, and loss buffers on the trainer device.
- [x] Run the focused tests and confirm they fail for the expected reasons.
- [x] Implement the minimum normalization, pinhole validation, and `.to(device)` changes.
- [x] Run focused tests and the Phase 3.1–3.4 regression set.

### Task 2: Determinism, finite safety, and one training horizon

**Files:**
- Modify: `src/bts_nvs/training/trainer.py`
- Test: `tests/unit/test_trainer_loop.py`

**Interfaces:**
- Consumes: required positive integer `config["max_steps"]`, optional integer `config["seed"]`.
- Produces: deterministic fresh runs, one-based completed-step logs, and fail-fast non-finite checks.

- [x] Add failing tests for seeded camera sampling, mismatched horizons, empty datasets, invalid checkpoint intervals, and non-finite loss/gradients.
- [x] Implement seed initialization, argument validation, canonical completed-step semantics, and finite checks before optimizer updates.
- [x] Verify focused tests pass and JSON outputs reject NaN/Infinity.

### Task 3: Complete checkpoint identity and real resume test

**Files:**
- Modify: `src/bts_nvs/training/checkpoint.py`
- Modify: `src/bts_nvs/training/trainer.py`
- Test: `tests/unit/test_trainer_loop.py`

**Interfaces:**
- Produces: SHA-256 over `manifest.json` plus referenced `arrays.npz`, a deterministic config hash, and checkpoint validation of both hashes.

- [x] Add failing tests proving NPZ changes alter the artifact hash and config changes reject resume.
- [x] Replace module-scope monkeypatches with scoped pytest monkeypatches and a differentiable renderer whose output depends on Gaussian parameters.
- [x] Verify continuous training equals a fresh-trainer split/resume run with actual parameter updates.
- [x] Run the complete test suite in more than one relevant collection order.

### Task 4: Final verification and commit

**Files:**
- Review all Phase 3.5 changes and this plan.

- [x] Run Phase 3 tests, then the full suite from a fresh command.
- [x] Inspect `git diff --check`, status, and the final diff for unrelated changes.
- [x] Commit the completed Phase 3.5 fix directly to `main` with one focused commit.
