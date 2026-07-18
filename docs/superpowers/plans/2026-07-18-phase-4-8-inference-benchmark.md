# Phase 4.8 Inference and Local Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add leakage-safe test-pose inference into `outputs/<scene>` and an independent local benchmark CLI for explicit reference images.

**Architecture:** Put reusable model loading, camera-domain conversion and rendering in a focused inference module. A thin Python CLI owns batch staging and validation, a path-only Bash wrapper launches it, and a separate evaluation CLI composes the existing evaluator APIs.

**Tech Stack:** Python 3.12, PyTorch, gsplat 1.4.0, NumPy, OpenCV headless, Pillow, argparse, pytest, Bash.

## Global Constraints

- Inference never reads test RGB or computes metrics.
- Output is exactly `outputs/<scene_id>/<test_output_name>.png`.
- Test output uses native width/height and original distortion domain.
- The final output root is created atomically and never overwritten.
- Benchmark requires an explicit reference root and explicit `psnr_max`.
- Scene selection uses the existing canonical validation contract.

---

### Task 1: Model and camera inference primitives

**Files:**
- Create: `src/bts_nvs/rendering/inference.py`
- Test: `tests/unit/test_inference.py`

**Interfaces:**
- Produces: `gaussians_from_checkpoint(state, device) -> GaussianParameters`.
- Produces: `normalized_test_world_to_camera(raw_w2c, normalization) -> np.ndarray`.
- Produces: `redistort_render(image, intrinsics, distortion) -> np.ndarray`.
- Produces: `render_test_camera(gaussians, w2c, intrinsics, distortion, normalization, active_sh_degree) -> np.ndarray`.

- [ ] Write failing tests for checkpoint tensor shapes, normalized pose equivalence, PINHOLE identity, SIMPLE_RADIAL destination mapping, RGB range and render argument propagation.
- [ ] Run `.venv\Scripts\python -m pytest tests/unit/test_inference.py -q` and confirm RED.
- [ ] Implement only the four interfaces using existing camera and renderer primitives.
- [ ] Run the same tests and confirm GREEN.

### Task 2: Atomic selected-scene batch inference

**Files:**
- Create: `src/bts_nvs/rendering/run_inference.py`
- Modify: `tests/unit/test_inference.py`

**Interfaces:**
- Produces: `run_inference(..., scene_ids=None) -> dict` and an argparse entry point.

- [ ] Write failing tests for selected order, checkpoint/hash validation, exact names/resolutions, one-model-at-a-time behavior, existing output rejection, cleanup on failure, validator gate and metric-free report.
- [ ] Run the targeted tests and confirm RED.
- [ ] Implement a temporary sibling output tree, per-scene checkpoint loading/rendering, PNG encoding, final submission validation and atomic rename.
- [ ] Run the targeted tests and confirm GREEN.

### Task 3: Independent benchmark CLI

**Files:**
- Create: `src/bts_nvs/evaluation/run_benchmark.py`
- Create: `tests/unit/test_benchmark_cli.py`

**Interfaces:**
- Produces: `run_local_benchmark(...) -> dict` and an argparse entry point.

- [ ] Write failing tests for selected manifests, mirrored output/reference schema, explicit PSNR max, LPIPS backend injection, deterministic report and missing/extra image rejection.
- [ ] Run `.venv\Scripts\python -m pytest tests/unit/test_benchmark_cli.py -q` and confirm RED.
- [ ] Compose `load_image_pairs`, `evaluate_benchmark`, `MetricConfig`, `LpipsBackend`, and `save_metric_report` without importing rendering or checkpoints.
- [ ] Run the targeted tests and confirm GREEN.

### Task 4: Shell wrapper, docs and verification

**Files:**
- Create: `scripts/run_phase4_inference.sh`
- Modify: `tests/unit/test_phase4_shell_scripts.py`
- Modify: `docs/phase_3_6_l4_runbook.md`

**Interfaces:**
- The wrapper forwards `"$@"` and only supplies path arguments.

- [ ] Add a failing shell contract test for `outputs` default, path overrides, module invocation, argument forwarding and absence of metric/reference flags.
- [ ] Implement the wrapper and document inference and benchmark commands.
- [ ] Run unit tests, integration tests, compileall and `git diff --check`.
- [ ] Red-review pose normalization, distortion direction, checkpoint hashes, output transaction and official-test leakage boundary.
- [ ] Commit as `feat: add phase 4 test inference and benchmark cli`.
