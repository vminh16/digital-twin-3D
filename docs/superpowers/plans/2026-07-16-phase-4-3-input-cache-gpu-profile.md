# Phase 4.3 Input Cache and GPU Profiling Implementation Plan

> **Execution:** use `superpowers:executing-plans`; implement with TDD and commit directly to `main` as explicitly approved by the user.

**Goal:** Remove repeated image preprocessing from the training hot path and produce controlled, reproducible evidence that the cached path improves HCM0181 factor-1 throughput without changing optimization behavior.

**Architecture:** Extend `SceneDataset` with an optional eager pageable-RAM cache containing contiguous `uint8` images and `bool` masks. Add a CUDA-only two-slot pinned transfer ring used by the existing trainer, without lookahead or sampler changes. Instrument the existing loop with a fixed 50-step warm-up and 500-step measurement window, then compare independent cached and uncached runs using their recorded sample indices, losses, Gaussian counts and timings. Keep resource checks in one small training utility and reuse them before cache allocation and checkpoints.

**Constraints:** No batch-size, optimizer, density strategy, loss, renderer or dependency changes. No pinned full-scene cache. No asynchronous sampler/lookahead. Preserve checkpoint/resume RNG semantics. Preserve the user's modified Phase 3 notebook and pilot archive.

---

## Task 1: Pageable image cache

**Files:**
- Modify: `src/bts_nvs/data/dataset.py`
- Modify: `tests/unit/test_dataset.py`

- [ ] Add RED tests proving cached and uncached samples are bit-identical, each source image is decoded once, returned samples cannot mutate cached state, cached arrays are contiguous `uint8`/`bool`, and insufficient RAM fails before decoding.
- [ ] Add `cache_images: bool = False`, exact cache-byte estimation, a 4 GiB post-allocation RAM gate, and eager preprocessing through one existing sample-building path.
- [ ] Return fresh `CameraSample` values from cached buffers so caller mutation cannot contaminate later samples.
- [ ] Run `pytest tests/unit/test_dataset.py -q`.

## Task 2: Two-slot pinned transfer ring

**Files:**
- Create: `src/bts_nvs/training/input_pipeline.py`
- Create: `tests/unit/test_input_pipeline.py`
- Modify: `src/bts_nvs/training/trainer.py`
- Modify: `tests/unit/test_trainer_loop.py`

- [ ] Add RED tests for exactly two reusable host slots, contiguous pinned `uint8`/`bool`/`float64` buffers, GPU-side image conversion to float32 `[0,1]`, and CPU fallback equivalence.
- [ ] Implement a minimal ring that waits for a slot's prior H2D event before overwriting it. Do not prefetch indices or pin scene-wide cache arrays.
- [ ] Route trainer sample conversion through the pipeline when `pinned_transfer=True`; preserve the current random draw location and order.
- [ ] Record sampled image index and data/transfer timing without changing update order.
- [ ] Run `pytest tests/unit/test_input_pipeline.py tests/unit/test_trainer_loop.py -q`.

## Task 3: Resource preflight and controlled profile records

**Files:**
- Create: `src/bts_nvs/training/resources.py`
- Create: `src/bts_nvs/training/profiling.py`
- Modify: `src/bts_nvs/training/run_training.py`
- Create: `src/bts_nvs/training/compare_input_profiles.py`
- Create: `tests/unit/test_training_resources.py`
- Create: `tests/unit/test_training_profiling.py`
- Modify: `tests/unit/test_run_training.py`

- [ ] Add RED tests for `/proc/meminfo` RAM/swap parsing, 4 GiB host headroom, `<20 GiB` peak allocated VRAM, and checkpoint disk requirement `2 * estimate + 10 GiB`.
- [ ] Check host RAM/swap before cache allocation and disk immediately before each checkpoint. Fail clearly; never delete artifacts.
- [ ] Add `--cache_images`, `--pinned_transfer`, and `--profile_input`. Profile mode requires a fresh 550-step run, records CUDA elapsed time with events, synchronizes only at profile-window boundaries, and skips large checkpoints.
- [ ] Write deterministic per-run `input_profile.json` containing the exact 50/500 window, mean wall step, median CUDA step, CPU preprocessing fraction, peak VRAM, loss trace, Gaussian counts and sampled indices.
- [ ] Implement a deterministic comparator that rejects mismatched indices/counts, applies `rtol=1e-4, atol=1e-6` to losses, and passes only when cached median step improves by at least 10% or cached preprocessing is below 10% of wall time.
- [ ] Run `pytest tests/unit/test_training_resources.py tests/unit/test_training_profiling.py tests/unit/test_run_training.py -q`.

## Task 4: Red review and phase gate

- [ ] Run the full local suite and compilation: `pytest -q`, `python -m compileall -q src`, and `git diff --check`.
- [ ] Review cache ownership, pinned-slot reuse safety, sampler/checkpoint RNG equivalence, CUDA timing boundaries, JSON finiteness and resource arithmetic.
- [ ] Commit only Phase 4.3 implementation files to `main`.
- [ ] Report the exact L4 commands for uncached/cached HCM0181 factor-1 profiling and comparison. Mark the runtime performance gate pending until those two real GPU runs pass; do not infer it from CPU tests.

## Plan self-review

- Every Phase 4.3 deliverable and acceptance criterion has a code path and test or an explicit L4 runtime gate.
- The ring does not alter sample order, global RNG state, batch size or optimization semantics.
- Cache memory remains pageable and compact; only two samples are pinned.
- Profiling reuses the production trainer instead of duplicating an optimization loop.
- Local completion cannot falsely claim the HCM0181/L4 performance acceptance.
