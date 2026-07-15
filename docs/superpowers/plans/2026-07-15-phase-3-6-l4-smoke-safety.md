# Phase 3.6 L4 Smoke Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Phase 3.6 fail-fast on invalid GPU environments and produce reproducible, measurable HCM0181 convergence evidence on an NVIDIA L4.

**Architecture:** Keep training mathematics in the existing Trainer. The CLI owns CUDA preflight, preprocessing identity, output safety, and orchestration; Trainer exposes one fixed train-view diagnostic using the same renderer, camera normalization, valid mask, and SSIM implementation as training.

**Tech Stack:** Python, PyTorch CUDA, gsplat 1.4.0, NumPy, Pillow, pytest.

## Global Constraints

- Do not use test images or test poses for optimization or convergence checks.
- Real-data smoke tests must never substitute a mock renderer.
- Run A is factor 4 for 500 steps; Run B is factor 2 for 7,000 steps.
- Keep code minimal and commit directly to `main` after full verification.

---

### Task 1: Runtime and run identity guards

- [x] Add failing unit tests for CUDA preflight, invalid resize factors, preprocessing config fields, and non-resume output reuse.
- [x] Implement fail-fast CUDA/gsplat preflight before scene initialization.
- [x] Bind scene, resize, undistortion, and seed to the checkpointed config.
- [x] Reject non-empty output directories unless `--resume` is supplied.

### Task 2: Honest real-data smoke test

- [x] Remove CPU/mock fallback and module-global mutation from the HCM0181 integration test.
- [x] Mark the test as real-data/CUDA and skip clearly when prerequisites are absent.
- [x] Build temporary manifest artifacts instead of mutating the source dataset.

### Task 3: Convergence and resource evidence

- [x] Add failing tests for masked train-view PSNR/SSIM and preview output.
- [x] Render the same fixed train camera before and after training.
- [x] Save reference/initial/final PNGs plus deterministic convergence JSON.
- [x] Reset and record device-specific peak VRAM and synchronized total wall time.

### Task 4: L4 deployment contract

- [x] Pin/document a CUDA-enabled PyTorch base and required GPU architecture.
- [x] Make the generic Docker smoke fail when CUDA/gsplat is unavailable.
- [x] Add exact L4 commands and pass/fail thresholds for Run A and Run B.

### Task 5: Verification and commit

- [x] Run focused RED/GREEN tests, Phase 3 regressions, full CPU suite, and static checks.
- [x] Review the final diff for leakage and unnecessary complexity.
- [x] Commit the Phase 3.6 safety changes directly to `main`.
