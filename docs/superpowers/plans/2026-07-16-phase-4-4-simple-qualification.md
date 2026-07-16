# Phase 4.4 Simple Qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize the local dataset layout and provide a minimal, leakage-controlled two-candidate qualification workflow for six fixed BTS scenes.

**Architecture:** Move all 18 BTS scenes under one flat `data/bts_scenes` root and isolate `bonsai/chair` under `data/auxiliary`. Keep per-scene training unchanged; add only candidate selection, final internal-validation evaluation, and pure report aggregation around the existing trainer and metric implementation.

**Tech Stack:** Python 3.12, NumPy, PyTorch, gsplat 1.4.0, Pillow/OpenCV, pytest.

## Global Constraints

- Train one independent Gaussian model per scene; never merge scene coordinate systems.
- Calibration scenes are fixed before reading validation metrics: `hcm0031`, `HCM0181`, `HCM0421`, `HCM1439`, `HNI0131`, `HNI0265`.
- Candidates are exactly `B0-reference: 0.0002` and `B0-compact: 0.0003`.
- Qualification uses factor 1, seed 0, 7,000 steps, cached images, pinned transfer, and internal holdout.
- Only internal validation RGB contributes to PSNR/SSIM/LPIPS; guard and official test poses do not.
- Invalid undistortion pixels are replaced by their target values before metrics; valid fraction is reported.
- Candidate decision uses the existing quality/resource rule and falls back to `B0-reference`.
- Preserve unrelated user deletions and do not commit generated manifests or model artifacts.

---

### Task 1: Canonical dataset layout

**Files:**
- Move: 18 BTS scene directories to `data/bts_scenes/<scene_id>`
- Move: `data/bonsai`, `data/chair` to `data/auxiliary/`
- Create: `data/scene_sources.json`
- Modify: active runbook and real-data integration test paths

- [ ] Verify all source and destination paths remain inside the workspace and scene IDs are unique.
- [ ] Move directories without copying or renaming scene IDs.
- [ ] Record source provenance and validate 18 BTS plus 2 auxiliary scenes.
- [ ] Update active paths and run real-data contract tests.

### Task 2: Adaptive holdout for the 103-image scene

**Files:**
- Modify: `src/bts_nvs/data/holdout.py`
- Modify: `src/bts_nvs/data/inventory.py`
- Test: `tests/unit/test_holdout.py`, `tests/unit/test_inventory.py`

- [ ] Add a failing test proving 103 images can retain at least 70% internal train and 8 validation images.
- [ ] Remove the absolute 120-train requirement while retaining the 70% rule and guard.
- [ ] Lower inventory eligibility to the observed 100-image floor.
- [ ] Regenerate and validate HCM1439 holdout.

### Task 3: Locked qualification candidate CLI

**Files:**
- Modify: `src/bts_nvs/training/run_training.py`
- Test: `tests/unit/test_run_training.py`

- [ ] Add failing tests for `--qualification_candidate` and fixed qualification invariants.
- [ ] Map the two candidate IDs to their locked `grow_grad2d` values.
- [ ] Make qualification imply internal holdout and reject wrong horizon/resolution/seed/cache settings.
- [ ] Keep ordinary production training behavior unchanged.

### Task 4: Internal-validation report and decision

**Files:**
- Create: `src/bts_nvs/training/qualification.py`
- Create: `src/bts_nvs/training/decide_qualification.py`
- Modify: `src/bts_nvs/training/trainer.py`, `src/bts_nvs/training/run_training.py`
- Test: `tests/unit/test_qualification.py`, `tests/unit/test_trainer_loop.py`

- [ ] Add failing tests for validation rendering, invalid-mask handling, report validation, and candidate decision.
- [ ] Render every internal-validation camera once after step 7,000 and reuse `evaluate_image` with one LPIPS backend.
- [ ] Save deterministic per-image/per-scene metrics plus peak Gaussian/VRAM/time data.
- [ ] Aggregate exactly 12 reports from the six locked scenes and select compact only when every quality bound and one resource bound pass.
- [ ] Add a thin CLI that writes the machine-readable decision JSON.

### Task 5: Verification and handoff

- [ ] Run the full test suite and CLI help smoke tests.
- [ ] Run artifact preparation on `data/bts_scenes` with expected count 18.
- [ ] Confirm no auxiliary scene enters the BTS artifact pool.
- [ ] Commit implementation directly to `main` and provide VM commands for the 12 qualification runs.
