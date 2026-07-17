# Phase 4.7 Full-Training Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe sequential runner that trains all 18 canonical scenes for 30,000 full-resolution steps with atomic latest-checkpoint resume and a deterministic cohort ledger.

**Architecture:** Keep scene optimization in the existing `run_training.py`. Add one generic rolling-checkpoint CLI flag, one focused Python orchestration module for qualification/pool/ledger/artifact rules, one thin Python CLI, and one path-only Bash wrapper.

**Tech Stack:** Python 3.12, PyTorch, PyYAML, Pillow, pytest, Bash, JSON.

## Global Constraints

- Commit directly to `main` after fresh verification.
- Production trains all physical train images; no internal holdout or official-test RGB.
- Scene pool is exactly the 18 case-sensitive IDs locked in the design spec.
- Backend and precision come only from the accepted qualification report.
- Fixed run contract: factor 1, 30,000 steps, checkpoint every 3,000, seed 0, cached images, pinned transfer.
- Retain only `checkpoints/recovery.pt`; it is latest state, never `best_validation`.
- One GPU process at a time; first scene failure stops the cohort.
- Never delete, overwrite, or silently adopt ambiguous artifacts.

---

### Task 1: Generic rolling-checkpoint CLI

**Files:**
- Modify: `src/bts_nvs/training/run_training.py`
- Modify: `tests/unit/test_run_training.py`

**Interfaces:**
- Consumes: `--rolling_checkpoint` boolean.
- Produces: config key `rolling_checkpoint`; `Trainer.train(..., rolling_checkpoint=args.rolling_checkpoint or args.full_length_qualification)`.

- [ ] **Step 1: Write RED tests**

Add the default field to `_args`, assert it enters the hashed config, assert it is incompatible with profile/backend/candidate qualification modes, and assert ordinary production mode may enable it.

- [ ] **Step 2: Verify RED**

Run: `.venv\Scripts\python -m pytest tests/unit/test_run_training.py -q`

Expected: FAIL because rolling checkpoint CLI validation does not exist.

- [ ] **Step 3: Implement minimal plumbing**

Add argparse, `validate_rolling_checkpoint_args(args)`, config identity, main validation call, and trainer call wiring. Preserve the existing implicit rolling behavior for Phase 4.5.

- [ ] **Step 4: Verify GREEN**

Run the same test file; expected: all pass.

### Task 2: Qualification and canonical scene-pool validation

**Files:**
- Create: `src/bts_nvs/training/full_training.py`
- Create: `tests/unit/test_full_training.py`

**Interfaces:**
- Produces: `CANONICAL_SCENES`, `load_or_create_backend_decision(backend_root) -> BackendDecision`, `validate_scene_pool(scenes_root, manifests_root) -> tuple[str, ...]`.
- Consumes: existing `load_backend_profile`, `compare_backend_profiles`, `write_backend_comparison`.

- [ ] **Step 1: Write RED qualification tests**

Use valid compact profile fixtures to prove a missing aggregate is generated from three profiles, an existing mismatched aggregate fails, `accepted=false` fails, and selected backend values are returned without override.

- [ ] **Step 2: Write RED pool tests**

Create 18 scene/manifest directories with minimal manifest JSON plus referenced NPZ. Assert exact sorted IDs pass; missing, extra, wrong-case, missing NPZ, and mismatched manifest `scene_id` fail before training.

- [ ] **Step 3: Verify RED**

Run: `.venv\Scripts\python -m pytest tests/unit/test_full_training.py -q`

Expected: import failure for the new module.

- [ ] **Step 4: Implement decision and pool validation**

Use frozen `BackendDecision`, exact set equality, JSON schema checks, atomic aggregate creation through the existing writer, and SHA-256 of the accepted aggregate.

- [ ] **Step 5: Verify GREEN**

Run the same test file; expected: all pass.

### Task 3: Atomic ledger and run-artifact validation

**Files:**
- Modify: `src/bts_nvs/training/full_training.py`
- Modify: `tests/unit/test_full_training.py`

**Interfaces:**
- Produces: `load_or_create_ledger(...) -> dict`, `set_scene_status(...) -> None`, `validate_trained_run(...) -> TrainedRun`, `inspect_scene_run(...) -> Literal["fresh", "resume", "trained"]`.
- Consumes: `compute_config_sha256`, `compute_manifest_sha256`, `load_checkpoint`.

- [ ] **Step 1: Write RED ledger tests**

Assert stable sorted JSON, atomic rewrite, exact status enum, immutable backend/qualification identity, relative run directories, and transitions for pending/running/trained/failed.

- [ ] **Step 2: Write RED artifact tests**

Build a small real checkpoint plus 30,000 lightweight metric lines. Assert valid trained artifacts pass; wrong count/order/non-finite loss, wrong config, manifest hash mismatch, missing/corrupt preview, wrong checkpoint step, and nonblank convergence failure are rejected.

- [ ] **Step 3: Verify RED**

Run targeted tests; expected: missing ledger/artifact functions.

- [ ] **Step 4: Implement minimal validators**

Stream `metrics.jsonl` instead of loading it all, decode preview with Pillow, validate locked YAML scalars, and use existing checkpoint hash checks. `inspect_scene_run` returns fresh for an absent/empty directory, trained for complete artifacts, resume for a valid recovery, and raises for ambiguous non-empty content.

- [ ] **Step 5: Verify GREEN**

Run targeted tests; expected: all pass.

### Task 4: Sequential subprocess runner

**Files:**
- Modify: `src/bts_nvs/training/full_training.py`
- Create: `src/bts_nvs/training/run_full_training.py`
- Modify: `tests/unit/test_full_training.py`

**Interfaces:**
- Produces: `run_full_training(repo_root, scenes_root, manifests_root, backend_root, output_root, python_bin) -> None` and CLI flags for those paths.
- Consumes: decision, pool, ledger, scene inspection, and `run_training.py`.

- [ ] **Step 1: Write RED command/orchestration tests**

Inject a subprocess runner. Assert exact sorted order, locked CLI, no holdout flags, selected backend propagation, `--resume` only for recovery, trained skip, ledger transitions, and first non-zero result stops later scenes.

- [ ] **Step 2: Verify RED**

Run targeted tests; expected: orchestration APIs missing.

- [ ] **Step 3: Implement command builder and loop**

Build argument lists without shell interpolation. Mark running atomically before launch; on non-zero mark failed and raise; after zero validate artifacts before marking trained. Preserve `running` on `KeyboardInterrupt`.

- [ ] **Step 4: Add thin argparse entry point**

Resolve paths only in the CLI and delegate all rules to `run_full_training`.

- [ ] **Step 5: Verify GREEN**

Run `.venv\Scripts\python -m pytest tests/unit/test_full_training.py -q`; expected: all pass.

### Task 5: Bash wrapper and runbook

**Files:**
- Create: `scripts/run_phase4_full_training.sh`
- Modify: `tests/unit/test_phase4_shell_scripts.py`
- Modify: `docs/phase_4_spec.md`
- Modify: `docs/phase_3_6_l4_runbook.md`

**Interfaces:**
- Consumes: five path/runtime environment overrides from the design.
- Produces: one command that sets `PYTHONPATH=<repo>/src`, prepares artifacts, and invokes `python -m bts_nvs.training.run_full_training`.

- [ ] **Step 1: Write RED shell contract test**

Assert strict Bash mode, path overrides, `PYTHONPATH`, artifact preparation, full runner invocation, and absence of `rm`, backend overrides, optimization overrides, or embedded scene-loop logic.

- [ ] **Step 2: Verify RED**

Run shell-script unit tests; expected: missing script failure.

- [ ] **Step 3: Implement wrapper and docs**

Keep the wrapper path-only. Document start, resume, ledger inspection, disk requirements, stop behavior, and the distinction between `trained` and compact-export completion.

- [ ] **Step 4: Full verification and red review**

Run:

```powershell
.venv\Scripts\python -m pytest tests/unit -q
.venv\Scripts\python -m pytest tests/integration -q
.venv\Scripts\python -m compileall -q src tests
git diff --check
```

Review that official test RGB is unreachable, all 18 identities are exact, resume cannot cross config/manifest hashes, report generation does not rerun qualification, and checkpoint retention remains one file.

- [ ] **Step 5: Commit**

Commit as:

```text
feat: add sequential phase 4 full training runner
```

## L4 handoff

Run one-scene operational dry check only if a separate root is desired; otherwise start the resumable cohort with:

```bash
bash scripts/run_phase4_full_training.sh
```

Inspect:

```bash
cat runs/phase4/full_training/ledger.json
```

Do not declare Phase 4.7 complete until a later compact-export step validates all 18 trained artifacts.

