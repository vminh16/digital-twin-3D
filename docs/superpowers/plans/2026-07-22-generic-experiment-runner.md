# Generic Experiment Runner and Decision Engine Implementation Plan

**Goal:** Implement Module 3 as a reusable, stage-first orchestration and
decision layer without duplicating the training loop.

**Design authority:**
`docs/superpowers/specs/2026-07-22-generic-experiment-runner-design.md`

**Execution rule:** Work directly on `main`, use TDD, commit each task
separately, and never stage user-owned `.gitignore` or `docs/research/` changes.

## Task 1: Experiment schema and stage-first paths

**Files:**
- Create: `src/bts_nvs/experiments/experiment.py`
- Create: `tests/unit/test_experiment_schema.py`

- [x] Write failing tests for the seven-scene cohort, stage enum, locked
  horizons, resource limits, legal stage/candidate pairs, and deterministic
  stage-first paths.
- [x] Implement immutable schema objects and strict validation with no runtime
  side effects.
- [x] Prove `screen/...` and `confirm/...` for the same candidate cannot
  collide.
- [x] Run the focused tests and commit only Task 1 files.

## Task 2: Generic training controls and reports

**Files:**
- Modify: `src/bts_nvs/training/run_training.py`
- Modify only if required: `src/bts_nvs/training/trainer.py`
- Modify: `tests/unit/test_run_training.py`
- Modify only if required: `tests/unit/test_trainer_loop.py`

- [x] Write failing tests for generic candidate/stage identity, internal
  holdout report generation, no-checkpoint 7k behavior, and `stop_step`
  separated from `max_steps=30000`.
- [x] Add the smallest generic controls to the existing process entry point;
  do not add another training loop.
- [x] Preserve B0 and all legacy CLI behavior when new arguments are absent.
- [x] Generate the existing Module 1 full-frame, detail, pose-strata, and
  experiment reports from generic internal-holdout runs.
- [x] Run focused plus legacy training tests and commit only Task 2 files.

## Task 3: 15k/30k confirmation snapshot contract

**Files:**
- Modify: `src/bts_nvs/training/run_training.py`
- Modify only if required: `src/bts_nvs/training/checkpoint.py`
- Modify: `tests/unit/test_run_training.py`
- Modify only if required: `tests/unit/test_checkpoint.py`

- [x] Write failing tests proving a 15k stop keeps the 30k schedule/config
  identity and preserves reports without retaining a milestone checkpoint.
- [x] Implement atomic snapshot report preservation and one rolling
  `checkpoints/recovery.pt` overwritten every 3k steps.
- [x] Prove a matching 15k recovery resumes to 30k and a mismatched recovery is
  rejected before training.
- [x] Run checkpoint/resume/training tests and commit only Task 3 files.

## Task 4: Fair command builder

**Files:**
- Create: `src/bts_nvs/experiments/commands.py`
- Create: `tests/unit/test_experiment_commands.py`

- [x] Write failing tests for one-process argument vectors at reference,
  screen, confirm-15k, confirm-30k, and production stages.
- [x] Implement a pure command builder that imports candidate settings and
  never executes a shell.
- [x] Assert paired B0/candidate commands differ only in registered candidate
  fields and output identity.
- [x] Run focused tests and commit only Task 4 files.

## Task 5: Artifact validation and failure ledger

**Files:**
- Create: `src/bts_nvs/experiments/artifacts.py`
- Create: `tests/unit/test_experiment_artifacts.py`

- [x] Write failing tests for required reports/renders/hashes, finite metrics,
  image completeness, step horizon, checkpoint policy, time/VRAM/Gaussian
  resources, and stale or partial directories.
- [x] Reuse provenance and existing full-training validation helpers where
  their contracts match; do not copy validation logic.
- [x] Implement an atomic durable failure ledger that records stage, scene,
  candidate, command, reason, and provenance.
- [x] Run focused and adjacent artifact tests and commit only Task 5 files.

## Task 6: Deterministic paired decisions

**Files:**
- Create: `src/bts_nvs/experiments/decisions.py`
- Create: `tests/unit/test_experiment_decisions.py`

- [x] Write table-driven failing tests for every quality, hard-stratum,
  detail, time-ratio, VRAM, integrity, and growth gate.
- [x] Implement scene decisions and exact tie-break ordering.
- [x] Prove 15k cannot accept a candidate, 30k reversal rejects it, and one
  scene failure cannot change another scene.
- [x] Implement seven-entry cohort decisions with explicit B0 fallback.
- [x] Run focused tests and commit only Task 6 files.

## Task 7: Thin Python CLI

**Files:**
- Create: `src/bts_nvs/experiments/run_experiment.py`
- Create: `tests/unit/test_run_experiment.py`

- [x] Write failing tests for preflight, one scene/candidate/stage invocation,
  subprocess failure, validated completion, resume, and no-overwrite behavior.
- [x] Compose Tasks 1, 4, 5, and 6; invoke the existing Python training entry
  point once and do not add a Bash runner.
- [x] Ensure all validation failures occur before GPU launch when inputs alone
  are sufficient to detect them.
- [x] Run focused tests and commit only Task 7 files.

## Task 8: Module gate

**Files:**
- Modify: `docs/superpowers/plans/2026-07-22-generic-experiment-runner.md`

- [x] Run all Module 1--3 unit tests and existing trainer/evaluator tests.
- [x] Run `git diff --check` and scan for Phase A/B/C names, revised opacity,
  hidden 0.0008 settings, duplicate training loops, and checkpoint buildup.
- [ ] On the L4, run a one-step preflight and a synthetic resume smoke; do not
  start a 7k scene run during Module 3 verification.
- [x] Record exact local test counts and the pending VM commands in this plan.
- [x] Commit the verification record. Passing this gate authorizes planning
  and execution of Stage A only.

## Verification record — 2026-07-22

Local verification on Windows:

- `pytest -q tests/unit`: **554 passed, 5 skipped in 16.73s**.
- Synthetic resume subset: **3 passed in 3.52s**, covering continuous-vs-resumed
  updates, runner recovery-path wiring, and strict 15k confirmation recovery.
- `git diff --check`: passed for Module 3 changes.
- No duplicate training loop exists under `src/bts_nvs/experiments`; the runner
  invokes `src/bts_nvs/training/run_training.py` exactly once with `shell=False`.
- No hidden `0.0008`, Phase A/B/C policy, or revised-opacity implementation was
  found in Module 3. The two repository-wide text matches are negative
  regression tests that reject the retired C1 ID and assert revised opacity is
  absent.
- Checkpoint policy remains zero model checkpoints at 7k and exactly one atomic
  `checkpoints/recovery.pt` at confirm/production; 15k preserves reports only.

L4 verification remains required before this module gate can pass. Run only:

```bash
BTS_RUN_CUDA_BACKEND_SMOKE=1 pytest -q tests/integration/test_cuda_backend_smoke.py
pytest -q \
  tests/unit/test_trainer_loop.py::test_fresh_resume_matches_continuous_training_with_real_updates \
  tests/unit/test_run_experiment.py::test_resume_uses_only_the_rolling_recovery_path \
  tests/unit/test_run_training.py::test_confirmation_resume_requires_complete_15k_recovery
```

Do not launch a 7k scene run as part of this verification. Stage A remains
unauthorized until both L4 commands pass and their exact output is recorded
here.
