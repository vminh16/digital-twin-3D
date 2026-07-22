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

- [ ] Write failing tests for the seven-scene cohort, stage enum, locked
  horizons, resource limits, legal stage/candidate pairs, and deterministic
  stage-first paths.
- [ ] Implement immutable schema objects and strict validation with no runtime
  side effects.
- [ ] Prove `screen/...` and `confirm/...` for the same candidate cannot
  collide.
- [ ] Run the focused tests and commit only Task 1 files.

## Task 2: Generic training controls and reports

**Files:**
- Modify: `src/bts_nvs/training/run_training.py`
- Modify only if required: `src/bts_nvs/training/trainer.py`
- Modify: `tests/unit/test_run_training.py`
- Modify only if required: `tests/unit/test_trainer_loop.py`

- [ ] Write failing tests for generic candidate/stage identity, internal
  holdout report generation, no-checkpoint 7k behavior, and `stop_step`
  separated from `max_steps=30000`.
- [ ] Add the smallest generic controls to the existing process entry point;
  do not add another training loop.
- [ ] Preserve B0 and all legacy CLI behavior when new arguments are absent.
- [ ] Generate the existing Module 1 full-frame, detail, pose-strata, and
  experiment reports from generic internal-holdout runs.
- [ ] Run focused plus legacy training tests and commit only Task 2 files.

## Task 3: 15k/30k confirmation snapshot contract

**Files:**
- Modify: `src/bts_nvs/training/run_training.py`
- Modify only if required: `src/bts_nvs/training/checkpoint.py`
- Modify: `tests/unit/test_run_training.py`
- Modify only if required: `tests/unit/test_checkpoint.py`

- [ ] Write failing tests proving a 15k stop keeps the 30k schedule/config
  identity and preserves reports without retaining a milestone checkpoint.
- [ ] Implement atomic snapshot report preservation and one rolling
  `checkpoints/recovery.pt` overwritten every 3k steps.
- [ ] Prove a matching 15k recovery resumes to 30k and a mismatched recovery is
  rejected before training.
- [ ] Run checkpoint/resume/training tests and commit only Task 3 files.

## Task 4: Fair command builder

**Files:**
- Create: `src/bts_nvs/experiments/commands.py`
- Create: `tests/unit/test_experiment_commands.py`

- [ ] Write failing tests for one-process argument vectors at reference,
  screen, confirm-15k, confirm-30k, and production stages.
- [ ] Implement a pure command builder that imports candidate settings and
  never executes a shell.
- [ ] Assert paired B0/candidate commands differ only in registered candidate
  fields and output identity.
- [ ] Run focused tests and commit only Task 4 files.

## Task 5: Artifact validation and failure ledger

**Files:**
- Create: `src/bts_nvs/experiments/artifacts.py`
- Create: `tests/unit/test_experiment_artifacts.py`

- [ ] Write failing tests for required reports/renders/hashes, finite metrics,
  image completeness, step horizon, checkpoint policy, time/VRAM/Gaussian
  resources, and stale or partial directories.
- [ ] Reuse provenance and existing full-training validation helpers where
  their contracts match; do not copy validation logic.
- [ ] Implement an atomic durable failure ledger that records stage, scene,
  candidate, command, reason, and provenance.
- [ ] Run focused and adjacent artifact tests and commit only Task 5 files.

## Task 6: Deterministic paired decisions

**Files:**
- Create: `src/bts_nvs/experiments/decisions.py`
- Create: `tests/unit/test_experiment_decisions.py`

- [ ] Write table-driven failing tests for every quality, hard-stratum,
  detail, time-ratio, VRAM, integrity, and growth gate.
- [ ] Implement scene decisions and exact tie-break ordering.
- [ ] Prove 15k cannot accept a candidate, 30k reversal rejects it, and one
  scene failure cannot change another scene.
- [ ] Implement seven-entry cohort decisions with explicit B0 fallback.
- [ ] Run focused tests and commit only Task 6 files.

## Task 7: Thin Python CLI

**Files:**
- Create: `src/bts_nvs/experiments/run_experiment.py`
- Create: `tests/unit/test_run_experiment.py`

- [ ] Write failing tests for preflight, one scene/candidate/stage invocation,
  subprocess failure, validated completion, resume, and no-overwrite behavior.
- [ ] Compose Tasks 1, 4, 5, and 6; invoke the existing Python training entry
  point once and do not add a Bash runner.
- [ ] Ensure all validation failures occur before GPU launch when inputs alone
  are sufficient to detect them.
- [ ] Run focused tests and commit only Task 7 files.

## Task 8: Module gate

**Files:**
- Modify: `docs/superpowers/plans/2026-07-22-generic-experiment-runner.md`

- [ ] Run all Module 1--3 unit tests and existing trainer/evaluator tests.
- [ ] Run `git diff --check` and scan for Phase A/B/C names, revised opacity,
  hidden 0.0008 settings, duplicate training loops, and checkpoint buildup.
- [ ] On the L4, run a one-step preflight and a synthetic resume smoke; do not
  start a 7k scene run during Module 3 verification.
- [ ] Record exact test counts and VM evidence in this plan.
- [ ] Commit the verification record. Passing this gate authorizes planning
  and execution of Stage A only.
