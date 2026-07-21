# C1 Phase C 30k Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the locked `C1-absgrad-t08-revopacity-v1` candidate from scratch for 30,000 steps on HCM0181 and emit a deterministic Phase C decision against the existing B0 30k reference.

**Architecture:** Extend the existing full-length training path with a mutually exclusive research-candidate identity; do not create a second trainer or Bash runner. Add one focused Phase C module that builds the locked command, supports only the colocated rolling recovery checkpoint, computes the existing high-frequency diagnostics, and applies the preregistered gate. A thin Python CLI supplies paths and the README records the operational command.

**Tech Stack:** Python 3.11+, PyTorch/gsplat training path, pytest, YAML/JSON artifacts.

## Global Constraints

- Candidate is exactly `C1-absgrad-t08-revopacity-v1`.
- Scene is exactly `HCM0181`; factor 1; seed 0; 30,000 steps; internal holdout enabled.
- Backend and precision come from the accepted backend qualification artifact.
- Training starts fresh; resume is allowed only from `<run>/checkpoints/recovery.pt`.
- Retain one rolling recovery checkpoint every 3,000 steps.
- Preserve B0 behavior and do not modify B0 artifacts.
- Output is `runs/c1/phase_c/HCM0181` plus `runs/c1/phase_c/phase_c_decision.json`.
- No Bash runner and no dependency change.

---

### Task 1: Full-length research candidate mode

**Files:**
- Modify: `src/bts_nvs/training/c1_candidates.py`
- Modify: `src/bts_nvs/training/run_training.py`
- Modify: `tests/unit/test_c1_candidates.py`
- Modify: `tests/unit/test_run_training.py`

**Interfaces:**
- Produces: `FULL_LENGTH_CANDIDATES`, `full_length_mode_enabled(args)`, and `selected_density_candidate(args)`.
- Consumes: existing `candidate_settings(candidate_id)` and full-length validation/report path.

- [ ] Add tests proving that the C1 identity selects threshold `0.0008`, AbsGrad, and revised opacity at 30k while B0 full-length defaults stay unchanged.
- [ ] Run the focused tests and confirm they fail because the new full-length identity is absent.
- [ ] Add `--full_length_candidate`, enforce mutual exclusion and the existing 30k/HCM0181/rolling-resume contract, and route all full-length evaluation/report behavior through one mode predicate.
- [ ] Run focused tests and preserve the existing B0 full-length behavior.

### Task 2: Phase C command, artifact validation, and decision

**Files:**
- Create: `src/bts_nvs/training/c1_phase_c.py`
- Create: `src/bts_nvs/training/c1_phase_c_runner.py`
- Create: `src/bts_nvs/training/run_c1_confirmation.py`
- Create: `tests/unit/test_c1_phase_c.py`

**Interfaces:**
- Produces: pure Phase C contract/decision functions in `c1_phase_c.py` and
  `run_phase_c(...)` lifecycle orchestration in `c1_phase_c_runner.py`.
- Consumes: Phase B decision, accepted backend decision, B0/C1 full-length reports/configs, existing high-frequency evaluator, and rolling recovery path.

- [ ] Add tests for the exact command, Phase-B-pass prerequisite, fresh/resume directory handling, paired holdout identity, and every Phase C decision gate.
- [ ] Run the focused tests and confirm the missing module/API fails.
- [ ] Implement the smallest runner: run or resume once, validate full-length artifacts/config, compute both diagnostics, emit `phase_c_decision.json`, and never launch Phase D.
- [ ] Run focused tests and confirm candidate failure cannot be promoted.

### Task 3: Operator handoff and verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: `run_c1_confirmation.py` CLI.
- Produces: exact VM command, resume command, expected runtime/storage policy, and Phase C gate description.

- [ ] Add the completed Phase B result and the Phase C launch/resume commands to README.
- [ ] Run focused and full unit tests available in the current environment, compile Python sources, and inspect the final diff.
- [ ] Commit the verified implementation on `ex1/absgrad-revopacity-phase-a` without creating another branch.

## Self-review

- Spec coverage: candidate identity, fair HCM0181 pairing, 30k fresh/resume contract, single rolling checkpoint, backend reuse, diagnostics, decision gate, and no automatic Phase D are covered.
- Scope: no model/loss/optimizer/codec changes and no new Bash script.
- Output nesting remains one scene directory below the Phase C root.
