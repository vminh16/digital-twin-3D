# Phase 4.7 Selected-Scene Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a case-sensitive `--scene_ids` subset argument while preserving the canonical 18-scene ledger and locked training contract.

**Architecture:** Parse the optional selection in the thin Python CLI, validate it against `CANONICAL_SCENES`, and pass the resulting ordered tuple into the existing orchestration loop. Keep scene-pool and ledger validation canonical; make the Bash wrapper forward arguments unchanged.

**Tech Stack:** Python 3.12, argparse, pytest, Bash.

## Global Constraints

- Omitted `--scene_ids` runs all canonical scenes.
- A supplied selection is non-empty, case-sensitive, duplicate-free and canonical.
- The selection order is the execution order.
- The ledger always retains all 18 canonical scene records.
- No training hyperparameter or checkpoint behavior changes.

---

### Task 1: Python selection contract

**Files:**
- Modify: `src/bts_nvs/training/full_training.py`
- Modify: `src/bts_nvs/training/run_full_training.py`
- Test: `tests/unit/test_full_training.py`

**Interfaces:**
- Produces: `validate_scene_selection(scene_ids: Sequence[str] | None) -> tuple[str, ...]`.
- Changes: `run_full_training(..., scene_ids: Sequence[str] | None = None)`.

- [ ] Add failing tests for default-all, ordered subset, duplicate and unknown IDs.
- [ ] Run `.venv\Scripts\python -m pytest tests/unit/test_full_training.py -q` and confirm RED.
- [ ] Implement minimal validation, CLI `--scene_ids` with `nargs="+"`, and iterate only the validated selection.
- [ ] Run the same test file and confirm GREEN.

### Task 2: Wrapper forwarding and runbook

**Files:**
- Modify: `scripts/run_phase4_full_training.sh`
- Modify: `tests/unit/test_phase4_shell_scripts.py`
- Modify: `docs/phase_3_6_l4_runbook.md`

**Interfaces:**
- Consumes: arbitrary runner arguments as `"$@"`.

- [ ] Add a failing shell contract test requiring `"$@"` forwarding and no embedded selected scene IDs.
- [ ] Run `.venv\Scripts\python -m pytest tests/unit/test_phase4_shell_scripts.py -q` and confirm RED.
- [ ] Forward `"$@"` after locked path arguments and document the five-scene command.
- [ ] Run unit, integration, compileall and `git diff --check`.
- [ ] Commit as `feat: allow selected phase 4 training scenes`.
