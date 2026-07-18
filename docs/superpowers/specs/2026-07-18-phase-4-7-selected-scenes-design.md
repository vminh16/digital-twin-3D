# Phase 4.7 Selected-Scene Training Design

## Goal

Allow the existing full-training wrapper to run an explicit ordered subset of
the canonical 18 scenes without changing the locked training configuration or
discarding the existing cohort ledger.

## Interface

```bash
bash scripts/run_phase4_full_training.sh \
  --scene_ids HCM0644 HCM0674 HCM0540 HCM0539 HCM0421
```

`--scene_ids` accepts one or more case-sensitive canonical scene IDs. Omitting
the option preserves the current behavior and runs all 18 scenes.

## Runtime semantics

- The physical scene and manifest roots must still contain the exact canonical
  18-scene pool.
- Selected scenes run once each in the order supplied by the user.
- Unknown, wrong-case, duplicate, or empty selections fail before any scene is
  launched.
- The ledger schema and identity remain bound to all 18 scenes. Only selected
  records transition; unselected records retain their existing state.
- Existing valid recovery checkpoints resume normally. Existing validated
  trained scenes are skipped.
- Full resolution, 30,000 steps, backend decision, seed, checkpoint policy and
  all artifact validation remain unchanged.

## Scope

The Python CLI parses and validates selection. The Bash wrapper only forwards
arguments and contains no scene list or training logic. No new ledger schema,
output root, checkpoint type, or training configuration is introduced.
