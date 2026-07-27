# Task 1: Experiment schema and stage-first paths

## Files

- Create: `src/bts_nvs/experiments/experiment.py`
- Create: `tests/unit/test_experiment_schema.py`

## Requirements

- Write failing tests for the seven-scene cohort, stage enum, locked horizons,
  resource limits, legal stage/candidate pairs, and deterministic stage-first
  paths.
- Implement immutable schema objects and strict validation with no runtime
  side effects.
- Prove `screen/...` and `confirm/...` for the same candidate cannot collide.
- Run the focused tests and commit only Task 1 files.

## Binding design values

- Cohort, in order: `HCM0644 HCM0674 HCM0540 HCM0539 HCM0421 chair bonsai`.
- Stages: `reference`, `screen`, `confirm`, `production`.
- Horizons: reference/screen 7000; confirm/production 30000.
- Peak VRAM limit is strictly below 23 GiB (`23 * 1024` MiB).
- Paired candidate/B0 wall-time ratio limit is 1.25.
- Layout:
  - `reference/<scene_id>/`
  - `screen/<scene_id>/<candidate_id>/`
  - `confirm/<scene_id>/<candidate_id>/`
  - `production/<scene_id>/<candidate_id>/`
- `reference` accepts only `B0-reference`.
- `screen` accepts registered non-B0 candidates.
- `confirm` accepts B0 or a caller-authorized recorded winner; the schema API
  must make winner authorization explicit rather than reading mutable global
  state.
- `production` accepts only a caller-authorized cohort policy; the schema API
  must make authorization explicit.
- Reuse the closed candidate registry in `bts_nvs.experiments.candidates`.
- No filesystem writes or GPU/runtime imports from schema validation.
