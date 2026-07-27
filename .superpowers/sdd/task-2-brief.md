# Task 2: Generic training controls and reports

## Files

- Modify: `src/bts_nvs/training/run_training.py`
- Modify only if required: `src/bts_nvs/training/trainer.py`
- Modify: `tests/unit/test_run_training.py`
- Modify only if required: `tests/unit/test_trainer_loop.py`

## Requirements

- Follow TDD and record RED/GREEN evidence.
- Add generic CLI identity fields `--candidate_id` and
  `--experiment_stage`; use the Task 1 stage/candidate validator.
- Add optional `--stop_step`; configured `max_steps` remains the schedule
  horizon. The invocation trains/evaluates through `stop_step` when supplied.
- Generic `reference` and `screen` require `max_steps=7000`, factor 1, seed 0,
  cached images, pinned transfer, internal holdout, and no resume.
- Generic `confirm` requires `max_steps=30000`, stop step 15000 or 30000,
  factor 1, seed 0, cached images, pinned transfer, rolling checkpoint and
  checkpoint interval 3000. Resume is allowed only from the output's
  `checkpoints/recovery.pt`.
- Generic `production` requires `max_steps=30000`, `stop_step=30000`, no
  internal holdout, and otherwise the same full-resolution backend/runtime
  controls as confirmation.
- Apply settings through `candidate_training_overrides(candidate_id)`; preserve
  legacy B0/default behavior exactly when generic identity is absent.
- Generic internal-holdout runs create a validation dataset and write:
  `qualification_report.json` (existing full-frame format),
  `detail_metrics.json`, `pose_strata.json`, and `experiment_report.json`.
- Reuse `evaluate_internal_validation`, `evaluate_detail_directory`,
  `build_pose_strata`, and `build_experiment_report`; do not duplicate metric
  or training logic.
- Resource summary for experiment report uses `summary.json` and
  `metrics.jsonl`: total time, max VRAM, peak/final Gaussian count.
- `reference` and `screen` save no checkpoints. Confirmation/production save
  rolling checkpoints; Task 3 owns snapshot preservation details.
- Evaluation/report `step` and preview filename use actual target stop step,
  not always max_steps.
- Holdout identity uses `split.manifest_sha256`; manifest identity uses
  `trainer.manifest_hash`; config identity uses `trainer.config_hash`.
- Existing qualification/full-length/backend/profile modes continue passing
  unchanged tests.
- Commit only Task 2 files.

## Constraints

- No second training loop and no Bash script.
- Do not add Phase A/B/C naming or revised opacity.
- Do not stage `.gitignore`, `docs/research/`, or `.superpowers/`.
