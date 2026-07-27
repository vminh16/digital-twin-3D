# Task 2 Report: Generic training controls and reports

## Status

Complete and reviewed at `87257ce`. Task 2 consists of:

- `4b50b1b feat: add generic training controls and reports`
- `87257ce fix: validate experiment authorization and resume resources`

Committed files only:

- `src/bts_nvs/training/run_training.py`
- `tests/unit/test_run_training.py`

`trainer.py` was not changed because its existing `Trainer.train(stop_step=...)`
API already supports a target step below the configured `max_steps` horizon.

## Implementation

- Added generic `--candidate_id`, `--experiment_stage`, and optional
  `--stop_step` CLI controls.
- Reused the Task 1 `Experiment` stage/candidate contract and pinned the
  reference, screen, confirm, and production runtime contracts.
- Preserved legacy behavior when generic identity is absent.
- Applied `candidate_training_overrides(candidate_id)` to generic configs;
  `stop_step` remains outside config identity.
- Reused the existing trainer loop and propagated the actual target step to
  training, final evaluation, preview naming, and reports.
- Disabled checkpoints for generic reference/screen and required rolling
  recovery for confirm/production.
- Added generic internal-holdout generation of `qualification_report.json`,
  `detail_metrics.json`, `pose_strata.json`, and `experiment_report.json`.
- Built experiment resources from `summary.json` and `metrics.jsonl`: total
  time, max VRAM, peak Gaussian count, and final Gaussian count.

## TDD evidence

Inherited RED reproduced before production changes:

```text
pytest tests/unit/test_run_training.py -q
6 failed, 30 passed in 3.34s
```

The failures were the expected missing CLI arguments, generic validator/target
helpers, checkpoint policy, and candidate config fields.

Additional report/resource RED added before production changes:

```text
pytest tests/unit/test_run_training.py::test_experiment_resource_summary_uses_summary_and_metric_records tests/unit/test_run_training.py::test_generic_internal_holdout_writes_all_module_one_reports -q
2 failed in 6.05s
```

Both failed for the intended missing report/resource APIs.

Focused GREEN:

```text
pytest tests/unit/test_run_training.py -q
38 passed in 3.16s
```

## Final verification

```text
pytest tests/unit/test_run_training.py tests/unit/test_experiment_report.py tests/unit/test_detail_metrics.py tests/unit/test_pose_strata.py tests/unit/test_experiment_candidates.py tests/unit/test_experiment_schema.py tests/unit/test_trainer_loop.py tests/unit/test_qualification.py tests/unit/test_backend_qualification.py tests/unit/test_training_profiling.py tests/unit/test_full_training.py -q
227 passed in 12.66s
```

`git diff --check` and the staged diff check passed. Git emitted only the
workspace's LF-to-CRLF conversion warnings.

## Concerns and limits

- No real CUDA scene training was run; verification is the requested focused
  unit/regression coverage.
- The inner training CLI uses the Task 1 contract for structural stage/candidate
  validation. Recorded scene-winner and cohort authorization remains the outer
  experiment runner's responsibility, as designed.
- Ruff was unavailable in this environment, so no Ruff command was included in
  the completion evidence.
- Existing unrelated `.gitignore`, `.superpowers/`, and `docs/research/` state
  was left untouched and was not committed. This report itself is intentionally
  untracked.

## Review findings follow-up — 2026-07-22

### Status and implementation

- Added structural `--authorized_candidate_id` input. Reference/screen require
  it to be absent; confirm/production require it to equal `candidate_id`.
- `_generic_experiment` now forwards caller-supplied authorization as
  `authorized_scene_winner` or `authorized_cohort_candidate`; it never derives
  authorization from `candidate_id`.
- Generic resumes capture and validate the prior `summary.json` before
  optimization. After a successful additional segment, total time is summed
  and max VRAM is maximized while all current summary fields remain
  authoritative.
- Completed resumes that run no optimization leave `summary.json` unchanged.
  Legacy modes do not enter the generic merge path.

### TDD evidence

Authorization RED:

```text
pytest tests/unit/test_run_training.py::test_parse_args_accepts_generic_identity_authorization_and_stop_step tests/unit/test_run_training.py::test_generic_identity_is_all_or_none_and_uses_stage_candidate_contract tests/unit/test_run_training.py::test_generic_early_stages_reject_candidate_authorization tests/unit/test_run_training.py::test_generic_late_stages_require_explicit_matching_candidate_authorization -q
8 failed in 5.60s
```

Authorization GREEN, including adjacent existing confirm/production contract
tests:

```text
pytest tests/unit/test_run_training.py::test_parse_args_accepts_generic_identity_authorization_and_stop_step tests/unit/test_run_training.py::test_generic_identity_is_all_or_none_and_uses_stage_candidate_contract tests/unit/test_run_training.py::test_generic_early_stages_reject_candidate_authorization tests/unit/test_run_training.py::test_generic_late_stages_require_explicit_matching_candidate_authorization tests/unit/test_run_training.py::test_generic_confirmation_locks_30k_schedule_and_rolling_recovery tests/unit/test_run_training.py::test_generic_production_locks_full_30k_without_holdout -q
10 passed in 3.34s
```

Resume resource RED:

```text
pytest tests/unit/test_run_training.py::test_resumed_training_summary_merge_accumulates_resources_only tests/unit/test_run_training.py::test_resumed_training_summary_merge_rejects_invalid_resources tests/unit/test_run_training.py::test_finalize_generic_resume_summary_feeds_merged_experiment_resources tests/unit/test_run_training.py::test_finalize_generic_resume_summary_does_not_double_count_without_optimization -q
8 failed in 3.91s
```

Resume resource GREEN:

```text
pytest tests/unit/test_run_training.py::test_resumed_training_summary_merge_accumulates_resources_only tests/unit/test_run_training.py::test_resumed_training_summary_merge_rejects_invalid_resources tests/unit/test_run_training.py::test_finalize_generic_resume_summary_feeds_merged_experiment_resources tests/unit/test_run_training.py::test_finalize_generic_resume_summary_does_not_double_count_without_optimization -q
8 passed in 4.15s
```

Focused GREEN:

```text
pytest tests/unit/test_run_training.py -q
52 passed in 4.40s
```

### Final verification

The same adjacent suite command recorded above for the original 227-test run:

```text
pytest tests/unit/test_run_training.py tests/unit/test_experiment_report.py tests/unit/test_detail_metrics.py tests/unit/test_pose_strata.py tests/unit/test_experiment_candidates.py tests/unit/test_experiment_schema.py tests/unit/test_trainer_loop.py tests/unit/test_qualification.py tests/unit/test_backend_qualification.py tests/unit/test_training_profiling.py tests/unit/test_full_training.py -q
241 passed in 15.66s
```

`git diff --check` passed. Git emitted only the workspace's LF-to-CRLF
conversion warnings. No real CUDA scene training was run.
