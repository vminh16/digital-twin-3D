# Task 3: 15k/30k confirmation snapshot contract

## Files

- Modify: `src/bts_nvs/training/run_training.py`
- Modify only if required: `src/bts_nvs/training/checkpoint.py`
- Modify: `tests/unit/test_run_training.py`
- Modify only if required: the existing checkpoint test file (do not invent a
  duplicate if checkpoint coverage already lives elsewhere).

## Requirements

- Follow TDD and record RED/GREEN evidence.
- A confirmation model config always pins `max_steps=30000`; `stop_step` is
  invocation control and must not alter config hash or LR schedule identity.
- At a 15k stop, preserve report artifacts under
  `snapshots/step_000015000/`; at 30k, final reports remain at run root.
- Snapshot artifacts include the full-frame validation report,
  `detail_metrics.json`, `pose_strata.json`, and `experiment_report.json`.
  Preserve/copy atomically; no milestone model checkpoint is retained.
- The only model state is `checkpoints/recovery.pt`, atomically overwritten at
  3000-step intervals and at the current stop step.
- Resume confirmation only from the same output's recovery file. Validate
  manifest/config hashes, complete state, and expected current step. A 15k
  recovery may resume to 30k; mismatched, beyond-target, or wrong-path
  recovery fails before training.
- Initial train-view and initial validation artifacts remain reusable across
  the matching resume; do not recompute or overwrite them inconsistently.
- Legacy full-length qualification and non-generic behavior must remain
  unchanged.
- Do not add milestone `.pt` files, a second training loop, or Bash scripts.
- Commit only Task 3 source/test files; never stage scratch/user files.
