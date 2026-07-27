# Task 5: Artifact validation and failure ledger

## Files

- Create: `src/bts_nvs/experiments/artifacts.py`
- Create: `tests/unit/test_experiment_artifacts.py`

## Requirements

- Follow TDD and record RED/GREEN evidence.
- Validate a completed run against an `Experiment`, manifest hash, holdout hash
  when applicable, config hash, and expected image names.
- Require `qualification_report.json`, `detail_metrics.json`,
  `pose_strata.json`, `experiment_report.json`, validation renders, summary,
  metrics and config/provenance artifacts for internal-holdout stages.
- Reject missing/extra renders, non-finite metrics/resources, wrong scene,
  candidate, step, hashes, dimensions/completeness already encoded by reports,
  stale/partial directories, OOM/non-finite/invalid-state failure records, and
  uncontrolled primitive growth when explicitly recorded.
- Enforce checkpoint policy: no `.pt/.pth` anywhere for reference/screen; only
  `checkpoints/recovery.pt` for confirm/production; 15k snapshot contains no
  model checkpoint.
- Enforce time and VRAM resource validators from `experiment.py`; paired time
  ratio validation is performed when a B0 report is supplied.
- Reuse Module 1 report structure, provenance utilities, and suitable existing
  full-training validation helpers instead of copying logic.
- Implement atomic durable JSONL/JSON failure ledger recording stage, scene,
  candidate, command argv, reason, and provenance. Preserve previous failures;
  malformed existing ledger must fail rather than be overwritten.
- No deletion/overwrite of run artifacts.
- Commit only Task 5 source/tests; never stage scratch/user files.
