# Task 4: Fair command builder

## Files

- Create: `src/bts_nvs/experiments/commands.py`
- Create: `tests/unit/test_experiment_commands.py`

## Requirements

- Follow TDD and record RED/GREEN evidence.
- Build a pure `list[str]` argument vector invoking the existing
  `src/bts_nvs/training/run_training.py`; never execute a subprocess or shell.
- Inputs must make repo root, scene dir, manifest dir, output dir, accepted
  backend/precision, and `Experiment` explicit.
- Common locked runtime: seed 0, resize factor 1, cache images, pinned transfer,
  accepted optimizer backend and precision.
- `reference`/`screen`: max/stop 7000, internal holdout, no checkpoint flags,
  no resume, no authorization argument.
- `confirm` 15k: max 30000, stop 15000, internal holdout, checkpoint every
  3000, rolling checkpoint, explicit authorized candidate, no resume.
- `confirm` 30k resume: same config, stop 30000, resume exactly
  `<output>/checkpoints/recovery.pt`.
- Fresh 30k confirm must be representable for the paired B0 run; candidate
  confirm authorization comes from `Experiment`.
- `production`: max/stop 30000, no internal holdout, 3000 rolling checkpoint,
  explicit authorized candidate, no resume unless a separate validated resume
  input is supplied.
- Paired B0/candidate commands must differ only in registered candidate fields,
  authorization, and output identity; all fairness settings are identical.
- Return args without platform-dependent shell quoting.
- Reject illegal stop/resume/stage combinations before command construction.
- No filesystem writes, subprocess calls, Phase naming, revised opacity, Bash,
  or duplicated training logic.
- Commit only Task 4 source/tests; never stage scratch/user files.
