# Task 4 — Fair command builder report

## Scope

- Added `src/bts_nvs/experiments/commands.py`.
- Added `tests/unit/test_experiment_commands.py`.
- Kept this report untracked and excluded it from the Task 4 commit.

## RED evidence

1. Initial `python -m pytest tests/unit/test_experiment_commands.py -q` could
   not start because the workspace launcher targets a missing Python 3.11
   executable.
2. Re-ran the untouched test file through the repository virtual environment:

   ```text
   .\\.venv\\Scripts\\python.exe -m pytest tests/unit/test_experiment_commands.py -q
   ModuleNotFoundError: No module named 'bts_nvs.experiments.commands'
   ```

   This failure was the expected missing-production-module failure.

## GREEN evidence

After the minimal implementation, the focused suite passed:

```text
.\\.venv\\Scripts\\python.exe -m pytest tests/unit/test_experiment_commands.py -q
15 passed in 0.07s
```

`git diff --check` also exited successfully with no whitespace findings.

## Self-review

- `build_training_command` returns only `list[str]`; it invokes neither a
  subprocess nor filesystem operations.
- The vector names the existing training script and pins the shared fairness
  settings: seed 0, factor 1, cached images, pinned transfer, backend, and
  precision.
- Stage validation rejects unsupported stop/resume combinations before building
  the vector. A resume is accepted only as the lexical
  `<output>/checkpoints/recovery.pt` path.
- Confirmation and production derive CLI authorization from the validated
  experiment candidate. This supplies the generic CLI's required
  `B0-reference` authorization without changing `Experiment`'s legal B0
  confirmation contract.
- Candidate-specific optimizer settings remain in the registered candidate
  registry and are applied by the existing CLI from `--candidate_id`; the
  builder does not duplicate training logic.
- Paired confirmation tests assert B0 and candidate vectors have identical
  non-identity settings.
