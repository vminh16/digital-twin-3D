# Task 1 Report: Experiment schema and stage-first paths

## Scope

Implemented the pure schema/path foundation for Module 3. The change is
limited to the Task 1 schema module, its unit tests, and this requested report.
It does not modify the candidate registry, training code, runtime imports, or
filesystem state.

## Files

- Added `src/bts_nvs/experiments/experiment.py`
  - Frozen `Experiment` schema object.
  - Locked seven-scene cohort, stages, horizons, and resource limits.
  - Explicit caller-supplied winner/cohort authorization for confirm and
    production stages.
  - Deterministic stage-first paths without filesystem writes.
- Added `tests/unit/test_experiment_schema.py`
  - Covers fixed values, legal and illegal stage/candidate combinations,
    identity validation, immutability, and non-colliding screen/confirm paths.

## TDD evidence

### RED

Command:

```powershell
pytest tests/unit/test_experiment_schema.py -q
```

Output:

```text
ERROR collecting tests/unit/test_experiment_schema.py
ModuleNotFoundError: No module named 'bts_nvs.experiments.experiment'
1 error in 0.12s
```

The failure was expected: the test defined the desired public schema module
before that module existed.

### GREEN

Command:

```powershell
pytest tests/unit/test_experiment_schema.py -q
```

Output:

```text
..................                                                       [100%]
18 passed in 0.03s
```

### Focused plus adjacent registry validation

Command:

```powershell
pytest tests/unit/test_experiment_schema.py tests/unit/test_experiment_candidates.py -q
```

Output:

```text
.............................................                            [100%]
45 passed in 0.06s
```

### Diff validation

Command:

```powershell
git diff --check
```

Output: exit code 0; no whitespace errors.

## Commit

Commit subject: `feat: add experiment schema and stage-first paths`

Only the Task 1 schema, test, and this requested Task 1 report are staged for
the commit. Existing `.gitignore`, `docs/research/`, and other
`.superpowers/` scratch changes remain unstaged.

## Self-review

- `Experiment` is a frozen dataclass and `STAGE_HORIZONS` is read-only.
- The schema validates cohort, enum type, registered candidate identities, and
  every legal stage/candidate authorization rule before a caller can use a
  horizon or path.
- Confirm and production authorization are explicit constructor values; the
  module does not read mutable decision/global state.
- `run_path()` constructs `pathlib.Path` values only. No code writes files or
  imports GPU/training runtime modules.
- `reference`, `screen`, `confirm`, and `production` use stage-first layout;
  the tests prove that identical scene/candidate values produce distinct
  screen and confirm paths.
