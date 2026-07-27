# Task 1 report

Status: complete after review fixes.

Commits:

- `0973d5a feat: add experiment schema and stage-first paths`
- `ba13b21 fix: enforce experiment resource gates`

Implemented immutable experiment/stage schema, exact seven-scene cohort,
stage-first paths, explicit winner/cohort authorization, locked horizons, and
pure resource validators. Peak VRAM must be finite, nonnegative, and strictly
below `23 * 1024` MiB. Paired wall-time ratio must be finite, nonnegative, and
at most `1.25`.

TDD evidence:

- Initial RED: `pytest tests/unit/test_experiment_schema.py -q` failed with
  `ModuleNotFoundError` before `experiment.py` existed.
- Initial GREEN: 18 schema tests passed; schema + candidate suite: 45 passed.
- Review-fix RED: resource-validator imports failed before validators existed.
- Review-fix GREEN:
  `pytest tests/unit/test_experiment_schema.py tests/unit/test_experiment_candidates.py -q`
  returned `65 passed in 0.10s`.
- `git diff --check` passed.

The scratch report was removed from tracking in `ba13b21` and recreated only
as an untracked local coordination record.
