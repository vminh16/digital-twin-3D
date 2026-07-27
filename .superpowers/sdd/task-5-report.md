# Task 5 report: artifact validation and failure ledger

Status: implementation ready for review.

Files:

- `src/bts_nvs/experiments/artifacts.py`
- `tests/unit/test_experiment_artifacts.py`

Implemented completed-run validation for stage identity, reports, resources,
render names, provenance, metrics and checkpoint policy, plus an atomic,
append-preserving failure ledger.

TDD evidence reported by implementer:

- Initial RED occurred before `artifacts.py` existed.
- Focused GREEN: 30 passed in 6.31s.

Fresh controller verification:

```text
pytest tests/unit/test_experiment_artifacts.py \
  tests/unit/test_experiment_report.py \
  tests/unit/test_experiment_provenance.py \
  tests/unit/test_experiment_schema.py \
  tests/unit/test_run_training.py \
  tests/unit/test_full_training.py -q
188 passed in 13.44s
```

`git diff --check` passed. No CUDA run was performed.
