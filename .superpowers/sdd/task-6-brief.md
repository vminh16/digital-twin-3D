# Task 6: Deterministic paired decisions

## Files

- Create: `src/bts_nvs/experiments/decisions.py`
- Create: `tests/unit/test_experiment_decisions.py`

## Requirements

- Follow TDD with table-driven gate tests.
- Consume already validated B0/candidate `experiment_report.json` mappings;
  reject mismatched scene, step, manifest, holdout, or non-finite data.
- Candidate eligible iff: overall score50 delta > 0; LPIPS delta <= 0; hard
  score50 delta >= 0; missing_edge and spurious_edge do not both worsen;
  paired time ratio <=1.25; peak VRAM strictly below 23*1024; integrity and
  growth flags pass.
- At 15k a candidate may only be rejected/pending; it can never be accepted.
- At 30k reject if 15k gain has reversed below any quality gate.
- Deterministic tie-break: larger score50 gain, lower LPIPS, lower symmetric
  edge distance, lower peak Gaussian count, then candidate ID for complete
  determinism.
- Scene decision records paired inputs/deltas/gates/selected candidate or
  explicit B0 fallback and provenance hashes.
- Cohort decision has exactly the locked seven scenes; one scene failure cannot
  affect another; missing/duplicate/unknown scene fails.
- Use canonical provenance save helpers; no runtime training/import or file
  discovery in pure decision functions.
- Commit only Task 6 source/tests; never stage scratch/user files.
