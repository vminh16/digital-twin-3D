# Task 7: Thin Python CLI

## Files

- Create: `src/bts_nvs/experiments/run_experiment.py`
- Create: `tests/unit/test_run_experiment.py`

## Requirements

- Follow TDD and record RED/GREEN evidence.
- CLI composes experiment schema, command builder, artifact validator, failure
  ledger, and decision engine. It must not contain training/metric logic.
- One invocation handles exactly one stage, scene, candidate and target stop;
  invokes the existing Python training entry point once.
- Validate experiment/authorization, manifest/holdout/backend identity, output
  state and command before subprocess/GPU allocation.
- Fresh run requires absent/empty output; resume requires matching recovery and
  permits existing matching output. Never overwrite a partial/stale directory.
- On subprocess nonzero or validation failure, atomically append durable failure
  evidence before exiting nonzero.
- On success, validate completed artifacts before returning success.
- Decision mode writes deterministic scene decision only from validated paired
  reports; cohort mode writes exactly seven entries. Training invocation and
  decision aggregation may be explicit subcommands if that keeps CLI thin.
- Use `sys.executable` by default; no Bash wrapper, shell=True, or command-string
  quoting.
- Unknown/mismatched identities fail before subprocess call.
- Commit only Task 7 source/tests; never stage scratch/user files.
