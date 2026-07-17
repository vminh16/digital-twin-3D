# Phase 4.7 Full-Training Runner Design

## Purpose

Train all 18 canonical BTS scenes sequentially on one NVIDIA L4 without losing
cohort progress when a process, VM, or individual scene fails.  This subphase
only establishes complete 30,000-step training artifacts.  Compact inference
export remains a separate Phase 4.7 completion step, so the ledger uses
`trained`, not `complete`.

## Qualification input

The runner requires the three existing HCM0181 backend profiles under:

```text
runs/phase4/backend_qualification/
├── reference/backend_profile.json
├── fused/backend_profile.json
└── amp/backend_profile.json
```

If `backend_qualification.json` is absent, the runner invokes the canonical
comparator with `<repo>/src` prepended to `PYTHONPATH` and creates the aggregate
report atomically.  It never reruns qualification automatically.

The aggregate report must have `accepted: true`.  Training uses exactly
`selected_optimizer_backend` and `selected_precision`; environment variables may
move roots or choose the Python executable but may not override the selected
backend.

The current audited profiles select:

```text
optimizer_backend: adam
precision: fp32
```

Fused Adam and AMP remain unselected because their measured speedups are below
the qualification gates.  The runner reads this result rather than hardcoding
it so the contract remains reproducible if a valid report is regenerated.

## Canonical scene pool

The production pool is the sorted intersection-free identity set represented by
`data/bts_scenes` and `runs/manifests`.  Both roots must contain exactly these 18
scene directories:

```text
hcm0031 hcm0034 HCM0181 HCM0193 HCM0204 HCM0249 HCM0254 HCM0276
HCM0421 HCM0539 HCM0540 HCM0644 HCM0674 HCM1439 HNI0131 HNI0265
HNI0366 HNI0437
```

Each manifest directory must contain `manifest.json` and its referenced NPZ.
Scene identity matching is case-sensitive.  Missing, extra, duplicate, or
mismatched identities fail before the first training process starts.

Production training uses all physical train images.  The runner must not pass
`--internal_holdout`, `--qualification_candidate`, `--profile_input`,
`--backend_qualification`, or `--full_length_qualification`.

## Training contract

Every scene uses the same locked CLI values:

```text
resize_factor: 1
max_steps: 30000
checkpoint_every: 3000
seed: 0
cache_images: true
pinned_transfer: true
optimizer_backend/precision: selected qualification result
```

The runner launches one process at a time.  It does not parallelize scenes on a
single L4 and does not alter batch size, loss, density strategy, learning rates,
SH schedule, or sampling.

## Bounded checkpoint and resume

`run_training.py` gains a production-safe `--rolling_checkpoint` switch.  It
passes `rolling_checkpoint=True` to `Trainer.train` without enabling internal
holdout or Phase 4.5 evaluation.  Checkpoints are written atomically to:

```text
<scene_run>/checkpoints/recovery.pt
```

The checkpoint is replaced every 3,000 steps and at step 30,000.  No numbered
checkpoints are retained.  Its config hash includes the selected backend and
precision, and AMP scaler state remains part of the checkpoint when applicable.

On rerun:

- a valid `trained` scene is skipped;
- a `running` or `failed` scene with a valid same-run `recovery.pt` resumes;
- a non-empty scene directory without a valid recovery or trained artifact
  fails clearly and is never overwritten;
- config, manifest, or backend hash mismatch fails before optimization;
- the runner never deletes or renames user artifacts.

## Ledger

The canonical ledger is:

```text
runs/phase4/full_training/ledger.json
```

It is deterministic JSON, written through a temporary sibling and atomic
replace.  It contains schema version, selected backend identity, qualification
report SHA-256, sorted scene IDs, and one record per scene with status:

```text
pending | running | trained | failed
```

Each scene record contains only stable or operationally necessary fields:

- scene ID;
- status;
- run directory relative to the full-training root;
- completed step when known;
- config and manifest hashes when known;
- concise error type/message for `failed`.

No timestamps are stored, preserving deterministic test fixtures.  Before
launch, status becomes `running`.  A non-zero subprocess result becomes
`failed`, the ledger is flushed, and the cohort runner exits non-zero.  Later
scenes remain `pending`.

## Trained artifact validation

A scene becomes `trained` only when all checks pass:

- `summary.json.total_steps == 30000`;
- `metrics.jsonl` has exactly 30,000 records with ordered steps 1 through 30,000
  and finite losses;
- `convergence.json.final_render_non_blank == true`;
- `config.yaml` has the locked full-resolution values and selected backend;
- `manifest_hash.json` agrees with the current manifest artifact;
- `checkpoints/recovery.pt` loads at step 30,000 with matching config and
  manifest hashes;
- final fixed train preview exists and decodes as non-empty RGB.

This validation establishes completed training, not novel-view quality or
leaderboard readiness.  No official test image or official test render is read.

## Components

Implementation remains small and separated by responsibility:

1. `src/bts_nvs/training/full_training.py` owns scene-pool validation, ledger
   load/save, trained-artifact validation, and subprocess orchestration.
2. `src/bts_nvs/training/run_full_training.py` is the thin argparse entry point.
3. `scripts/run_phase4_full_training.sh` resolves repository-relative defaults,
   sets `PYTHONPATH=<repo>/src`, prepares manifests, and invokes the Python
   runner.
4. `run_training.py` only gains the generic rolling-checkpoint CLI plumbing.

The shell wrapper contains no business logic and no embedded JSON mutation.

## Environment overrides

Only location/runtime overrides are supported:

```text
PYTHON_BIN
BTS_SCENES_ROOT
BTS_MANIFESTS_ROOT
BTS_BACKEND_ROOT
BTS_FULL_TRAINING_ROOT
```

All resolved roots must remain explicit paths supplied to the Python runner.
Optimization values and scene identities are not overrideable.

## Error handling

- Preflight and qualification failures happen before ledger status changes.
- Scene subprocess failure preserves recovery and records `failed`.
- Disk/checkpoint capacity failures propagate from the existing trainer guard.
- `SIGINT`/termination leaves the current scene as `running`; rerun audits its
  recovery and resumes it.
- Corrupt JSON, YAML, metrics, image, manifest, or checkpoint fails closed.
- Error output includes the scene ID and artifact path but does not dump tensors.

## Testing

Unit tests cover:

- exact 18-scene pool and case-sensitive mismatch rejection;
- aggregate report generation from existing profiles with `src` import path;
- selected backend propagation and override rejection;
- deterministic atomic ledger transitions;
- fresh, resumable, trained, failed, and ambiguous non-empty scene states;
- 30,000-record validation and non-finite/out-of-order rejection;
- rolling checkpoint CLI plumbing;
- subprocess failure stopping later scenes;
- shell wrapper contract and absence of destructive commands.

Local tests mock only subprocess execution; they use real parsers and artifact
validators.  A one-scene L4 dry invocation may be run before the full cohort,
but the implementation must not claim 18-scene completion from mocked tests.

## Output layout

```text
runs/phase4/full_training/
├── ledger.json
└── scenes/
    ├── HCM0181/
    │   ├── config.yaml
    │   ├── environment.json
    │   ├── manifest_hash.json
    │   ├── metrics.jsonl
    │   ├── timing.json
    │   ├── summary.json
    │   ├── convergence.json
    │   ├── checkpoints/recovery.pt
    │   └── train_previews/
    └── ...
```

## Exit condition

The training-only runner exits zero when all 18 scene records are `trained` and
their artifacts pass validation.  This authorizes compact inference export; it
does not by itself mark Phase 4.7 complete or authorize final test rendering.

