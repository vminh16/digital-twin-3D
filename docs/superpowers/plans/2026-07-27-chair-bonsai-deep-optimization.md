# Chair and bonsai deep-optimization execution plan

**Authority:** `../specs/2026-07-27-chair-bonsai-deep-optimization.md`

**Status:** ACTIVE — Phase 1 code complete; L4 CUDA smoke is the exit gate.

## Success criteria

1. One generic harness owns v3 research runs for both scenes.
2. Historical v1/v2 reports continue to validate without migration.
3. Research uses a 30k optimizer horizon stopped at 15k.
4. Targeted holdouts expose the audited catastrophic regions using train RGB.
5. E3 candidates fail preflight until their implementation and tests exist.
6. Selection requires both mean and worst-tail improvement.
7. Production remains full-data and requires a fresh authorization artifact.

## Phase 0 — closure and harness unification

Status: complete when focused and affected tests pass.

- Close `B0-submission-q99-v1` and `MVP-hybrid-4scene-q99-v1`.
- Mark Stage A, Stage B1 and five-scene MVP documents historical.
- Consolidate durable history into one closure record.
- Add generic `research` stage:
  - scenes: `chair`, `bonsai`;
  - `max_steps=30000`;
  - `stop_step=15000`;
  - internal holdout;
  - no checkpoint or resume.
- Add the thin `scripts/run_chair_bonsai_research.sh` wrapper.
- Do not register E3 candidates yet.

## Phase 1 — targeted evidence and diagnostics

Status: active.

Implementation tasks:

1. Add a versioned targeted-holdout artifact without modifying historical
   `holdout.json`.
2. Chair: include close-mesh interpolation sentinels selected from train
   frames around `840–920`.
3. Bonsai: include frame `390` and high-pose-gap train sentinels.
4. Extend reports with:
   - 3D scale `p50/p95/p99/max`;
   - projected-radius `p95/p99/max`;
   - count and opacity mass above configured thresholds;
   - worst-decile LPIPS and symmetric-edge distance;
   - deterministic veil/collapse flags.
5. Add a diagnostic render that suppresses extreme-radius Gaussians only for
   root-cause verification. It must never be accepted as submission output.
6. Run CPU tests and one bounded CUDA smoke.

Exit gate: both incumbent policies can run through the v3 research harness and
produce complete comparable reports.

Implementation result:

- the wrapper creates deterministic `holdout_research_v3.json` with an
  identity hash distinct from historical `holdout.json`;
- `gaussian_diagnostics.json` contains scale, opacity, projected-radius and
  observed density-count diagnostics;
- root-cause renders are isolated in `diagnostic_filtered_renders/`;
- research `experiment_report.json` schema 2 contains worst-decile metrics and
  deterministic veil/collapse flags;
- CPU suite and shell syntax pass;
- run the bounded CUDA smoke on L4 before declaring Phase 1 closed:

```bash
BTS_RUN_RESEARCH_DIAGNOSTICS_SMOKE=1 \
python -m pytest tests/integration/test_research_diagnostics_smoke.py -q
```

## Phase 2 — chair

1. Register and test `E3-chair-scale-guard-v1`.
2. Run paired 15k research against the chair incumbent.
3. If the scale candidate passes, register
   `E3-chair-scale-fregs-v1`.
4. Screen the frequency candidate.
5. Reject any candidate that leaves one catastrophic sentinel, even when mean
   Score50 improves.

Stop after a passing scale candidate when the remaining error is no longer
high-frequency blur. FreGS is conditional, not mandatory.

## Phase 3 — bonsai

1. Register and test `E3-bonsai-scale-guard-sh4-v1`.
2. Run paired 15k research against the SH4 incumbent.
3. If scale control removes veil but depth remains unstable, register
   `E3-bonsai-geometry-sh4-v1`.
4. Add only provided-COLMAP and train-view geometry supervision.
5. Reject any candidate that worsens frame `390` or the high-gap tail.

Do not add Spec-Gaussian/GaussianShader until geometry passes.

## Phase 4 — confirmation

For each selected scene winner:

- train fresh incumbent and candidate at 30k on the same targeted holdout;
- preserve 15k and 30k reports;
- use one rolling recovery checkpoint;
- require the v3 tail gates at 30k;
- reject a gain that reverses between 15k and 30k.

At most two candidates proceed.

## Phase 5 — production

- Write a new cohort/submission candidate ID.
- Train each accepted policy fresh at 30k with all train images and no
  internal holdout.
- Keep frozen scene folders byte-identical.
- Rerender from exact `test_poses.csv` names, poses, intrinsics and dimensions.
- Use JPEG Q99, 4:4:4, optimized, non-progressive.
- Validate exactly seven top-level scene folders before packaging.

## L4 compute envelope

Decision path before full-data production:

```text
2 incumbent research runs x 15k
2 mechanism research runs x 15k
2 paired confirmation runs x 30k
= 120k scene-steps to select winners
```

If both winners pass, two fresh full-data production runs add 60k steps, for a
maximum end-to-end total of 180k steps. This is more total compute than two
blind 70k runs (`140k scene-steps`), but the 120k decision path yields controlled
evidence before production and avoids committing compute to an unidentified
failure mode. The 70k shortcut remains unauthorized. Run one GPU process at a
time.

## Current command boundary

Only registered candidates may be invoked:

```bash
bash scripts/run_chair_bonsai_research.sh chair E2-loss-local-laplacian-v1
bash scripts/run_chair_bonsai_research.sh bonsai E2-appearance-sh4-v1
```

These commands are harness examples, not authorization to allocate L4 compute
before the targeted holdout in Phase 1 is complete.
