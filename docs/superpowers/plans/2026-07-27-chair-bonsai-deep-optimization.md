# Chair and bonsai deep-optimization execution plan

**Authority:** `../specs/2026-07-27-chair-bonsai-deep-optimization.md`

**Status:** ACTIVE — Phase 1 and candidate-theory gate validated; chair E3
research control implemented locally, L4 run pending.

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

Status: complete.

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

Exit gate: both incumbent policies run through the v3 research harness and
produce complete reports.

Implementation result:

- the wrapper creates deterministic `holdout_research_v3.json` with an
  identity hash distinct from historical `holdout.json`;
- `gaussian_diagnostics.json` contains scale, opacity, projected-radius and
  observed density-count diagnostics;
- root-cause renders are isolated in `diagnostic_filtered_renders/`;
- research `experiment_report.json` schema 2 contains worst-decile metrics and
  deterministic veil/collapse flags;
- CPU suite and shell syntax pass;
- complete L4 runs exist for both incumbents and all reports validate.

### Phase 1 result and interpretation

The historical Stage B1 contains no chair/bonsai run. For these scenes the
valid comparison chain is B0 7k versus the E2 7k screen on the same historical
holdout, followed by a supportive-only comparison with the targeted v3 15k
run. The latter changes the holdout and optimizer horizon and is not a paired
winner test.

| Scene | Valid paired B0 -> E2 7k | Views improved | Targeted v3 15k result |
|---|---:|---:|---|
| chair | Score50 `+0.5688`, PSNR `+0.3354`, LPIPS `-0.0068` | 16/20 Score50 | Score50 `60.0966`; one collapse (`525`) |
| bonsai | Score50 `+0.5963`, PSNR `+0.3640`, LPIPS `-0.0069` | 22/24 Score50 | Score50 `57.6896`; collapses `290/340/390` |

The incumbent method changes are therefore real but modest. They do not prove
that the targeted 15k runs are universal winners. On common views, chair
improves on all 3/3 available comparisons; bonsai improves on 6/8, while
frames `340` and `650` regress. The final 1k-step mean training loss is below
the 10k–11k window by 7.6% for chair and 13.3% for bonsai, but densification
still causes a non-monotonic curve. This is partial under-convergence, not
evidence for blindly extending to 70k.

The mechanism evidence is stronger:

- chair has 8,572 Gaussians above 128 px and a projected-radius maximum of
  275,805 px; filtering that tail repairs frame `525` but hurts most normal
  views, so Phase 2 must split/control the tail rather than globally delete it;
- bonsai has 21,088 Gaussians above 128 px and 2.16% of opacity mass in that
  tail; hard filtering loses about 6.8 dB PSNR on average, so scale control is
  only a safety screen and geometry consistency remains the primary method;
- the absolute veil flag fires on every view and is not a valid selection
  gate. Phase 2 must replace it with paired, incumbent-relative tail gates.

## Phase 2 — paired mechanism screen

All candidates use the exact v3 targeted holdout hash, seed, 30k optimizer
horizon and 15k stop of their scene incumbent. The evidence and rejected
alternatives are in
`../research/2026-07-27-chair-bonsai-candidate-validation.md`.

### Phase 2A — chair

Implementation status:

- complete: non-integer reprojection regression test;
- complete: continuous mapping, old/new reprojection and color-MAE artifact;
- complete: E3 registry, chair/research-only scope and artifact validation;
- complete: real-chair CPU preflight recovered scale `(1.50001973,
  1.50001935)` and reduced mean RGB MAE `77.10911 -> 33.29314`;
- pending: paired E3 15k L4 run and per-view decision;
- conditional: E4 AbsGrad implementation.

Execution:

1. Fix the paired decision gate:
   - replace the unusable absolute veil gate with paired per-view regressions;
   - make `525`, `870` and `885` explicit sentinels;
   - report final/initial Gaussian count and opacity-weighted radius tails.
2. Add a regression test for chair’s non-integer legacy observation scale:
   stored points2D must map continuously to current-camera reprojections.
3. Register `E3-chair-observation-scale-v1` as a research-only control.
4. Replace the integer-ceiling color-observation mapping with a robust
   continuous reprojection fit. Log old/new mapping residual and sparse-color
   error before training.
5. Run one paired 15k control against `E2-loss-local-laplacian-v1`. This
   measures selector bias and cannot authorize production.
6. Accept only when initialization error, mean Score50 and worst-decile LPIPS
   improve, no sentinel collapses, and normal-view PSNR does not regress.
7. Register and run `E4-chair-observation-scale-absgrad-v1` against E3 only if
   the mapping control passes but a giant-radius sentinel failure remains.

### Phase 2B — bonsai

1. Apply the same paired-gate correction and pin sentinels
   `290`, `340`, `390`, `420`, `430` and `650`.
2. Add deterministic progressive-resolution contract tests: both spatial
   dimensions downsampled by factor 4 at start, monotonic schedule, exact full
   resolution by step 5k, correctly scaled intrinsics and unchanged output
   resolution.
3. Register `E3-bonsai-c2f-absgrad-sh4-v1`.
4. Implement the paper-defined density curriculum as one mechanism: AbsGrad
   plus factor-4-downsample-to-full progressive train resolution. Retain SH4
   and every other incumbent setting.
5. Run one paired 15k screen against `E2-appearance-sh4-v1`.
6. Reject any candidate that worsens frame `390`, either of `340/650`, the
   hard-stratum Score50 or opacity-weighted radius growth.
7. Only when floaters/tails improve but frame `390` remains geometrically
   unstable, register
   `E4-bonsai-c2f-absgrad-sh4-sparse-depth-v1`.
8. Add robust sparse-depth anchors from provided COLMAP tracks. No dense
   monocular depth, synthetic interior anchors or external data are allowed.

Do not add higher SH, ASG, GaussianShader, HMGS, 2DGS/PGSR, MCMC or exact
Pixel-GS to this first screen.

### Phase 2 stop and compute policy

- Minimum initial screen: chair observation mapping and bonsai c2f:
  `2 x 15k = 30k` scene-steps, about 62 minutes of estimated L4 training plus
  validation.
- Conditional chair AbsGrad or bonsai sparse depth adds one 15k run only after
  its preceding causal gate passes.
- Stop a ladder immediately after a failed causal gate; do not compensate by
  increasing steps.
- Do not run 70k. A 5–10k low-LR polish is considered only after a candidate
  passes paired 30k confirmation with stable radius and opacity tails.

## Phase 3 — confirmation

For each selected scene winner:

- train fresh incumbent and candidate at 30k on the same targeted holdout;
- preserve 15k and 30k reports;
- use one rolling recovery checkpoint;
- require the v3 tail gates at 30k;
- reject a gain that reverses between 15k and 30k.

At most two candidates proceed.

## Phase 4 — production

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
2 required mechanism screens x 15k
up to 2 conditional screens x 15k
2 paired confirmation runs x 30k
= 90k to 120k new scene-steps to select winners
```

If both winners pass, two fresh full-data production runs add 60k steps, for a
maximum of 150k to 180k new scene-steps. The decision path yields paired
evidence before production and avoids committing compute to an unidentified
failure mode. The 70k shortcut remains unauthorized. Run one GPU process at a
time.

## Current command boundary

Only registered candidates may be invoked:

```bash
bash scripts/run_chair_bonsai_research.sh chair E2-loss-local-laplacian-v1
bash scripts/run_chair_bonsai_research.sh chair E3-chair-observation-scale-v1
bash scripts/run_chair_bonsai_research.sh bonsai E2-appearance-sh4-v1
```

The E2 commands reproduce the closed Phase 1 incumbents. E3 is the first
executable Phase 2 research control. Remaining Phase 2 IDs stay reserved until
their implementation, contracts and tests are merged.
