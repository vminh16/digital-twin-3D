# Chair and bonsai deep-optimization execution plan

**Authority:** `../specs/2026-07-27-chair-bonsai-deep-optimization.md`

**Status:** ACTIVE — E4–E8 rejected. Fresh
`E3-chair-observation-scale-30k-control-v1` is the active paired control for
E5-30k; no chair candidate is authorized for production.

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
- complete: paired E3 15k L4 run; mapping fixed frame `525` collapse while
  giant-radius failures remained at `870/885`;
- complete: E4 AbsGrad registry, chair/research-only scope and artifact
  validation;
- pending: paired E4 15k L4 run against E3.

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

E3 repaired the initialization defect. E4 was then rejected at 15k because it
fit train cameras better while reducing holdout Score50 by `1.4725`, increasing
spurious edges and growing the final population by 65%.

### Phase 2A2 — chair MCMC full horizon

1. Register `E5-chair-observation-scale-mcmc-v1` as chair/research-only.
2. Retain E3 observation mapping and local-Laplacian supervision.
3. Replace DefaultStrategy with MCMC, cap at 2M, noise LR `5e5`, opacity
   regularization `0.001`, scale regularization `0.01`, and relocation through
   25k.
4. Run E5 fresh to 30k with the same targeted holdout and rolling recovery
   every 3k.
5. Compare E5-30k first against the existing E3-15k lower bound. If it does not
   gain at least `+0.75` mean Score50 while keeping LPIPS, hard stratum and the
   870/885 tail pair non-regressing, reject it without running E3-30k.
6. Only after that lower-bound gate passes, spend a fresh E3-30k run for the
   paired causal comparison required before MVP authorization.
7. On the paired 30k comparison, require `+0.5` Score50 at up to `1.5x` wall
   time, or `+1.0` at `1.5x–2.0x`; reject above `2.0x`.

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

- Chair MCMC spends 30k scene-steps first. E3-30k is conditional, saving 30k
  when MCMC cannot beat the already available E3-15k lower bound.
- Conditional chair AbsGrad or bonsai sparse depth adds one 15k run only after
  its preceding causal gate passes.
- Stop a ladder immediately after a failed causal gate; do not compensate by
  increasing steps.
- Do not run 70k. A 5–10k low-LR polish is considered only after a candidate
  passes paired 30k confirmation with stable radius and opacity tails.

## Phase 3 — confirmation

For each selected scene winner, except that E5 uses its full-horizon research
run as the candidate side of the conditional paired comparison:

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
chair E5 lower-bound gate: 30k
conditional chair E3 paired control: +30k
bonsai ladder remains separately gated
```

If both winners pass, two fresh full-data production runs add 60k steps, for a
maximum of 150k to 180k new scene-steps. The decision path yields paired
evidence before production and avoids committing compute to an unidentified
failure mode. The 70k shortcut remains unauthorized. Run one GPU process at a
time.

## Registered command provenance

These entrypoints reproduce registered historical candidates. None is
currently authorized for a new chair run:

```bash
bash scripts/run_chair_bonsai_research.sh chair E2-loss-local-laplacian-v1
bash scripts/run_chair_bonsai_research.sh chair E3-chair-observation-scale-v1
bash scripts/run_chair_bonsai_research.sh chair E4-chair-observation-scale-absgrad-v1
bash scripts/run_chair_bonsai_research.sh bonsai E2-appearance-sh4-v1
bash scripts/run_chair_mcmc_research.sh
bash scripts/run_chair_perceptual_research.sh
bash scripts/run_chair_perceptual_adc_research.sh
```

The E2 commands reproduce the closed Phase 1 incumbents. Chair E3 is the
completed Phase 2 control. E4 and E5 are closed as rejected. E6 is closed as
an under-densified implementation. E7 passed its mechanism gate but failed
quality, compute and tail gates. E4–E7 must not be resumed or promoted; all
unregistered Phase 2 IDs remain reserved.

### Closed chair E6 staged research

1. Generate and hash sensitivity maps from the locked internal-train split.
2. Train `E6-chair-observation-scale-perceptual-v1` on the fixed 30k schedule,
   stopping with rolling recovery at 15k.
3. Reject before resume if mean Score50 gains less than `+0.50`, LPIPS or hard
   stratum regresses, either 870/885 drops by more than `0.25`, spurious edges
   increase, or scale/radius opacity tails exceed E3.
4. Resume the same run to 30k only after the 15k mechanism gate passes.
5. Require `+0.75` Score50 against E3-15k with non-regressing LPIPS,
   hard/worst-decile and 870/885 before spending a fresh E3-30k paired control.
6. On paired 30k evidence, require approximately `+1.0` Score50 before any
   production authorization.

E6 failed at step 15k because its growth override replaced standard 100-step
ADC and ended at about 297k Gaussians. The LPIPS and easy-stratum regressions
reject the run. These steps remain only as historical provenance.

### Closed chair E7 corrected ADC research

1. Run `E7-chair-perceptual-adc-corrected-v1` fresh with
   `scripts/run_chair_perceptual_adc_research.sh`.
2. Keep standard ADC every 100 steps and OR independent HD/MD perceptual masks
   at the locked 1000/1500-step events.
3. At 15k, reject the implementation if final population is below 1.2M or ADC,
   contribution and perceptual event diagnostics are incomplete.
4. Resume with `E7_STOP_STEP=30000` only if LPIPS/easy tail no longer repeats
   E6's regression, hard views retain their gain and scale/radius tails remain
   controlled.
5. Spend a fresh E3-30k paired control only after E7-30k clears the
   predeclared lower-bound gate.

E7 completed 15k at 1.662M Gaussians with complete mechanism diagnostics. Its
paired Score50 gain versus E3 was only `+0.0647`; SSIM, spurious-edge,
runtime and scale/radius tails failed their gates. Do not run the resume
command and do not spend E3-30k confirmation.

### Phase 2E — active chair E8 spectral split

Candidate:

```text
E8-chair-observation-scale-spectral-split-v1
```

Implementation order:

1. Add `spectral_math.py`:
   - FP32 spectral entropy and condition number from physical scales;
   - paper-faithful anisotropic child scales with `k=0.6`, `k0=1.0`;
   - finite-input validation and deterministic tensor shapes.
2. Add unit tests before strategy integration:
   - isotropic entropy equals `ln(3)` and condition number equals one;
   - increasingly elongated scales reduce entropy and increase condition;
   - selected children satisfy `kappa_child <= kappa_parent`;
   - empty masks and near-zero scales remain finite.
3. Add `spectral_density_strategy.py`:
   - subclass/compose pinned gsplat DefaultStrategy;
   - preserve normal ADC and pruning;
   - OR a low-entropy, gradient-independent split mask at refine events;
   - give ADC first claim and enforce the 2.1M cap;
   - expose separate ADC/spectral split counters.
4. Add `spectral_policy.py`, register E8 and enforce
   `chair/research`-only scope in training and artifact preflight.
5. Add `spectral_diagnostics.py` with entropy/condition quantiles, opacity
   mass below entropy 0.5, split count, cap-hit step and child-condition
   violations.
6. Add the thin locked wrapper
   `scripts/run_chair_spectral_research.sh`.
7. Run focused CPU tests, affected suite, shell syntax and one bounded CUDA
   smoke that proves optimizer state remains aligned after spectral splits.
8. Commit the experiment implementation before spending the fresh L4 run.

Implementation status: complete. The modular policy/math/strategy/diagnostic
path, staged artifact validation, locked wrapper and automatic L4 CUDA
preflight are present. Local result: `684 passed, 3 skipped`; the skipped CUDA
tests require the NVIDIA L4. The next action is the fresh 15k command below,
not further code expansion.

First compute gate:

```bash
bash scripts/run_chair_spectral_research.sh
```

This is a fresh factor-1, seed-0, internal-holdout run on the 30k optimizer
trajectory, stopped durably at 15k. Estimated compute is approximately one
E3-15k run plus the covariance-spectrum overhead; one GPU process at a time.

Decision ladder:

```text
implementation/smoke fail
  -> fix code only; spend no L4 research run

E8-15k mechanism or quality gate fail
  -> reject E8; no resume; no E3-30k

E8-15k pass
  -> resume same E8 run to 30k

E8-30k fails +0.75 lower-bound versus E3-15k
  -> reject E8; no E3-30k

E8-30k clears lower-bound
  -> run fresh E3-30k paired control

paired E8-30k fails final quality/tail gate
  -> reject E8

paired E8-30k passes
  -> issue decision artifact; run fresh full-data E8 production 30k
```

Final paired promotion requires at least `+0.50` Score50 over E3-30k, LPIPS
and SSIM non-regression, hard/worst-tail improvement, no sentinel regression,
lower spectral/scale/radius tails, runtime at most `1.25x` and VRAM below
23 GB.

### Atomic MVP fallback

At every failure branch, the submission fallback is the already closed
`MVP-hybrid-4scene-q99-v1`:

| Failure point | Compute stopped | Chair used for MVP |
|---|---|---|
| Code/preflight/CUDA smoke | before L4 run | existing v2 production chair |
| E8 15k gate | 15k | existing v2 production chair |
| E8 30k lower-bound | 30k | existing v2 production chair |
| Paired E3-30k confirmation | 60k candidate/control | existing v2 production chair |
| Full-data E8 production or rerender | production stage | existing v2 production chair |

Fallback is scene-atomic: copy/rerender the whole chair scene from
`runs/scene_opt_v2/production_mvp/scenes/chair`. Never choose E8 per frame.
The five frozen BTS scenes and current bonsai folder remain unchanged. A
missing E8 checkpoint, incomplete test pose, invalid filename/codec/dimension
or non-blank failure automatically selects the incumbent chair folder.
