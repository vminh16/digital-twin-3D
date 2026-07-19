# C1 AbsGrad × Revised Opacity 30k Experiment Design

**Date:** 2026-07-19  
**Status:** Approved direction; implementation is gated on review of this written specification.  
**Baseline authority:** `B0-submission-q99-v1` remains CLOSED and immutable.

## 1. Objective

Implement the smallest reproducible research extension needed to test whether
AbsGrad densification reduces blurred or missing BTS details, and whether revised
opacity reduces haze, halos, and floaters after clone/split operations.

The experiment runs directly to 30,000 optimization steps on an NVIDIA L4 with
23 GB usable VRAM. A 7,000-step run is not the final experiment. Short execution
is limited to unit/integration smoke checks; research decisions use full 30k runs.

## 2. Why direct 30k is justified

The earlier 7k proposal was a low-cost hypothesis screen. It is no longer the
selected design because:

- the training target is a VM with a dedicated NVIDIA L4 rather than local compute;
- the existing HCM0181 artifact provides a clean B0 30k internal-holdout reference;
- the repository already has rolling 3k recovery checkpoints and validated L4
  backend selection;
- the HCM0181 baseline improved materially between 7k and 30k, so selecting only
  at 7k could reject a candidate whose post-densification optimization is useful.

The design remains adaptive across scenes: it completes both candidates on
HCM0181 before spending compute on HCM0421.

## 3. Hypotheses

### H1 — AbsGrad

The default strategy accumulates signed projected-position gradients before
taking their norm. Opposing pixel gradients may cancel, suppressing splits at
thin structures. AbsGrad accumulates component-wise absolute gradients, so
opposing evidence remains non-zero.

Expected observable effects:

- lower missing-edge error on cables, antenna boundaries, and lattice structures;
- lower LPIPS or higher SSIM if recovered details match held-out RGB;
- a changed Gaussian growth trajectory without uncontrolled primitive explosion.

### H2 — revised opacity conditioned on AbsGrad

Clone/split operations alter the number and overlap of primitives. Revised
opacity changes the opacity assigned during densification so the aggregate
optical contribution is less distorted.

Expected observable effects relative to AbsGrad alone:

- lower spurious-edge error and fewer halos/floaters;
- no loss of the detail recovery produced by AbsGrad;
- no regression in the evaluator-aligned composite score.

This experiment does not include a revised-opacity-only arm. It estimates the
incremental value of revised opacity conditional on AbsGrad, saving one 30k run
per scene.

## 4. Candidate identities and locked configuration

Two new candidate IDs are allowed:

| Candidate ID | Renderer `absgrad` | Strategy `absgrad` | `revised_opacity` | `grow_grad2d` |
|---|---:|---:|---:|---:|
| `C1-absgrad-t08-v1` | `true` | `true` | `false` | `0.0008` |
| `C1-absgrad-t08-revopacity-v1` | `true` | `true` | `true` | `0.0008` |

`0.0008` is fixed for C1. The installed `gsplat==1.4.0` documentation states
that AbsGrad requires a higher threshold than signed-average gradient and gives
`0.0008` as the example value. C1 does not add a threshold sweep.

Every other optimization value remains locked to B0 research settings:

- `max_steps=30000`;
- `resize_factor=1`;
- `seed=0`;
- internal holdout enabled;
- `prune_opa=0.005`;
- `grow_scale3d=0.01`;
- `refine_start_step=500`;
- `refine_stop_step=15000`;
- `refine_every=100`;
- `reset_every=3000`;
- loss `0.8 × L1 + 0.2 × (1 − SSIM)`;
- SH degree schedule, initialization, optimizer learning rates, classic
  rasterization, `eps2d=0.3`, and output color handling unchanged.

AbsGrad must have one source of truth. The same config value is passed to both
`rasterization(absgrad=...)` and `DefaultStrategy(absgrad=...)`. A mismatch is a
configuration error, not a supported mode.

## 5. Experiment sequence and compute budget

### Phase A — HCM0181 hypothesis test

Reference:

- existing committed B0 30k internal-holdout artifact from commit `411c8de`;
- PSNR `22.6915559431`, SSIM `0.8053435291`, LPIPS `0.1112587926`;
- 6,861,805 final Gaussians, 11.79 GB peak VRAM, 2.90 hours on NVIDIA L4.

New runs:

1. `HCM0181/C1-absgrad-t08-v1`, fresh 30k;
2. `HCM0181/C1-absgrad-t08-revopacity-v1`, fresh 30k.

Estimated sequential compute: about 5.8 GPU-hours using the B0 HCM0181 run as
the planning anchor. Actual time is recorded and not assumed equal.

### Phase B — HCM0421 locked confirmation

Run only if Phase A selects a winner.

New runs:

1. `HCM0421/B0-research-30k-v1`, fresh 30k with internal holdout;
2. `HCM0421/<Phase-A-winner>`, fresh 30k with the identical holdout.

The production HCM0421 checkpoint cannot serve as the reference because it was
trained with `internal_holdout=false`; evaluating it on training images would
leak the comparison. A new B0 research control is required.

Estimated additional sequential compute: about 5.8 GPU-hours. The maximum C1
budget is therefore about 11.6 GPU-hours before any submission-cohort retraining.

HCM1439 is not used in this first 30k experiment because no clean B0 30k
internal-holdout reference exists. Using it would require an additional B0 run.

## 6. Decision metrics

### 6.1 Primary evaluator-aligned metrics

For each held-out image, compute PSNR, SSIM, and LPIPS with the repository's
existing evaluation path, then average per scene. Report the local composite
using the inferred normalization:

```text
Score50 = 40 - 40 × LPIPS + 30 × SSIM + 0.6 × PSNR_dB
```

`Score50` is explicitly local and must not be described as an official score.
The official LPIPS backbone and exact evaluator internals remain unconfirmed.

### 6.2 Automated high-frequency diagnostics

High-frequency diagnostics use only internal validation RGB and rendered RGB.
They do not alter training or submission images.

For each image:

1. convert RGB to luminance with fixed coefficients
   `Y = 0.299R + 0.587G + 0.114B`;
2. compute Sobel gradient magnitude for reference and render;
3. define the reference edge mask from the top 10% reference gradient values;
4. define the reference flat mask from the bottom 50% reference gradient values;
5. compute normalized missing-edge energy on the edge mask;
6. compute normalized spurious-edge energy on the flat mask;
7. compute full-image Laplacian L1 error (`HF-L1`).

These diagnostics are explanatory and veto-only. They are not added to the
training loss and are not presented as official metrics. Masks are generated
algorithmically; no manual BTS crop or per-pose annotation is allowed.

## 7. Selection rules

### Phase A eligibility

A candidate is eligible only when all conditions hold:

- final validation artifacts cover every internal validation image;
- PSNR, SSIM, LPIPS, Score50, HF-L1, missing-edge, and spurious-edge are finite;
- final checkpoint is exactly step 30,000 and config/manifest hashes match;
- no CUDA OOM, non-finite loss, or invalid Gaussian state occurred;
- peak VRAM is below 23 GB;
- wall time and Gaussian trajectory are recorded.

### Phase A winner

Compare each candidate against the existing B0 HCM0181 reference.

1. Reject a candidate when `ΔScore50 <= 0`.
2. Reject a candidate when both missing-edge and spurious-edge means worsen.
3. Among remaining candidates, choose the highest `Score50`.
4. If the Score50 difference between candidates is numerically tied at the
   stored precision, choose lower HF-L1; then lower peak Gaussian count; then
   `C1-absgrad-t08-v1` as the simpler model.

No significance claim is made from one seed. The same seed and holdout provide
a controlled paired engineering comparison, not a population-level estimate.

### Phase B confirmation

The Phase-A winner passes C1 only if, against the fresh HCM0421 B0 research
control:

- `ΔScore50 > 0`;
- LPIPS does not increase;
- missing-edge and spurious-edge do not both worsen;
- the run satisfies the same integrity and resource gates.

If Phase A passes but Phase B fails, classify the effect as scene-specific and
do not retrain the seven-scene submission cohort.

## 8. L4 CUDA execution contract

The experiment reuses the repository's backend qualification rather than
hard-coding a precision mode.

Before C1:

```bash
bash scripts/run_phase4_backend_qualification.sh
```

The accepted `optimizer_backend` and `precision` are loaded once from the
backend qualification artifact and used for every C1/B0-control run. The current
L4 profiles show `adam-fused/amp-fp16` as the fastest measured profile, but C1
uses it only if the existing qualification gates accept it.

Required runtime settings:

- NVIDIA L4, compute capability 8.9;
- `cache_images=true`;
- `pinned_transfer=true`;
- accepted fused optimizer/precision pair from backend qualification;
- Gaussian parameters remain FP32 under AMP;
- one scene per GPU process;
- rolling atomic recovery checkpoint every 3,000 steps;
- no `torch.compile`, TF32 override, multi-GPU synchronization, custom CUDA
  kernels, or dependency upgrade in C1.

The exclusions keep the implementation small and prevent CUDA optimization
changes from becoming extra experimental factors.

## 9. Software design

### 9.1 Configuration boundary

Add a separate `--research_candidate` argument with exact allowed IDs. Do not
weaken or overload `--qualification_candidate`, `--full_length_qualification`,
or production training contracts.

Research candidate mode requires:

- fresh or exact rolling-checkpoint resume;
- factor 1, seed 0, 30k steps, checkpoint interval 3k;
- cached images and pinned transfer;
- internal holdout;
- candidate ID included in the config hash;
- resume path exactly `<output_dir>/checkpoints/recovery.pt`.

Candidate mode is mutually exclusive with profile, backend qualification,
7k qualification, and B0 full-length qualification modes.

### 9.2 Rendering and density adapter

Extend existing function signatures only:

- `render_gaussians(..., absgrad: bool = False)`;
- `GsplatStrategy(..., absgrad: bool = False, revised_opacity: bool = False)`.

Defaults preserve B0 behavior. The trainer reads both values from its immutable
config and passes them to the renderer and strategy.

### 9.3 Diagnostics and decision artifact

Add one focused high-frequency metrics module and one C1 decision module. The
decision output is deterministic JSON containing:

- candidate IDs and source run paths;
- config and manifest hashes;
- raw PSNR/SSIM/LPIPS and Score50;
- high-frequency diagnostic aggregates;
- time, VRAM, peak/final Gaussian count;
- per-gate booleans;
- selected candidate or explicit rejection reason.

### 9.4 Runner

Add a shell runner following existing Phase 4 scripts. It:

- verifies the repository root and required manifests;
- loads the accepted backend decision;
- creates candidate-specific output directories without overwriting B0;
- resumes only from a matching rolling recovery checkpoint;
- runs Phase A candidates sequentially;
- computes the Phase A decision;
- stops before Phase B when no candidate passes;
- runs the HCM0421 B0 control and locked winner only after Phase A passes.

## 10. Failure handling

- Existing non-empty run directories are rejected unless a valid recovery
  checkpoint matches the config and manifest hashes.
- CUDA OOM, non-finite values, missing validation renders, incomplete metrics,
  or a checkpoint below 30k prevent selection.
- A failed candidate does not cause another candidate to be declared winner by
  default; the remaining candidate must independently pass against B0.
- The runner records failure type and message in a ledger before exiting.
- B0 checkpoints, outputs, manifests, and submission archives are read-only.

## 11. Test strategy

Unit tests must prove:

- B0 defaults still pass `absgrad=False` and `revised_opacity=False`;
- AbsGrad is passed identically to renderer and strategy;
- revised opacity reaches `DefaultStrategy` only for the combined candidate;
- candidate IDs map to exactly the locked config values;
- invalid mixed modes and resume paths are rejected;
- high-frequency metrics are zero for identical images, detect a blurred edge
  as missing energy, and detect an added checker/noise pattern as spurious energy;
- decision rules reject non-finite/incomplete/resource-invalid runs and select
  deterministically;
- runner commands contain factor 1, 30k, 3k checkpointing, seed 0, cache,
  pinned transfer, accepted backend, precision, and rolling checkpoint flags.

GPU verification on the VM must include:

1. existing real gsplat CUDA forward/backward smoke;
2. a bounded C1 candidate smoke that reaches one densification event and confirms
   `means2d.absgrad` exists and all gradients are finite;
3. the full Phase A runs only after the smoke passes.

## 12. Out of scope

C1 does not change:

- loss weights or add LPIPS/depth/frequency loss;
- optimizer learning rates;
- densification schedule other than gradient interpretation and fixed threshold;
- MCMC strategy;
- antialiasing or rasterization mode;
- scene normalization, camera model, COLMAP initialization, SH schedule;
- inference codec or submission packaging;
- official test-data access or tuning;
- manual image editing, BTS masks, crops, or per-pose adjustments.

Any later change to those items requires a new candidate ID and a separate
design/decision record.
