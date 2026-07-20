# C1 AbsGrad × Revised Opacity Staged Experiment Design

**Date:** 2026-07-19
**Status:** Revised after compute/storage review; implementation is gated on approval of this specification.
**Baseline authority:** `B0-submission-q99-v1` remains CLOSED and immutable.

## 1. Objective

Implement the smallest reproducible research extension needed to test whether
AbsGrad densification reduces blurred or missing BTS details, and whether revised
opacity reduces haze, halos, and floaters after clone/split operations.

The experiment uses the repository's established 7,000-step qualification
contract to screen candidates cheaply and fairly. Only the selected winner is
retrained from scratch for a 30,000-step confirmation. If it passes, the same
locked method is trained from scratch on all seven submission scenes.

## 2. Why 7k screening precedes 30k

The baseline screening authority is the six-scene, seed-0, factor-1, fresh 7k
qualification. Comparing a new 7k candidate with those existing B0 references
holds steps, seed, holdout, resolution, optimizer, and data constant. Directly
running every arm to 30k would spend substantially more compute before showing
that the mechanism has any repeatable benefit.

Thirty thousand steps are still required at the confirmation and production
stages. On HCM0181, the clean B0 internal-holdout result improved from 7k to 30k
by about `+3.646 Score50`, while the 30k run cost 2.90 L4 GPU-hours. This is strong
evidence that 7k is not a substitute for final training, but it is evidence from
only one scene. Therefore 30k is used after multi-scene screening, not before it.

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
incremental value of revised opacity conditional on AbsGrad, saving one candidate
run per screening scene.

## 4. Candidate identities and locked configuration

Two new candidate IDs are allowed:

| Candidate ID | Renderer `absgrad` | Strategy `absgrad` | `revised_opacity` | `grow_grad2d` |
|---|---:|---:|---:|---:|
| `C1-absgrad-t08-v1` | `true` | `true` | `false` | `0.0008` |
| `C1-absgrad-t08-revopacity-v1` | `true` | `true` | `true` | `0.0008` |

`0.0008` is fixed for C1. The installed `gsplat==1.4.0` documentation states
that AbsGrad requires a higher threshold than signed-average gradient and gives
`0.0008` as the example value. C1 does not add a threshold sweep.

Every other optimization value remains locked to the matching B0 stage:

- `max_steps=7000` during screening and `max_steps=30000` during confirmation
  and production;
- `resize_factor=1`;
- `seed=0`;
- internal holdout enabled for screening/confirmation and disabled for production;
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

### Phase A — two-scene 7k mechanism screen

Use existing B0-reference 7k artifacts for `HCM0421` and `HCM1439`. Run both C1
candidates fresh at 7k on both scenes. These scenes provide one exact submission
scene and one additional BTS scene while retaining existing paired controls.

New runs: four. The committed six-scene B0-reference aggregate took 5,555.62
seconds, or 15.43 minutes per 7k scene on average. On that measured anchor, the
planning estimate is about 1.03 sequential L4 GPU-hours.

### Phase B — six-scene 7k robustness screen

If one candidate wins on both Phase-A scenes, lock it and run it fresh at 7k on
the remaining four qualification scenes: `hcm0031`, `HCM0181`, `HNI0131`, and
`HNI0265`. Compare against their existing paired B0-reference artifacts. No
second candidate is run in Phase B.

New runs: four; planning estimate about 1.03 L4 GPU-hours. Together, Phases A and
B evaluate the winner across all six established screening scenes for about 2.06
GPU-hours, without retraining any B0 control. AbsGrad may grow more Gaussians, so
these are planning anchors rather than runtime guarantees.

### Phase C — fresh 30k confirmation

If the winner passes the six-scene screen, train it from scratch for 30k on
`HCM0181` with the exact holdout used by the existing committed B0 30k artifact
from commit `411c8de`. That reference has PSNR `22.6915559431`, SSIM
`0.8053435291`, LPIPS `0.1112587926`, 11.79 GB peak VRAM, and 2.90 L4 GPU-hours.

The 7k checkpoint is not resumed: the learning-rate schedule horizon and config
hash differ, so a fresh run is required for a controlled 30k comparison.

New runs: one; planning estimate about 2.9 L4 GPU-hours. Total pre-production
budget is therefore about 4.96 GPU-hours, subject to measured candidate growth.

### Phase D — seven-scene production retraining

If Phase C passes, train the locked winner from scratch for 30k on exactly
`HCM0644 HCM0674 HCM0540 HCM0539 HCM0421 chair bonsai`, using the production
contract (`internal_holdout=false`) and a new candidate/baseline ID. Then render,
validate, and package all seven scenes. Phase C is only a compute-risk gate;
Phase D is the actual deployment experiment.

Using 2.90 hours as a conservative per-scene anchor, Phase D is roughly 20.3
sequential L4 GPU-hours. Actual scene times and peak VRAM must be recorded.

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

### Run eligibility

A candidate is eligible only when all conditions hold:

- final validation artifacts cover every internal validation image;
- PSNR, SSIM, LPIPS, Score50, HF-L1, missing-edge, and spurious-edge are finite;
- final metrics are exactly at the stage horizon (7k or 30k) and config/manifest
  hashes match;
- no CUDA OOM, non-finite loss, or invalid Gaussian state occurred;
- peak VRAM is below 23 GB;
- wall time and Gaussian trajectory are recorded.

### Phase A candidate lock

For each candidate, compute paired `ΔScore50` against B0 on HCM0421 and HCM1439.

1. A candidate must have `ΔScore50 > 0` on both scenes.
2. It is rejected if missing-edge and spurious-edge both worsen on either scene.
3. If both remain eligible, select the larger two-scene mean `ΔScore50`.
4. A numerical tie is broken by lower mean HF-L1, then lower peak Gaussian
   count, then `C1-absgrad-t08-v1` as the simpler method.

### Phase B six-scene gate

The locked candidate passes screening only when:

- mean paired `ΔScore50 > 0` across all six scenes;
- at least four of six scene deltas are positive;
- aggregate LPIPS does not worsen;
- missing-edge and spurious-edge do not both worsen in aggregate;
- every run passes integrity and resource gates.

Report all six paired deltas and an exact two-sided sign-test p-value as
descriptive evidence. With six scenes the strongest possible result is
`p=0.03125`; no broader generalization claim is made from one fixed seed.

### Phase C 30k gate

Against the existing HCM0181 B0 30k internal-holdout reference, require:

- `ΔScore50 > 0`;
- LPIPS does not increase;
- missing-edge and spurious-edge do not both worsen;
- integrity/resource gates pass.

Only then may Phase D retrain all seven production scenes. A Phase-C failure
means the 7k gain did not survive the final training horizon.

## 8. L4 CUDA execution contract

The experiment reuses the repository's backend qualification rather than
hard-coding a precision mode.

Before C1:

```bash
bash scripts/qualify_training_backend.sh
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
- no model checkpoint during 7k screening; metrics and timing remain durable;
- one rolling atomic `recovery.pt` every 3,000 steps during 30k confirmation and
  production; it overwrites the same path rather than accumulating milestones;
- no `torch.compile`, TF32 override, multi-GPU synchronization, custom CUDA
  kernels, or dependency upgrade in C1.

The exclusions keep the implementation small and prevent CUDA optimization
changes from becoming extra experimental factors.

## 9. Software design

### 9.1 Configuration boundary

Extend the exact allow-list behind `--qualification_candidate` with the two C1
IDs. Preserve its existing hard contract: fresh run, factor 1, seed 0, 7k,
internal holdout, cached images, pinned transfer, and no checkpoints.

Add a separate full-length research candidate mode for Phase C. It requires
factor 1, seed 0, 30k, internal holdout, candidate ID in the config hash, and
resume only from `<output_dir>/checkpoints/recovery.pt`. Production Phase D uses
the existing production orchestration with a new immutable candidate ID.

The full-length research mode remains mutually exclusive with profile, backend
qualification, 7k qualification, and B0 full-length qualification modes.

### 9.2 Rendering and density adapter

Extend existing function signatures only:

- `render_gaussians(..., absgrad: bool = False)`;
- `GsplatStrategy(..., absgrad: bool = False, revised_opacity: bool = False)`.

Defaults preserve B0 behavior. The trainer reads both values from its immutable
config and passes them to the renderer and strategy.

### 9.3 Diagnostics and decision artifact

Add one focused high-frequency metrics module and one C1 decision module. The
per-step `metrics.jsonl` remains the primary training record and stores loss,
Gaussian count, means learning rate, and sample index; `timing.json` stores the
timing breakdown. The deterministic decision JSON contains:

- candidate IDs and source run paths;
- config and manifest hashes;
- raw PSNR/SSIM/LPIPS and Score50;
- high-frequency diagnostic aggregates;
- time, VRAM, peak/final Gaussian count;
- per-gate booleans;
- selected candidate or explicit rejection reason.

### 9.4 Runner

Use the shared Python screening entry point; no C1 shell wrapper is added. The
Phase-A invocation is:

```bash
python src/bts_nvs/training/run_c1_screening.py \
  --stage phase-a \
  --repo_root "$PWD" \
  --scenes_root "$PWD/data/bts_scenes" \
  --manifests_root "$PWD/runs/manifests" \
  --backend_root "$PWD/runs/phase4/backend_qualification" \
  --baseline_root "$PWD/runs/phase4/qualification" \
  --output_root "$PWD/runs/c1/phase_a"
```

The runner:

- verifies the repository root and required manifests;
- loads the accepted backend decision;
- creates candidate-specific output directories without overwriting B0;
- never resumes or saves model checkpoints for 7k screening;
- runs the four Phase-A candidate/scene pairs sequentially;
- computes the Phase A decision;
- stops before Phase B when no candidate passes;
- stops after emitting the Phase-A decision. Later stages require an explicit,
  separately reviewed invocation.

## 10. Failure handling

- Existing non-empty 7k run directories are rejected. Existing non-empty 30k
  directories require a valid rolling recovery checkpoint with matching hashes.
- CUDA OOM, non-finite values, missing validation renders, incomplete metrics,
  or failure to reach the stage horizon prevent selection.
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
- 7k runner commands contain factor 1, 7k, seed 0, cache, pinned transfer,
  accepted backend/precision, and checkpoint saving disabled;
- 30k runner commands contain factor 1, 30k, 3k rolling recovery, seed 0,
  cache, pinned transfer, and the same accepted backend/precision;
- decision logic enforces two-scene locking, six-scene gating, and 30k gating.

GPU verification on the VM must include:

1. existing real gsplat CUDA forward/backward smoke;
2. a bounded C1 candidate smoke that reaches one densification event and confirms
   `means2d.absgrad` exists and all gradients are finite;
3. the four 7k Phase-A runs only after the smoke passes.

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
