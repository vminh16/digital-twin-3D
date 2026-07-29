# Chair and bonsai deep-optimization authority

**Status:** ACTIVE — chair E5 rejected; E6 perceptual staged research
implemented, L4 15k run pending.

**Plan ID:** `scene-opt-v3-chair-bonsai`

This is the only active research authority after submission
`MVP-hybrid-4scene-q99-v1`. `AGENTS.md` remains authoritative for data,
evaluation, production and output contracts.

## 1. Closed starting point

- `B0-submission-q99-v1` is immutable: official Score `70.98330`.
- `MVP-hybrid-4scene-q99-v1` is submitted and closed: official Score
  `71.2124`, PSNR `24.629191`, SSIM `0.807208`, LPIPS `0.194533`, 7/7 scenes.
- Stage A, Stage B1, the five-scene screen, four override productions,
  rerender and hybrid assembly are historical and must not be reopened.
- `HCM0644`, `HCM0674`, `HCM0540`, `HCM0421` and `HCM0539` are frozen during
  this plan.
- Official test RGB and official per-scene metrics are unavailable and cannot
  be used for candidate selection.

## 2. Evidence that opens this plan

The full-image audit covered 453 train images and 86 submission renders.

### Chair

- 5/58 outputs were catastrophic; four form the `855–895` close-mesh cluster.
- Adjacent train images are sharp and pose distance does not predict severity.
- Source disagreement predicts severity, consistent with an occlusion and
  projected-scale instability.
- Primary failure: oversized or excessively elongated Gaussians becoming
  visible over a narrow view interval.
- Secondary failure: seven strong-motion-blur train images cause mild
  softness, not the catastrophic veil.

### Bonsai

- 10/28 outputs were major or catastrophic.
- Nearest-train pose gap strongly predicts severity; all outputs with
  normalized gap at least 2 were non-clear.
- The pot often remains stable while the table, legs, carpet and foreground
  become translucent or geometrically inconsistent.
- Sharp internal-holdout frame `390` already failed at 7k; SH4 did not repair
  geometry.
- Primary failure: weakly constrained depth and multi-view geometry around the
  reflective table under local pose gaps.
- Secondary failure: SH cannot fully represent sharp view-dependent
  reflections.

The repository currently disables world-scale pruning with
`prune_scale3d=inf`. Densification stops at 15k. More steps alone therefore do
not repair missing or malformed topology.

## 3. Active method families

The candidate-theory evidence is recorded in
`../research/2026-07-27-chair-bonsai-candidate-validation.md`.

### Method C — chair: repair initialization before density control

Goal: remove the proved sparse-color coordinate defect, then repair any
remaining large-Gaussian gradient collision without deleting useful splats.

Candidate ladder:

1. `E3-chair-observation-scale-v1` — implemented research control.
   - inherit the current chair incumbent
     `E2-loss-local-laplacian-v1`;
   - replace integer-ceiling observation-scale inference with a robust
     continuous reprojection mapping;
   - prove that current image coordinates, intrinsics and points agree after
     mapping;
   - log old/new sparse-color error before Gaussian parameter allocation;
   - change no loss, density or appearance setting.
   - allow only `stage=research, scene=chair`; confirm and production must
     reject this ID because production already uses `points3D.bin` colors.
2. `E4-chair-observation-scale-absgrad-v1` — implemented research candidate.
   - inherit the passing observation-scale candidate;
   - add the already-supported AbsGrad `t04` density policy;
   - run only if giant-radius sentinel failure remains.
   - allow only `stage=research, scene=chair`; confirm and production reject
     this ID.
   - paired 15k evidence rejects it: holdout Score50 `-1.4725`, PSNR
     `-0.9794 dB`, SSIM `-0.02293`, with 2/20 Score wins despite lower train
     loss and 65% more final Gaussians.
3. `E5-chair-observation-scale-mcmc-v1` — implemented full-horizon research
   candidate.
   - inherit E3 observation mapping and local-Laplacian supervision;
   - replace DefaultStrategy with gsplat MCMC relocation/SGLD;
   - cap at 2M Gaussians, use noise LR `5e5`, relocate through step 25k;
   - add opacity regularization `0.001` and scale regularization `0.01`;
   - require a fresh 30k internal-holdout run with rolling recovery every 3k;
   - allow only `stage=research, scene=chair`; confirm and production reject
     this ID until a new decision artifact authorizes a later candidate.

Exact Pixel-GS is a second-line method. An ellipse-area or tile-count proxy is
not allowed to use the Pixel-GS name because current `gsplat` does not expose
its exact post-transmittance pixel-participation count.

### Method B — bonsai: geometry-consistent reflective appearance

Goal: prevent early floater overgrowth, recover split signals for large
Gaussians and then add only the geometry supervision supported by provided
observations.

Candidate ladder:

1. `E3-bonsai-c2f-absgrad-sh4-v1` — reserved, not executable yet.
   - inherit the current bonsai incumbent `E2-appearance-sh4-v1`;
   - use the Spec-Gaussian density curriculum: start with both image dimensions
     downsampled by factor 4, reach full resolution by step 5k and use absolute
     projected gradients;
   - retain SH4 unchanged.
2. `E4-bonsai-c2f-absgrad-sh4-sparse-depth-v1` — reserved, not executable yet.
   - inherit the passing c2f candidate;
   - add robust sparse-depth anchors from the provided COLMAP tracks;
   - use confidence from track length and reprojection error;
   - correct legacy observation coordinates before projecting any anchor;
   - retain SH4 as the appearance component.

External monocular-depth weights, diffusion priors and external images remain
out of scope. Sparse depth cannot be claimed to supervise the table interior:
the frame-390 tabletop ROI has only half the image-average SfM point density.
2DGS/PGSR, exact Pixel-GS, MCMC, ASG, GaussianShader, HMGS and SpecTRe-GS are
documented fallbacks and are not authorized in the first ladder.

## 4. Unified research harness

Historical `reference` and `screen` stages retain their exact 7k semantics.
The active harness adds:

```text
stage          = research
authorized     = chair, bonsai only
schedule       = max_steps 30000
stop_step      = 15000
holdout        = true
checkpoint     = none
experiment root= runs/scene_opt_v3
```

The 30k schedule with a 15k stop preserves the production learning-rate
trajectory. It replaces the old practice of setting `max_steps=7000`, which
made a screen follow a different means-LR schedule from production.

The operational wrappers are:

```bash
bash scripts/run_chair_bonsai_research.sh <chair|bonsai> <candidate-id>
bash scripts/run_chair_mcmc_research.sh
```

The MCMC wrapper is an explicit exception to the 15k research runtime:

```text
candidate       = E5-chair-observation-scale-mcmc-v1
stop_step       = 30000
checkpoint      = rolling recovery every 3000
Gaussian cap    = 2000000
```

The exception is necessary because its relocation policy remains active to
25k. It must not relax the 15k/no-checkpoint contract for other candidates.

## 5. Validation design

Research selection uses train RGB only.

- Chair sentinel coverage must include sharp close-mesh train poses in the
  `840–920` interval, including a held-out interpolation challenge around
  frames `870/885`.
- Bonsai must include sharp frame `390` and algorithmically selected
  high-nearest-pose-gap train cameras.
- The existing generic pose-FPS holdout remains provenance; a v3 targeted
  holdout extension requires a separate hash and must never rewrite old
  `holdout.json`.

Implemented Phase 1 contract:

- artifact: `holdout_research_v3.json`;
- algorithm: `targeted_pose_guard2_v3`;
- chair sentinels: deterministic nearest available frames to 870 and 885
  within 840–920;
- bonsai sentinels: exact frame 390 plus the four cameras with largest
  all-camera nearest-pose gap;
- diagnostic thresholds: normalized max-axis scale `0.1` and projected radius
  `128 px`; these observe tails and do not define pruning policy;
- `gaussian_diagnostics.json` records scale/opacity quantiles, conservative
  max-over-view radius quantiles, opacity mass above both thresholds and exact
  observed net Gaussian-count changes;
- `diagnostic_filtered_renders/` suppresses over-threshold radius primitives
  per validation camera and is marked diagnostic-only;
- schema-2 research reports add worst-decile LPIPS/edge distance and
  deterministic veil/collapse flags. Historical reports remain schema 1.

The current `gsplat` strategy API does not expose separate clone/split/prune
counts. Phase 1 records exact before/after count deltas instead of assigning
unverifiable backend event labels.

Candidate gates:

1. all artifacts, renders, metrics and hashes are complete and finite;
2. mean Score50 improves against the paired incumbent;
3. LPIPS does not worsen;
4. hard-stratum Score50 does not worsen;
5. worst-decile LPIPS and symmetric-edge error improve;
6. catastrophic/veil sentinel count is zero;
7. missing-edge and spurious-edge do not both worsen;
8. peak VRAM remains below 23 GB;
9. paired wall time remains at most 1.25;
10. scale/radius tails and Gaussian growth remain controlled.

Mean improvement cannot override a failed tail gate.

## 6. Step and production policy

- 15k research evidence selects ordinary mechanisms only. E5 MCMC requires
  30k because its transition policy remains active after 15k.
- E3-30k is not an upfront prerequisite. First compare E5-30k against the
  existing E3-15k lower bound. If E5 does not win clearly, reject it without
  spending a fresh E3-30k run.
- The lower-bound gate means `Delta Score50 >= +0.75`, mean LPIPS no worse,
  hard-stratum Score50 no worse, and no regression on the 870/885 tail pair.
- If E5 clears that lower-bound gate, a fresh E3-30k paired control becomes
  mandatory before attributing the gain to MCMC or authorizing an MVP.
- The old `1.25x` screen-time rule is not applied across a 30k-vs-15k
  comparison. On the eventual paired 30k comparison: up to `1.5x` wall time
  requires at least `+0.5` Score50; `1.5x–2.0x` requires at least `+1.0`;
  above `2.0x` is rejected for this deadline.
- A winner receives a fresh paired 30k internal-holdout confirmation.
- Production is a separate fresh 30k full-data run with no internal holdout.
- A 30k winner may receive one explicit 5–10k low-LR polish experiment only
  when the 15k→30k held-out curve is still improving and scale tails are
  stable.
- A 70k run is not authorized. Changing `max_steps` to 70k changes the LR
  trajectory and is not a continuation of the closed 30k policy.

## 7. Active phase

```text
Phase 0  CLOSED  close old phases; add research stage and unified wrapper
Phase 1  CLOSED  two complete L4 incumbent runs and reports validated
Phase 2A CLOSED   chair E3/E4 paired mechanism evidence
Phase 2B CLOSED   chair E5 MCMC rejected at lower-bound gate
Phase 2C CLOSED   chair E6 perceptual under-densification rejected
Phase 2D ACTIVE   chair E7 corrected perceptual ADC staged research
Phase 3  PENDING  conditional fresh E3-30k paired confirmation
Phase 4  PENDING full-data production, rerender and new submission assembly
```

E3 through E7 remain chair/research-only IDs. E5, E6 and E7 are executable
only through their locked wrappers. The former `E3-*-scale-guard-*` names are
superseded and must not be registered.
## E6 perceptual densification

E5 completed 30k but failed its lower-bound gate, so E3-30k remains unspent.
The next registered candidate is
`E6-chair-observation-scale-perceptual-v1`, restricted to `chair/research`.

E6 keeps the E3 observation mapping, local-Laplacian RGB loss, SH3 and holdout.
It adds deterministic binary sensitivity maps from internal-train images only,
one learned sensitivity logit per Gaussian, a BCE sensitivity render branch,
high/medium sensitivity-guided density control, clone opacity decline with
exponent `1.2`, and a hard safety cap of `2.1M` Gaussians. Sensitivity maps use
gamma `1.5`, Sobel threshold `0.05`, `5x5` average pooling and threshold `0.3`.

The gsplat port measures each sampled-view Gaussian contribution as the
derivative of the summed sensitivity render with respect to Gaussian
sensitivity color. This is the accumulated alpha-compositing weight supported
by the installed renderer; it is not claimed to be byte-equivalent to the
official custom CUDA `render_imp` implementation.

E6 uses a 30k optimizer schedule with a durable 15k stop:

```bash
bash scripts/run_chair_perceptual_research.sh
E6_STOP_STEP=30000 bash scripts/run_chair_perceptual_research.sh
```

The second command is authorized only after review of the 15k gate. Depth
reinitialization, Spectral-GS, MCMC and GaussianSpa are forbidden in E6.

## E7 corrected perceptual ADC

E6 15k is rejected and must not be resumed as the authoritative
Perceptual-GS test. It ended with about 297k Gaussians because its `_grow_gs`
override replaced standard 100-step ADC with HD/MD-only events. The official
implementation retains normal `densify_and_prune` and ORs perceptual masks
into clone/split at the 1000/1500-step events:

- paper: <https://arxiv.org/html/2506.12400>
- official training loop:
  <https://github.com/eezkni/Perceptual-GS/blob/main/train.py>
- official density control:
  <https://github.com/eezkni/Perceptual-GS/blob/main/scene/gaussian_model.py>

`E7-chair-perceptual-adc-corrected-v1` is a fresh `chair/research` candidate
with these locked invariants:

1. Keep the E3 continuous observation mapping, local-Laplacian loss, SH3,
   fixed internal holdout and 30k optimizer horizon.
2. Preserve standard gradient ADC at every 100-step refine event.
3. At HD/MD events, OR contribution-qualified sensitivity masks into the
   standard clone/split masks. Perceptual additions do not require the
   positional gradient threshold a second time.
4. Compute each event contribution as the maximum over all internal-train
   views. The gsplat derivative remains the accumulated alpha-compositing
   weight adapter described for E6.
5. Keep the published thresholds and intervals: sensitivity `0.9/0.3`,
   contribution `25/10`, intervals `1000/1500`, BCE weight `0.1`, opacity
   exponent `1.2`.
6. Keep the local L4 safety cap at 2.1M. If capacity is exhausted, baseline ADC
   has first claim before perceptual-only additions.
7. Scene-adaptive depth reinitialization remains out of scope. The paper's
   `w/o SDR` ablation isolates the corrected density mechanism with materially
   less implementation risk.

Run the fresh 15k mechanism gate with:

```bash
bash scripts/run_chair_perceptual_adc_research.sh
```

Only after review may the same run resume to 30k:

```bash
E7_STOP_STEP=30000 bash scripts/run_chair_perceptual_adc_research.sh
```

At 15k, reject the implementation if it remains below 1.2M Gaussians or if
the event contribution/ADC diagnostics are absent. Continue only when LPIPS
and easy-view tail no longer show E6's clear regression, the hard stratum
retains its gain, and scale/projected-radius tails do not worsen.
