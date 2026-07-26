# Five-scene MVP authority

**Status:** Active from 2026-07-26.

This document is the concise scientific and execution authority for the next
MVP. `AGENTS.md` still governs data, output, metric and reproducibility
constraints. Older optimization and runner specs are historical provenance,
not required reading for this MVP.

## 1. Baseline and scope

- Closed official baseline: `B0-submission-q99-v1`, Score `70.98330`.
- `HCM0421` and `HCM0539` have deadline-exception AbsGrad production
  checkpoints. They are operational artifacts, not paired-confirmed winners.
- This MVP covers the other five scenes:

```text
HCM0644 HCM0674 HCM0540 chair bonsai
```

- Hidden-test submission renders are never used for candidate selection.
- Unit of comparison is one scene, one fixed holdout, seed 0 and 7,000 steps.

## 2. Evidence and policy

| Scene | B0 Score50 | LPIPS | Hard−easy | Policy |
|---|---:|---:|---:|---|
| HCM0644 | 69.155 | 0.1737 | -9.225 | Freeze at B0; no new screen |
| HCM0674 | 62.826 | 0.2316 | -16.046 | Screen antialiased rasterization |
| HCM0540 | 65.804 | 0.2060 | -14.899 | Screen antialiased rasterization |
| chair | 58.961 | 0.3559 | -3.745 | Screen bounded local sharpness weighting |
| bonsai | 58.529 | 0.3927 | -19.454 | Screen weighting and SH4 separately |

Missing-edge error dominates spurious-edge error in all five scenes. Global
opacity sparsity, unconstrained random initialization, camera refinement and
external monocular-depth priors are out of scope.

## 3. Locked candidates

Every candidate changes one mechanism relative to `B0-reference`.

### `E2-raster-aa-v1`

- `rasterize_mode=antialiased`;
- all density, loss, SH and sampling settings remain B0;
- authorized screen scenes: `HCM0674`, `HCM0540`.

### `E2-loss-local-laplacian-v1`

- camera sampling remains uniform;
- precompute an aligned uint8 confidence map after undistortion and resize;
- confidence is local variance of a 3×3 Laplacian, pooled over 31×31 patches;
- normalize log-variance between the valid-pixel p20 and p80;
- map range is `[0.5, 1.0]`; invalid pixels remain invalid;
- weight only the L1 term; DSSIM remains unchanged;
- authorized screen scenes: `chair`, `bonsai`.

This bounded formulation cannot delete low-texture supervision and does not
claim that low Laplacian always means blur.

### `E2-appearance-sh4-v1`

- allocate 25 RGB SH bases per Gaussian instead of 16;
- activate degrees `0,1,2,3,4` at steps `0,1000,2000,3000,4000`;
- all other settings remain B0;
- authorized screen scene: `bonsai`.

## 4. Run and decision contract

- Reuse the completed `runs/scene_opt_v1/reference/<scene>/` B0 reports.
- Write new candidate runs under `runs/scene_opt_v2/screen/`.
- Run one GPU process at a time.
- A screen is fresh, full resolution, internal holdout, seed 0, 7,000 steps,
  cached images and the accepted backend/precision.
- Do not resume a screen and do not save a model checkpoint at 7k.

A candidate is eligible only when:

1. all expected renders and reports are complete and finite;
2. Score50 delta is strictly positive;
3. LPIPS does not increase;
4. hard-stratum Score50 does not decrease;
5. missing-edge and spurious-edge do not both worsen;
6. wall-time ratio is at most `1.25`;
7. peak VRAM is below 23 GB and primitive growth remains controlled.

If both bonsai candidates pass, choose larger Score50 gain, then lower LPIPS,
then lower symmetric-edge distance. If none pass, retain B0.

## 5. Confirmation boundary

A 7k winner is mechanism evidence only. It must not replace a production
baseline until one fresh 30k B0 and one fresh 30k candidate run pass the same
holdout gates. With the current deadline, confirm at most two winners. Every
model, loss, renderer or SH change uses a new candidate/baseline ID.

## 6. Completion

This MVP is complete when:

- code and tests support all three locked candidates;
- HCM0644 has a recorded B0 decision;
- five authorized candidate runs across four scenes complete or have a durable
  failure record;
- bonsai has one deterministic decision across its two candidates;
- selected winners, fallback decisions, timings and blockers are summarized;
- no production or submission artifact is silently overwritten.
