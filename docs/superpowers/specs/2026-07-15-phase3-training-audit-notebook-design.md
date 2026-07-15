# Phase 3 Training Audit Notebook Design

## Goal

Create a reader-facing, reproducible notebook that audits Phase 3 training
artifacts and makes image quality, convergence, resource use, and remaining
risks visually obvious. The notebook supports the completed HCM0181 Run A and
Run B now, and accepts an optional full-resolution pilot run later without code
changes.

This work does not change the trainer, checkpoint format, optimization
mathematics, or full-training CLI.

## Inputs

The notebook exposes one parameter cell containing:

- repository root;
- Run A artifact directory;
- Run B artifact directory;
- optional pilot artifact directory;
- rolling-window size for curves;
- preview camera index, currently fixed to the saved train camera index 0.

Each run is read only from `config.yaml`, `environment.json`, `summary.json`,
`convergence.json`, `metrics.jsonl`, `timing.json`, `train_previews/`, and
checkpoint file metadata. Checkpoint tensors are not loaded because the audit
does not require several gigabytes of model state.

## Validation Contract

Before plotting, the notebook checks:

- required files exist and contain valid JSON/YAML;
- metric steps are finite, unique, ordered, and cover `1..total_steps`;
- loss, learning rate, Gaussian count, and timing values are finite;
- summary step count agrees with metrics and timing;
- preview images decode, share the expected resolution, and are non-empty;
- the final checkpoint exists and has non-zero size;
- the recorded device and CUDA environment are present.

Failures appear as a visible blocker table near the top. A run with a blocking
integrity failure is not used to make readiness claims.

## Visual Story

The notebook follows this order:

1. **TL;DR and gate scorecard** — compact pass/caveat/block table for artifact
   integrity, convergence, image alignment, resource headroom, storage, and
   novel-view evidence.
2. **Run overview** — comparable cards for resolution, steps, elapsed time,
   final loss, peak VRAM, final Gaussian count, PSNR, and SSIM.
3. **Image reconstruction panel** — GT, initialization, and final render on one
   row with identical scale and no axes.
4. **Error analysis** — final absolute RGB error heatmap, initialization error
   heatmap, per-channel absolute error, and selected detail crops. Heatmaps use
   the same fixed scale so improvement is visually comparable.
5. **Training dynamics** — raw loss with a rolling median, Gaussian count,
   position learning rate, and milestone annotations at densification start,
   opacity resets, SH degree changes, and checkpoints.
6. **Performance and storage** — total step-time distribution, rolling
   throughput, checkpoint-size growth, peak VRAM, and projected storage cost.
7. **Decision and caveats** — evidence-backed recommendation with explicit
   separation between train-view reconstruction and unseen-view quality.

Charts use descriptive titles, labeled axes and units, consistent run colors,
bounded legends, and no raw log dumps. Tables show only decision-relevant rows.

## Metric Semantics

The masked PSNR and SSIM in `convergence.json` remain the authoritative smoke
metrics because they use the training valid mask. The notebook independently
recomputes image-space PSNR and absolute-error maps from the saved PNGs as a
visual consistency check and labels them as unmasked, PNG-quantized metrics.
It does not present these recomputed values as leaderboard-comparable.

LPIPS is omitted from the current artifact-only audit because the run saved one
train camera and the grading backbone is not confirmed. It may be added to a
future held-out validation protocol after its weights and implementation are
locked.

## Readiness Rules

The completed smoke run passes the Phase 3.6 engineering gate when:

- all artifact-integrity checks pass;
- all steps are present and finite;
- final render is non-blank and geometrically aligned with GT;
- masked PSNR and SSIM improve materially;
- training completes without NaN, OOM, or checkpoint failure;
- VRAM remains below the configured safety ceiling.

It does not by itself pass the novel-view quality gate because all available GT
images participated in training and the saved comparison is train camera 0.
The notebook must state this caveat prominently. Full multi-scene training is
not recommended until the full-resolution pilot passes and a deterministic
internal holdout evaluation has been run without using official test data.

## Pilot Extension

When a pilot directory is supplied, the same integrity, convergence,
performance, image, and storage panels add it automatically. The comparison
must emphasize:

- factor-1 peak VRAM and runtime versus factor-2;
- Gaussian growth and checkpoint size;
- train-view quality change;
- whether resource headroom remains safe for longer densification.

Missing pilot artifacts are shown as `not available`, not treated as a current
failure.

## Verification

- Execute the notebook top to bottom in the project environment.
- Confirm no cell errors and no unbounded debug output.
- Inspect all rendered charts and image panels for clipping, inconsistent
  scales, misleading axes, or unreadable labels.
- Reconcile headline values against `summary.json` and `convergence.json`.
- Preserve `runs.tar.gz` and all run artifacts as read-only user data.
