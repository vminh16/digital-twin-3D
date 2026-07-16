# Phase 4.3 Holdout and Profile Gate Fix

## Decision

Keep guard cameras during internal qualification. Guard cameras are deliberately
excluded from both training and validation metrics so near-duplicate camera
poses cannot make validation optimistic. After hyperparameters are locked, a
production retrain may omit internal holdout mode and use all physical train
images, which effectively merges train, guard and validation again.

## Artifact preparation

Each available scene uses one artifact directory:

```text
<manifests_root>/<scene_id>/
├── manifest.json
├── arrays.npz
└── holdout.json
```

`prepare_phase4_artifacts.py` scans physical scene directories in sorted order.
It builds a missing manifest or validates an existing one, then deterministically
builds and atomically saves `holdout.json`. The script supports an optional
expected scene count and a strict mode for the future full 13-scene cohort. It
does not create a new sparse artifact schema.

## Training modes

`run_training.py --internal_holdout` loads `holdout.json` from the selected
manifest directory. `--profile_input` implies this mode and fails before
training if the holdout artifact is absent or stale.

Internal holdout mode:

- samples only `split.train_image_names`;
- excludes guard and validation names from the sampler;
- initializes geometry/colors through
  `build_split_sparse_initialization(manifest, scene_root, split)`;
- records holdout algorithm, manifest hash, and train/guard/validation counts
  in the immutable training config.

Normal mode remains unchanged and uses the full manifest. This is the explicit
production retrain path after qualification.

## Profile equivalence

The performance window remains 50 warm-up plus 500 measured steps. Sample
indices and training identity must match for all 500 measured steps.

Loss and Gaussian-count equivalence are evaluated only on completed steps before
the first topology-changing density refinement. With the current baseline this
is measured steps 51 through 499. Step 500 remains part of performance timing,
but its post-refinement Gaussian count is diagnostic because CUDA gradient
accumulation can move a threshold-borderline Gaussian to either side. The report
records the compared prefix and final Gaussian-count delta. Loss equivalence
uses `rtol=5e-4, atol=1e-6` to tolerate small independent-run CUDA drift while
remaining strict enough to reject a changed optimization trajectory.

## Acceptance

- Batch preparation succeeds on the currently available HCM0181 scene and can
  later require exactly 13 scenes.
- Profile artifacts fail fast when `holdout.json` is missing or stale.
- HCM0181 sampler domain is exactly 169 internal-train images; no guard or
  validation image can be sampled.
- Split sparse initialization uses internal-train observations only.
- Cached and uncached profiles compare the same 500 sampled indices and the same
  pre-refinement optimization trace.
- Performance gate remains: at least 10% median wall-step speedup or cached CPU
  preprocessing below 10% of measured wall time.
- No batch-size, optimizer, renderer, density strategy, loss, or dependency
  changes.

## Self-review

- No guard-removal ambiguity remains: guard is qualification-only and full-data
  production retrain is an existing explicit mode.
- No new sparse serialization format or compatibility layer is introduced.
- Density refinement remains enabled and timed; only the semantic-equivalence
  boundary is corrected.
- No placeholder or deferred interface remains.
