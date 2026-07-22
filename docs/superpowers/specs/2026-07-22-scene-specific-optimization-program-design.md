# Scene-Specific Seven-Scene Optimization Program Design

**Date:** 2026-07-22

**Status:** Approved umbrella design; module implementation is gated by this specification.

**Execution branch:** `main`; this program does not create a new experiment branch.

**Baseline authority:** `B0-submission-q99-v1` remains CLOSED and immutable.

**Historical evidence:** `origin/ex1/absgrad-revopacity-phase-a@144ade1` remains read-only evidence for C1 implementation and results.

## 1. Objective

Build a staged optimization program for exactly the seven submission scenes:

```text
HCM0644 HCM0674 HCM0540 HCM0539 HCM0421 chair bonsai
```

Each scene is trained independently, so the program may select a different
approved training policy for each scene. The final production cohort may mix
new candidates and `B0-reference`; a scene keeps B0 whenever no candidate
passes its validation gate.

The program must improve image quality, geometric fidelity, and held-out pose
robustness without creating seven unrelated pipelines. It therefore adds a
small reusable experiment layer around the existing manifest, holdout,
training, evaluation, checkpoint, inference, and submission contracts.

## 2. Mathematical basis

Let `theta_s` be the parameters for scene `s` and `R_s` its expected rendering
risk. With no shared model parameters or cross-scene regularizer, the cohort
objective decomposes:

```text
min_(theta_1,...,theta_7) (1/7) * sum_s R_s(theta_s)
= (1/7) * sum_s min_(theta_s) R_s(theta_s)
```

Using one hyperparameter set for every scene is therefore an engineering
constraint, not a mathematical requirement. Candidate selection must still be
performed on a deterministic internal holdout:

```text
h_s* = argmax_h estimated_score_validation(s, h)
```

The production baseline uses `internal_holdout=false`. Its convergence preview
is a re-rendered train camera and is not a model-selection metric. A production
run cannot serve as the paired validation reference for this program.

The local training objective remains:

```text
0.8 * L1 + 0.2 * (1 - SSIM)
```

Candidate decisions track PSNR, SSIM, LPIPS, and the repository-local
composite. The local composite is diagnostic and must never be described as an
official score until `PSNR_max`, LPIPS backbone, and SSIM implementation are
confirmed against the grading harness.

## 3. Selected architecture

The selected approach is a thin generic experiment layer around the current
pipeline.

Rejected alternatives:

1. Merging the complete C1 branch would import phase-specific runners,
   rejected `t08 + revised_opacity` policy, and large run artifacts.
2. Writing one runner/config stack per scene would create operational drift and
   make paired comparisons difficult to audit.

The selected design:

- preserves B0 defaults and output behavior;
- reuses existing code by import rather than copying logic;
- extracts only generic, tested ideas from the historical C1 branch;
- adds renderer/trainer hooks only when a candidate requires them;
- uses one Python orchestration path for all stages and scenes;
- gives every module an independent test, review, and commit gate.

## 4. Global constraints

- Work directly on `main`; do not create a branch for this program.
- Do not modify, rewrite, or silently relabel `B0-submission-q99-v1`.
- Do not access held-out test RGB or use official test results as a tuning
  signal.
- Do not use external images, pretrained depth, pretrained segmentation, or
  external 3D assets.
- Do not alter the submission filename, dimension, payload, JPEG Q99, 4:4:4,
  optimized, non-progressive, or archive validation contracts.
- A candidate changes one primary mechanism during screening.
- Every paired comparison uses the same scene, manifest, holdout, seed,
  resolution, step horizon, backend, precision, and metric configuration.
- A 7k run stores metrics, timing, validation renders, config, hashes, and
  decision inputs; it stores no model checkpoint.
- A 30k run stores one rolling atomic `recovery.pt`, overwritten at the
  checkpoint interval; it does not accumulate milestone checkpoints.
- One scene runs per GPU process. Target hardware is one NVIDIA L4 with less
  than 23 GB usable VRAM.
- No new Bash runner is added. Existing shell wrappers may invoke the generic
  Python CLI only when an operational wrapper is already required.
- Existing user changes in the worktree are not folded into experiment commits.

## 5. Repository boundaries

### 5.1 Reused without semantic change

| Existing component | Reuse contract |
|---|---|
| `bts_nvs.data.manifest` | Scene/data identity and hashes remain authoritative. |
| `bts_nvs.data.holdout` | Keep `pose_fps_guard2_v1`; do not create a second split algorithm in Module 1. |
| `bts_nvs.data.dataset` | Load train and validation samples through the existing dataset adapter. |
| `bts_nvs.evaluation.metrics` | Reuse PSNR, SSIM, LPIPS, and composite implementation. |
| `bts_nvs.training.full_training` | Reuse backend decision, rolling recovery, ledger, and trained-run validation. |
| `bts_nvs.training.run_training` | Remain the only process-level training entry point. |
| `bts_nvs.rendering.inference` | Production rendering remains unchanged. |
| `bts_nvs.submission` | Final validation and JPEG packaging remain unchanged. |

### 5.2 Selectively reused from historical C1

The following logic may be transplanted with renamed generic APIs and fresh
tests on `main`:

- high-frequency luminance/Sobel/Laplacian diagnostics;
- renderer and density-strategy forwarding for AbsGrad;
- AMP unscale of `means2d.absgrad`;
- atomic JSON writing, config hashing, provenance hashes, and completed-run
  integrity checks.

The following C1 content must not be imported:

- Phase A/B/C-specific runners and decision modules;
- C1 candidate IDs and `grow_grad2d=0.0008` lock;
- revised-opacity selection or production policy;
- C1 run directories, images, metrics logs, or checkpoints;
- C1 assumptions about six calibration scenes or HCM0181 authority.

## 6. Output contract

The program uses a shallow, stage-first experiment layout. Stage is part of
the path because a 7k screen and a 30k confirmation for the same
scene/candidate are different evidence and must coexist without overwrite:

```text
runs/scene_opt_v1/
├── experiment.json
├── reference/<scene_id>/
├── screen/<scene_id>/<candidate_id>/
├── confirm/<scene_id>/<candidate_id>/
├── production/<scene_id>/<candidate_id>/
├── decisions/
│   └── <scene_id>.json
└── cohort_decision.json
```

Each run directory contains only the standard training artifacts plus:

```text
detail_metrics.json
pose_strata.json
experiment_report.json
```

`experiment.json` pins schema version, experiment ID, seven-scene cohort,
backend decision hash, metric configuration, stage horizons, resource limits,
and allowed candidate IDs.

`decisions/<scene_id>.json` pins the paired inputs, deltas, gate booleans,
selected policy or explicit rejection, and all provenance hashes.

`cohort_decision.json` maps every scene to either an accepted candidate or
`B0-reference`. It is the only authority used by production orchestration.

## 7. Module 1 — validation foundation

### 7.1 Purpose

Create trustworthy, scene-aware measurements before changing training behavior.
This module is CPU-testable and does not modify the trainer or renderer.

### 7.2 Files and interfaces

```text
src/bts_nvs/evaluation/detail_metrics.py
src/bts_nvs/evaluation/pose_strata.py
src/bts_nvs/evaluation/experiment_report.py
```

`detail_metrics.py` provides image-pair diagnostics:

- `hf_l1` from luminance Laplacian error;
- `missing_edge` from reference-edge energy absent in the prediction;
- `spurious_edge` from predicted energy in reference-flat regions;
- symmetric edge distance, which penalizes both missing and displaced or
  hallucinated edges.

Reference edge pixels are the top 10 percent of fixed 3x3 Sobel magnitudes.
Reference flat pixels are the bottom 50 percent. Luminance uses
`0.299 R + 0.587 G + 0.114 B`. Invalid-mask pixels are replaced by reference
pixels before metrics are computed.

`pose_strata.py` measures each validation camera against its nearest retained
train camera using normalized camera-center distance and optical-axis angle.
Validation images are deterministically divided by ordered nearest-train
distance into three strata:

- `easy`: bottom third;
- `medium`: middle third;
- `hard`: top third.

Filename order breaks quantile-boundary ties. Official test poses may be
profiled geometrically but never have image metrics because test RGB is absent.

`experiment_report.py` combines existing full-frame metrics, detail metrics,
pose-stratified aggregates, timing, Gaussian counts, VRAM, config identity, and
manifest/holdout hashes. It rejects missing renders, extra renders, resolution
mismatches, non-finite values, and incomplete provenance.

### 7.3 Scope limit

Module 1 does not claim to segment the BTS. Without external models or trusted
labels, an automatic tower mask is not reliable enough for a decision gate.
High-frequency diagnostics are explanatory and veto-only. Full-frame
PSNR/SSIM/LPIPS/local composite remain the primary metrics.

### 7.4 Gate

Module 1 passes when unit tests prove:

- identical images return zero detail error;
- Gaussian blur increases missing-edge error;
- flat-region noise increases spurious-edge error;
- an edge shifted in image space increases symmetric edge distance;
- pose strata are deterministic, exhaustive, disjoint, and order-independent;
- report generation rejects invalid files, hashes, shapes, and numbers;
- existing evaluator and holdout tests remain green.

No GPU run is authorized by Module 1.

## 8. Module 2 — generic candidate registry and hooks

### 8.1 Package

```text
src/bts_nvs/experiments/__init__.py
src/bts_nvs/experiments/candidates.py
src/bts_nvs/experiments/contracts.py
src/bts_nvs/experiments/provenance.py
```

The immutable candidate settings interface is:

```python
@dataclass(frozen=True)
class CandidateSettings:
    candidate_id: str
    absgrad: bool
    grow_grad2d: float
    grow_scale3d: float
    prune_opa: float
    refine_stop_step: int
    rasterize_mode: str
    appearance_mode: str
    sampling_mode: str
```

Only fields in this dataclass may differ from B0. Optimizer learning rates,
loss weights, SH schedule, initialization, camera model, normalization, and
codec remain inherited B0 values.

### 8.2 First executable candidates

| Candidate ID | AbsGrad | `grow_grad2d` | `grow_scale3d` | Other fields |
|---|---:|---:|---:|---|
| `B0-reference` | false | 0.0002 | 0.01 | B0 |
| `E1-density-absgrad-t04-v1` | true | 0.0004 | 0.01 | B0 |
| `E1-density-scale005-v1` | false | 0.0002 | 0.005 | B0 |

AbsGrad and scale threshold are isolated so a gain can be attributed to one
mechanism. Revised opacity is excluded from the first executable registry.

### 8.3 Minimal hooks

- `render_gaussians(..., absgrad: bool = False, rasterize_mode: str = "classic")`;
- `GsplatStrategy(..., absgrad: bool = False, ...)`;
- AMP unscales signed projected gradients and `means2d.absgrad` when present;
- `Trainer` forwards immutable candidate settings without candidate-specific
  branches.

B0 defaults must produce the same effective config and calls as before Module 2.

### 8.4 Gate

- candidate IDs map to exact locked values;
- unknown IDs and unsupported setting combinations fail before training;
- AbsGrad reaches renderer and strategy together;
- B0 never requests AbsGrad;
- AMP tests prove both projected gradient forms are unscaled once;
- existing renderer, strategy, precision, trainer, and run-training tests pass;
- an L4 CUDA smoke reaches one densification event with finite absolute
  gradients before any 7k candidate run.

## 9. Module 3 — generic runner and decision engine

The only new orchestration entry point is:

```text
src/bts_nvs/experiments/run_experiment.py
```

It invokes `run_training.py` and never embeds a second training loop.

Responsibilities:

- load the accepted backend decision once;
- validate experiment, scene, candidate, manifest, and holdout identity;
- create shallow output directories without overwriting existing runs;
- build fair 7k or 30k commands;
- disable checkpoints at 7k;
- use one 3k-interval rolling recovery file at 30k;
- validate completed artifacts before comparison;
- write a durable failure ledger before exiting;
- emit deterministic scene and cohort decisions;
- support one scene/candidate invocation so expensive stages remain explicit.

The runner has no Phase A/B/C names. Stage names are `reference`, `screen`,
`confirm`, and `production`.

## 10. Stage A — seven-scene B0 references

Create or validate a B0 internal-holdout 7k reference for all seven submission
scenes. An existing artifact is reusable only when all of these match:

- scene ID;
- manifest hash;
- holdout hash and algorithm;
- `max_steps=7000`;
- `seed=0`;
- `resize_factor=1`;
- accepted backend and precision;
- cached images and pinned transfer;
- B0 candidate config hash;
- complete validation renders and reports;
- no model checkpoint in the run directory.

Production 30k artifacts with `internal_holdout=false` are ineligible as
references.

Planning estimate: 2–3 sequential L4 GPU-hours for seven fresh B0 7k runs.
This is an estimate, not a runtime gate.

## 11. Stage B1 — density and thin-detail screen

Scenes:

```text
HCM0539 HCM0421
```

Run both first executable candidates fresh at 7k against their paired B0
references.

A candidate is eligible on a scene only when:

- all artifacts and per-image metrics are complete and finite;
- local composite delta is strictly positive;
- LPIPS does not increase;
- missing-edge and spurious-edge do not both worsen;
- hard-stratum local composite does not decrease;
- peak VRAM is below 23 GB;
- no OOM, non-finite loss, invalid Gaussian state, or uncontrolled primitive
  growth occurs.

When both candidates are eligible, select larger local-composite gain, then
lower LPIPS, then lower symmetric edge distance, then lower peak Gaussian
count. If no candidate is eligible, that scene retains B0.

Seven thousand steps are a mechanism screen, not final evidence. Each scene
winner proceeds to Stage C.

## 12. Stage B2 — input quality, appearance, and bonsai capacity

Scenes:

```text
HCM0644 chair bonsai
```

This stage is split into separately reviewed submodules:

1. image-quality audit with no training behavior change;
2. bounded quality-aware sampling for HCM0644 and chair;
3. regularized per-image affine exposure for chair and bonsai;
4. bonsai capacity candidate changing exactly one of densification horizon or
   pruning before any combination with affine exposure.

Quality-aware sampling must retain a non-zero floor for every pose cell. It may
down-weight poor images but must not delete images or collapse pose coverage.

Affine exposure uses only per-scene train RGB and regularizes its transform
toward identity. It does not use a pretrained feature network or external data.

Each B2 submodule requires its own design amendment or sub-spec that locks the
candidate ID, numeric settings, tests, and gate before it becomes executable.
This is a deliberate authorization boundary, not an unspecified implementation
detail.

## 13. Stage B3 — geometry and sampling-rate robustness

Scenes:

```text
HCM0674 HCM0540
```

B3 opens only if Module 1 reports a lower local composite in the hard pose
stratum than the easy stratum for the paired B0 reference.

Candidate order is fixed:

1. minimal density/capacity change;
2. gsplat antialiased rasterization if the pinned `gsplat==1.4.0` API supports
   it without a dependency change;
3. a separate Mip-Splatting or geometry-architecture design only after the
   first two mechanisms fail.

HCM0540 remains B0 when pose-gap diagnostics do not translate into held-out
metric loss. Geometric distance alone does not authorize a model change.

Like B2, each new B3 candidate requires a focused sub-spec with locked values
before implementation.

## 14. Stage C — per-scene 30k confirmation

Every scene with a 7k winner receives a fresh paired confirmation:

- one B0 30k internal-holdout run;
- one winner 30k internal-holdout run;
- identical manifest, holdout, seed, resolution, backend, precision, and metric
  configuration;
- validation snapshots at steps 15,000 and 30,000;
- one rolling recovery checkpoint, not milestone checkpoints.

The 7k checkpoint is never resumed into a 30k run because the schedule horizon
and config hash differ.

A scene candidate passes only when at step 30,000:

- local composite delta is strictly positive;
- LPIPS does not increase;
- hard-stratum local composite does not decrease;
- missing-edge and spurious-edge do not both worsen;
- all integrity and resource gates pass;
- any gain observed at step 15,000 has not reversed below the gates at 30,000.

Candidate failure affects only that scene; it does not invalidate winners on
other scenes.

## 15. Stage D — cohort lock and production

After all authorized confirmations, write `cohort_decision.json` with exactly
seven entries. Each entry records:

- `scene_id`;
- selected candidate or `B0-reference`;
- source decision hash;
- confirmed config hash;
- manifest and holdout hashes used for selection;
- confirmation metric deltas;
- production authorization boolean.

Production retrains all seven scenes fresh at 30k with
`internal_holdout=false`. Each scene uses only its locked policy. There is no
silent fallback when a non-B0 production run fails; the run stops and records
the failure.

The final render, output validation, JPEG conversion, archive size check, and
submission packaging reuse the closed B0 operational path under a new
submission candidate ID.

Measured B0 production time for the seven scenes is approximately 13.1
sequential L4 GPU-hours. New policies may cost more and must record actual
time, VRAM, and Gaussian trajectories.

## 16. Decision metrics and interpretation

Primary held-out metrics:

- PSNR;
- SSIM;
- LPIPS;
- repository-local composite/Score50.

Secondary explanatory metrics:

- HF-L1;
- missing-edge;
- spurious-edge;
- symmetric edge distance;
- easy/medium/hard pose-stratum aggregates;
- peak/final Gaussian count;
- peak VRAM;
- total training time.

Secondary metrics may reject a visually unsafe candidate but cannot select a
candidate whose primary composite does not improve. No metric computed from
train previews may replace held-out validation.

The official evaluator supplies no per-scene breakdown. This program must not
claim that a scene caused a specific number of official points.

## 17. Storage and recovery policy

- 7k screens save no `.pt` or `.pth` file.
- 30k confirmation and production save only
  `checkpoints/recovery.pt`, overwritten atomically every 3,000 steps and at
  final step.
- Validation renders, reports, config, hashes, metrics, and timing are durable.
- Large exploratory previews are not committed.
- Run artifacts are committed only when they are required evidence and remain
  within repository storage policy; otherwise their hashes and remote storage
  location are recorded.
- A run directory that is non-empty but has neither a valid complete report nor
  a matching recovery checkpoint is rejected rather than overwritten.

## 18. Failure handling

- Unknown candidate, scene, stage, or unsupported setting fails before GPU
  allocation.
- Manifest, holdout, config, backend, or render hash mismatch makes a paired
  comparison ineligible.
- Missing or extra validation renders, wrong image dimensions, NaN/Inf metrics,
  CUDA OOM, non-finite loss, or invalid Gaussian state reject the run.
- A failed candidate is not declared a winner because its competitor also
  failed.
- A failed scene does not authorize changing another scene's policy.
- Production never reads a decision artifact whose schema, hashes, or seven
  scene identities do not match the current cohort authority.

## 19. Test and review strategy

Every implementation module follows test-first development:

1. write a failing unit test for one behavior;
2. verify the expected failure;
3. add the minimal implementation;
4. verify the focused test and affected existing suite;
5. commit only that module's files;
6. request review before opening the next module.

CPU unit tests cover metrics, pose strata, schemas, candidate mapping, command
construction, provenance, decision rules, and failure handling.

GPU verification is limited to:

- existing real gsplat CUDA forward/backward smoke;
- one bounded AbsGrad densification smoke on L4;
- authorized 7k or 30k runs only after the corresponding module gate passes.

No module may rely on a later module to make its tests pass.

## 20. Delivery sequence

Implementation is divided into independently reviewed tasks and commits:

1. Module 1 validation foundation.
2. Candidate registry and contracts.
3. AbsGrad renderer, strategy, precision, and trainer hooks.
4. Generic runner and decision engine.
5. Seven-scene B0 7k references.
6. HCM0539/HCM0421 density screen.
7. Approved B2 quality/appearance submodules.
8. Approved B3 geometry/sampling submodules.
9. Per-scene 30k confirmations.
10. Cohort policy lock and production integration.
11. Fresh seven-scene production, inference, validation, and packaging.

The umbrella spec remains the authority for scope, ordering, invariants, and
gates. Before implementation, each code module receives a detailed plan with
exact files, interfaces, tests, commands, expected failures, and commit
boundary. Later B2/B3 algorithm candidates require focused sub-spec approval
because their numeric policies are intentionally not authorized by this
umbrella design.

## 21. Program acceptance criteria

The program is complete only when:

- every scene has a valid paired decision;
- `cohort_decision.json` contains exactly the seven authorized scene IDs;
- each non-B0 policy passed fresh 30k internal-holdout confirmation;
- scenes without a passing candidate remain B0;
- all production runs reach 30k with valid hashes and recovery artifacts;
- all requested test poses render with exact names and dimensions;
- submission validation and archive size checks pass;
- the final candidate ID, configs, dependencies, logs, hashes, timings, VRAM,
  and decisions are reproducible;
- official results, if later submitted, are reported separately from local
  validation and are not retroactively used to justify candidate selection.

## 22. Out of scope

This umbrella program does not authorize:

- cross-scene training or a shared global model;
- manual output compositing, retouching, deblurring, or per-pose intervention;
- official test RGB access or leakage;
- dependency upgrades, custom CUDA kernels, multi-GPU training, or
  `torch.compile`;
- pretrained depth, segmentation, or perceptual feature models beyond the
  already pinned local LPIPS evaluator;
- changing optimizer learning rates, base loss weights, camera convention,
  scene normalization, SH schedule, initialization, or JPEG packaging without
  a separate approved design;
- blindly averaging renders from multiple models;
- treating 7k screening as production evidence;
- treating high-frequency diagnostics as official score components.
