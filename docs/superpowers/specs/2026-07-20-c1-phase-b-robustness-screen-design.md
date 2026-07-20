# C1 Phase B Six-Scene Robustness Screen Design

**Date:** 2026-07-20
**Status:** Approved for implementation
**Parent experiment:** `2026-07-19-c1-absgrad-revised-opacity-30k-design.md`
**Branch policy:** Continue on `ex1/absgrad-revopacity-phase-a`; Phase B does not create a new branch.

## 1. Decision and objective

Phase A passed and permanently locks the winner to:

```text
C1-absgrad-t08-revopacity-v1
```

Phase B tests whether its positive 7,000-step effect survives across the complete
pre-registered six-scene calibration cohort. It adds only the four missing runs:

```text
hcm0031 HCM0181 HNI0131 HNI0265
```

The Phase-A runs on `HCM0421` and `HCM1439` are reused. Existing paired
`B0-reference` reports and validation renders are reused; no B0 scene is
retrained.

Phase B does not implement or launch the 30,000-step confirmation. A passing
six-scene decision authorizes the separate Phase C implementation.

## 2. Why these scenes are fixed

The six-scene cohort was used by the historical baseline qualification before
C1 results existed:

```text
hcm0031 HCM0181 HCM0421 HCM1439 HNI0131 HNI0265
```

Using its remaining four scenes avoids post-result scene selection. The cohort
contains 130 held-out images and spans baseline `Score50` from 61.979 to 70.567,
peak Gaussian counts from 2.985M to 4.148M, and both HCM and HNI site groups.
`HCM0181` is also the bridge to the existing 30,000-step B0 holdout used by
Phase C.

These are calibration scenes, not the seven production scenes. Phase B does not
claim cross-scene generalization or estimate the official score.

## 3. Locked run contract

Each new run must use:

```text
candidate_id       C1-absgrad-t08-revopacity-v1
max_steps          7000
resize_factor      1
seed               0
internal_holdout   true
cache_images       true
pinned_transfer    true
absgrad             true in renderer and density strategy
revised_opacity    true
grow_grad2d        0.0008
rolling_checkpoint false
```

All other model, optimizer, loss, camera, renderer, and densification settings
remain identical to Phase A. Backend and precision are loaded from the same
accepted backend qualification artifact. Existing non-empty incomplete run
directories are rejected. Complete runs may be reused only after their config,
manifest, holdout, backend, metrics, timing, renders, and report are validated.

Runs execute sequentially in this order:

```text
hcm0031 HCM0181 HNI0131 HNI0265
```

No model checkpoint is written. Durable artifacts are per-step metrics and
timing, final validation renders, qualification report, high-frequency
diagnostics, convergence, environment, config, and hashes.

## 4. Reuse and naming policy

Phase B must not add a shell script. The existing thin C1 Python entry point is
renamed from:

```text
src/bts_nvs/training/run_c1_phase_a.py
```

to:

```text
src/bts_nvs/training/run_c1_screening.py
```

It accepts `--stage phase-a|phase-b` and dispatches to focused stage modules.
`c1_phase_a_runner.py` remains focused on Phase A. Phase B orchestration lives
in `c1_phase_b_runner.py`; six-scene calculations live in
`c1_phase_b.py`. The common CLI contains argument parsing and dispatch only.

The repository's durable shell entry points are renamed by meaning, using
`git mv` so script count does not increase:

| Old name | New name | Meaning |
|---|---|---|
| `prepare_phase4_artifacts.sh` | `prepare_scene_manifests.sh` | Build and validate canonical scene manifests and holdouts |
| `run_phase4_qualification.sh` | `run_baseline_screening.sh` | Run the locked B0 qualification matrix |
| `run_phase4_30k_dry_run.sh` | `run_full_length_qualification.sh` | Run the controlled 30k internal-holdout qualification |
| `run_phase4_backend_qualification.sh` | `qualify_training_backend.sh` | Measure and select the accepted L4 backend/precision |
| `run_phase4_full_training.sh` | `train_scene_cohort.sh` | Train a selected scene cohort through the production runner |
| `run_phase4_inference.sh` | `render_scene_cohort.sh` | Render a trained scene cohort and emit an inference report |

No compatibility wrapper is added. Active script call sites, operational docs,
and tests are updated atomically. Historical result paths such as
`runs/phase4/...` are not renamed because they identify closed artifacts, not
entry-point behavior. `run_submission_auxiliary_training.sh` and JPEG submission
scripts already describe their purpose and remain unchanged.

## 5. Input authority and restoration

Phase B reads:

- Phase-A decision and candidate artifacts from `runs/c1/phase_a`;
- four new scene manifests and holdouts from `runs/manifests`;
- six B0 reports and validation-render directories from
  `runs/phase4/qualification`;
- accepted backend decision from `runs/phase4/backend_qualification`.

The runner never performs a Git restore. If historical B0 artifacts are absent,
preflight reports every missing path and exits before GPU work. Restoration is
an explicit operator action limited to the required report and render paths.

Phase B rejects an input unless `phase_a_passed` is true and
`selected_candidate` exactly matches the locked winner. Scene IDs are
case-sensitive; `hcm0031` must remain lowercase.

## 6. Output contract

New artifacts are written under:

```text
runs/c1/phase_b/
├── hcm0031/C1-absgrad-t08-revopacity-v1/
├── HCM0181/C1-absgrad-t08-revopacity-v1/
├── HNI0131/C1-absgrad-t08-revopacity-v1/
├── HNI0265/C1-absgrad-t08-revopacity-v1/
├── baseline_diagnostics/
└── phase_b_decision.json
```

`phase_b_decision.json` references, but does not copy, the two Phase-A candidate
reports. It records all six paired scene deltas, aggregate metrics, exact sign
test, edge diagnostics, resource statistics, each gate boolean, and an explicit
pass or rejection reason.

## 7. Six-scene decision gate

For scene `s`:

```text
Score50(s) = 40 - 40*LPIPS(s) + 30*SSIM(s) + 0.6*PSNR_dB(s)
delta(s)   = candidate Score50(s) - B0 Score50(s)
```

The candidate passes only when all conditions hold:

1. arithmetic mean of the six scene-level deltas is greater than zero;
2. at least four of six scene deltas are greater than zero;
3. unweighted mean scene LPIPS does not increase relative to B0;
4. aggregate missing-edge and spurious-edge do not both worsen;
5. all run-integrity gates pass;
6. peak VRAM for every run is below 23 GB.

The aggregation unit is a scene, matching the final metric's per-scene mean.
Holdout image counts do not weight scenes. The decision also reports median
delta and per-image distributions for diagnosis, but they do not alter the
pre-registered gate.

The exact two-sided sign-test p-value is descriptive. Four, five, and six
positive scenes correspond respectively to weak, moderate, and strongest
available consistency evidence; only six of six yields `p=0.03125`. A four-of-
six pass requires explicit analysis of both negative scenes before Phase C, but
does not change the locked gate after observing results.

## 8. Integrity and failure handling

Every new run requires:

- exactly 7,000 ordered finite metric records for steps 1 through 7,000;
- exactly 7,000 finite timing records;
- finite PSNR, SSIM, LPIPS, Score50, HF-L1, missing-edge, and spurious-edge;
- complete validation-render coverage and matching image count;
- matching config, manifest, and holdout hashes;
- non-blank convergence output;
- positive peak/final Gaussian counts;
- recorded environment, wall time, and peak VRAM;
- no CUDA OOM, non-finite loss, or invalid Gaussian state.

A failed process stops the sequence. A completed scene remains reusable after
full validation. The runner does not delete, overwrite, or silently repair
partial output. No decision is emitted from fewer than six paired scenes.

## 9. Testing strategy

Implementation proceeds in independently testable blocks:

1. rename shell entry points and update all active call sites/tests without
   changing behavior;
2. add Phase-B decision fixtures and tests before implementation;
3. implement six-scene decision calculation and deterministic JSON output;
4. add runner command/preflight/reuse/failure tests before implementation;
5. implement the four-scene sequential runner;
6. rename the thin C1 CLI to `run_c1_screening.py` and test both dispatch paths;
7. run focused unit tests, the full CPU suite, then inspect generated commands;
8. on the VM, validate existing backend and inputs before launching GPU work.

Tests must cover Phase-A rejection, wrong winner, missing B0 artifacts, exact
scene casing, wrong holdout, incomplete metrics/timing/renders, NaN/Inf, resource
failure, three-of-six rejection, four-of-six conditional pass, six-of-six pass,
and safe reuse of a complete run.

The existing Phase-A tests remain green after the common CLI rename. B0 defaults
remain `absgrad=false` and `revised_opacity=false`.

## 10. Compute and promotion

The four historical B0 runs took 3,847.48 seconds in total. Applying the two
measured Phase-A candidate/B0 time ratios gives a planning range of 44.7 to 53.9
minutes of sequential L4 training, excluding evaluation overhead.

Phase B passing does not launch Phase C automatically. It emits the decision and
the exact next-stage candidate identity. The 30,000-step mode, rolling recovery
checkpoint, and HCM0181 comparison are implemented only after the Phase-B result
is reviewed and approved.

## 11. Out of scope

Phase B does not change the candidate hyperparameters, loss, optimizer,
densification schedule, sky representation, camera model, codec, production
scene selection, 30,000-step training contract, or closed B0 artifacts. It does
not add manual BTS/sky masks, per-image tuning, new dependencies, or another
shell runner.
