# Stage B1 — HCM0539/HCM0421 Density and Thin-Detail Screen

**Date:** 2026-07-24

**Status:** Active next-stage plan. This plan authorizes only four fresh 7k
screen runs and their deterministic decisions. It does not authorize 15k/30k
confirmation or production.

**Parent authority:**
`docs/superpowers/specs/2026-07-22-scene-specific-optimization-program-design.md`

## 1. Terminology and current position

`B0-reference` is the baseline candidate policy; it is not an experiment
phase. The executable stages remain `reference`, `screen`, `confirm`, and
`production`.

The project has two already-complete B0 evidence sets:

1. `B0-submission-q99-v1`: seven independent 30k scene models using the same
   B0 configuration, official Score `70.98330`;
2. `runs/scene_opt_v1/reference/<scene>`: seven deterministic 7k
   internal-holdout references for paired screening.

The official result proves the common 30k B0 configuration is a valid
production baseline. It does not contain local held-out RGB and therefore
cannot replace a paired internal-holdout reference.

The current workspace does not retain the seven production 30k model
directories or official per-image/per-scene metrics. Their role and common
configuration are evidenced by the closed baseline record and production
orchestration code; this plan does not reconstruct unavailable 30k
scene-level statistics.

## 2. Harness contract

The published official values satisfy:

```text
0.709833
= 0.4 * (1 - 0.198195)
+ 0.3 * 0.804805
+ 0.3 * (24.611499 / 50)
```

Recomputed score is `0.709832494`; the difference from the published score is
rounding. Solving for the denominator gives `PSNR_max=49.99983`, so this plan
locks `PSNR_max=50`.

The repository harness:

- decodes RGB to `[0,1]`;
- computes full-frame PSNR with `data_range=1`;
- uses Gaussian SSIM 11x11, sigma 1.5, valid padding;
- uses LPIPS 0.1 with the configured backbone, currently `alex`;
- computes image metrics, averages images within each scene, then averages
  scenes with equal scene weight.

Only the composite formula and PSNR denominator are established from official
outputs. Official LPIPS/SSIM implementation and averaging details remain
unknown. Stage B1 therefore selects by paired local deltas, never by estimating
official per-scene points.

## 3. Evidence for the selected scenes

| Scene | B0 Score50 | Hard Score50 | Easy Score50 | Hard gap | Missing edge | Worst image Score50 |
|---|---:|---:|---:|---:|---:|---:|
| HCM0539 | 68.897 | 62.148 | 74.221 | -12.072 | 0.2797 | 37.29 |
| HCM0421 | 68.404 | 58.204 | 75.611 | -17.407 | 0.2912 | 32.77 |

Visual audit shows blurred BTS/antenna structure and severe hard-pose sky
failures. The density candidates test the thin-detail mechanism only. They are
not expected to solve an unbounded or black-background sky failure; sky
alpha/transmittance must be audited under a separate later sub-spec.

## 4. Locked experiment matrix

Run exactly:

| Order | Scene | Candidate | Primary change |
|---:|---|---|---|
| 1 | HCM0539 | `E1-density-absgrad-t04-v1` | `absgrad=true`, `grow_grad2d=0.0004` |
| 2 | HCM0539 | `E1-density-scale005-v1` | `grow_scale3d=0.005` |
| 3 | HCM0421 | `E1-density-absgrad-t04-v1` | `absgrad=true`, `grow_grad2d=0.0004` |
| 4 | HCM0421 | `E1-density-scale005-v1` | `grow_scale3d=0.005` |

All other settings remain paired with the scene B0 reference: manifest,
holdout, seed 0, full resolution, backend, precision, loss, sampling, renderer,
7k horizon and metric configuration.

Do not add revised opacity, exposure, sky modeling, quality-aware sampling, a
second seed, or a third candidate to this screen.

## 5. Minimal implementation task — complete

The training path already existed. Deterministic screen selection is exposed
through the same generic CLI without adding a Bash or phase-specific runner:

1. add a `decide-screen` subcommand to
   `bts_nvs.experiments.run_experiment`;
2. load one B0 report and the two candidate reports;
3. call the existing `select_scene_candidate`;
4. save with the existing `save_scene_decision`;
5. add focused CLI tests; do not change decision mathematics.

Success criterion: the same three input reports always produce the same
hash-bearing scene decision, and malformed/mismatched reports fail before
writing output. Focused and affected experiment tests passed `160/160` on
2026-07-24.

## 6. Preflight

Before allocating GPU:

1. commit the documentation/CLI state used by the run;
2. verify the accepted backend report exists;
3. validate the two existing B0 references with the generic `validate` command;
4. run focused unit tests for candidates, runner, artifacts and decisions;
5. run the pretrained LPIPS smoke on the VM;
6. require each target output directory to be absent or empty.

No screen run may overwrite an existing non-empty directory.

## 7. Execution commands

Set shared paths once:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export EXP_ROOT="$PWD/runs/scene_opt_v1"
export SCENES_ROOT="$PWD/data/bts_scenes"
export MANIFESTS_ROOT="$PWD/runs/manifests"
export BACKEND_ROOT="$PWD/runs/phase4/backend_qualification"
```

Define one reusable shell function in the VM session:

```bash
run_screen() {
  scene_id="$1"
  candidate_id="$2"
  python -m bts_nvs.experiments.run_experiment run \
    --repo-root "$PWD" \
    --scenes-root "$SCENES_ROOT" \
    --manifests-root "$MANIFESTS_ROOT" \
    --backend-root "$BACKEND_ROOT" \
    --experiment-root "$EXP_ROOT" \
    --stage screen \
    --scene-id "$scene_id" \
    --candidate-id "$candidate_id" \
    --stop-step 7000 \
    --b0-report "$EXP_ROOT/reference/$scene_id/experiment_report.json"
}
```

Execute in locked order:

```bash
run_screen HCM0539 E1-density-absgrad-t04-v1
run_screen HCM0539 E1-density-scale005-v1

python -m bts_nvs.experiments.run_experiment decide-screen \
  --b0-report "$EXP_ROOT/reference/HCM0539/experiment_report.json" \
  --candidate-report "$EXP_ROOT/screen/HCM0539/E1-density-absgrad-t04-v1/experiment_report.json" \
  --candidate-report "$EXP_ROOT/screen/HCM0539/E1-density-scale005-v1/experiment_report.json" \
  --output "$EXP_ROOT/decisions/screen/HCM0539.json"

run_screen HCM0421 E1-density-absgrad-t04-v1
run_screen HCM0421 E1-density-scale005-v1

python -m bts_nvs.experiments.run_experiment decide-screen \
  --b0-report "$EXP_ROOT/reference/HCM0421/experiment_report.json" \
  --candidate-report "$EXP_ROOT/screen/HCM0421/E1-density-absgrad-t04-v1/experiment_report.json" \
  --candidate-report "$EXP_ROOT/screen/HCM0421/E1-density-scale005-v1/experiment_report.json" \
  --output "$EXP_ROOT/decisions/screen/HCM0421.json"
```

Expected output:

```text
runs/scene_opt_v1/screen/<scene_id>/<candidate_id>/
```

Run sequentially on one L4. Do not run two scene processes concurrently.

## 8. Acceptance and selection

For candidate `c` against paired B0:

```text
delta_score50(c) = Score50(c) - Score50(B0)
delta_lpips(c) = LPIPS(c) - LPIPS(B0)
delta_hard(c) = Score50_hard(c) - Score50_hard(B0)
rho_time(c) = time(c) / time(B0)
```

A candidate screen passes only when:

- `delta_score50 > 0`;
- `delta_lpips <= 0`;
- `delta_hard >= 0`;
- missing-edge and spurious-edge do not both increase;
- `rho_time <= 1.25`;
- peak VRAM is below 23 GB;
- artifact integrity and primitive-growth gates pass.

If both pass, select larger Score50 gain, then lower LPIPS, lower symmetric edge
distance, lower peak Gaussian count and finally candidate ID. If neither
passes, select `B0-reference`.

Write one decision per scene under:

```text
runs/scene_opt_v1/decisions/screen/<scene_id>.json
```

## 9. Compute and storage budget

Measured B0 times were 16.1 minutes for HCM0539 and 15.6 minutes for HCM0421.
Four B0-equivalent runs therefore estimate 63.4 minutes. The locked 1.25x
paired-time gate gives an approximate upper training budget of 79.3 minutes,
excluding environment setup and report generation.

Screen runs save no model checkpoint. Preserve reports, hashes, timing,
metrics, previews and validation renders. Stop immediately for OOM, NaN/Inf,
invalid Gaussian state, missing render, hash mismatch or uncontrolled
primitive growth.

## 10. Stop/go boundary

After both decisions:

- a scene with a non-B0 screen winner becomes eligible for a separate Stage C
  30k confirmation plan;
- a scene that falls back to B0 does not receive another 30k B0 run;
- do not infer a seven-scene policy from these two scenes;
- do not submit official test outputs as part of this stage.

The next review reports per candidate and scene: all primary deltas, hard-pose
deltas, edge metrics, time ratio, VRAM, Gaussian counts, worst validation
images and the deterministic selected candidate.
