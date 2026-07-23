# Stage A — Seven-scene B0 7k Reference Execution Plan

**Status:** Authorized after the user-reported L4 Module 3 gate on 2026-07-23.

**Design authority:**
`docs/superpowers/specs/2026-07-22-scene-specific-optimization-program-design.md`
and
`docs/superpowers/specs/2026-07-22-generic-experiment-runner-design.md`.

## Goal

Create or validate one fresh, leakage-safe, internal-holdout `B0-reference`
artifact at 7,000 steps for every submission scene:

```text
HCM0644 HCM0674 HCM0540 HCM0539 HCM0421 chair bonsai
```

These artifacts are paired comparison authorities for later scene-specific
screens. Stage A does not run either non-B0 candidate and does not authorize
confirmation or production.

## Locked execution contract

- candidate: `B0-reference`;
- stage: `reference`;
- step horizon and stop step: 7,000;
- seed: 0;
- resize factor: 1;
- deterministic internal holdout from each scene manifest;
- backend and precision from the accepted backend qualification;
- image cache and pinned transfer enabled;
- no resume and no `.pt`/`.pth` checkpoint;
- output root: `runs/scene_opt_v1/reference/<scene_id>`;
- complete finite metrics, validation renders, detail metrics, pose strata,
  experiment report, config and provenance hashes.

The generic runner enforces this contract and validates artifacts before a run
is considered complete.

Scene storage is split into two existing data profiles:

| Profile | Scenes root | Manifests root | Scenes |
|---|---|---|---|
| BTS | `data/bts_scenes` | `runs/manifests` | five HCM scenes |
| Auxiliary | `data/auxiliary` | `runs/manifests_auxiliary` | `chair`, `bonsai` |

The experiment output root remains shared at `runs/scene_opt_v1`. A runner must
select the correct input profile per scene; it must not copy auxiliary scenes
into the canonical BTS pool.

## Runtime reuse decision

`scripts/run_phase4_qualification.sh` is historical baseline machinery, not a
Stage A runner. Although it includes `B0-reference` at 7k, it is locked to six
old calibration scenes, also runs `B0-compact`, invokes the legacy
`--qualification_candidate` mode, writes `runs/phase4/qualification`, and emits
the old aggregate decision. Executing it cannot create Stage A authorities.

Stage A must reuse `bts_nvs.experiments.run_experiment.run_one` and the existing
training entry point. Reusable orchestration may record a short deployment
manifest or command, choose the BTS/auxiliary input profile, and invoke that
generic runner sequentially. It must not copy training, validation, resume or
decision logic into Bash.

## Execution order

Run scenes sequentially on the single L4:

```text
HCM0539
HCM0421
HCM0644
chair
bonsai
HCM0674
HCM0540
```

`HCM0539` and `HCM0421` run first because they are the pre-registered Stage B1
density/thin-detail screening scenes. This ordering yields the next useful
paired evidence earliest without changing any training mathematics. Remaining
scenes are grouped by the later B2 and B3 analysis stages.

## Task 1 — Read-only preflight

From a clean tracked checkout of `main`, record:

```bash
git rev-parse HEAD
git status --short
nvidia-smi
df -h .
```

Require:

- expected Module 3 commits are present;
- no tracked source change;
- L4 is visible and has no conflicting process;
- sufficient disk remains for seven metric/render directories;
- all seven scene directories and manifest/holdout artifacts exist;
- backend qualification is accepted.

Do not delete or overwrite an existing `runs/scene_opt_v1/reference/<scene>`.
An existing artifact must be validated against the locked contract before
reuse; otherwise move it outside the experiment root or choose a new root after
explicit review.

## Task 2 — First reference canary

Run only `HCM0539` through the checked-in deployment record:

```bash
bash scripts/run_scene_opt_references.sh HCM0539
```

The canary passes only if the command exits zero and post-run artifact
validation succeeds. Verify:

```bash
find runs/scene_opt_v1/reference/HCM0539 -type f \
  \( -name '*.pt' -o -name '*.pth' \)
```

Expected output is empty. Preserve metrics and reports; do not create an
additional model checkpoint.

Every expanded Python command and the current Git commit are appended to:

```text
runs/scene_opt_v1/deployment_commands.log
```

Rerunning the command does not retrain a valid reference. It invokes the
generic `validate` operation and skips only after the complete artifact
contract passes. A partial, stale or mismatched directory fails and stops the
script.

## Task 3 — Second B1 reference

After reviewing HCM0539 runtime, peak VRAM, Gaussian growth, finite loss and
validation completeness, execute:

```bash
bash scripts/run_scene_opt_references.sh HCM0421
```

Stop Stage A immediately for OOM, non-finite metric/loss, artifact validation
failure, unexpected checkpoint, uncontrolled Gaussian growth, or disk pressure.

## Task 4 — Remaining five references

Run the same generic reference operation sequentially for:

```text
HCM0644 chair bonsai HCM0674 HCM0540
```

For `chair` and `bonsai`, replace only:

```text
--scenes-root "$PWD/data/auxiliary"
--manifests-root "$PWD/runs/manifests_auxiliary"
```

All HCM scenes continue to use `data/bts_scenes` and `runs/manifests`.

After the two canaries pass, run all seven in the locked order:

```bash
bash scripts/run_scene_opt_references.sh
```

The first two are validated and skipped; the remaining five run sequentially.
To override runtime roots without editing the script:

```bash
PYTHON_BIN="$PWD/.venv/bin/python" \
BTS_EXPERIMENT_ROOT="$PWD/runs/scene_opt_v1" \
BTS_BACKEND_ROOT="$PWD/runs/phase4/backend_qualification" \
bash scripts/run_scene_opt_references.sh
```

Do not use parallel scene training on one L4. Parallelism would change memory
headroom and invalidate resource comparisons without reducing total GPU work
reliably.

## Task 5 — Stage A audit

For every scene, collect:

- total and median observed step time;
- peak VRAM;
- peak and final Gaussian count;
- overall and hard-stratum local composite;
- PSNR, SSIM and LPIPS;
- missing, spurious and symmetric edge metrics;
- validation render count;
- manifest, holdout and config hashes;
- proof that no model checkpoint exists.

Require exactly seven valid reference directories and no missing scene. Compare
resource ranges to identify outliers, but do not reject a reference merely
because one scene is slower than another; the paired time gate applies between
a candidate and B0 on the same scene.

## Stop/go decision

Stage A passes only when all seven references satisfy the locked contract.
After Stage A:

1. report and review all seven scene references;
2. inspect HCM0539 and HCM0421 image/detail evidence;
3. explicitly authorize Stage B1;
4. only then run `E1-density-absgrad-t04-v1` and
   `E1-density-scale005-v1` at 7k on those two scenes.

No 15k, 30k, candidate 7k, or production run is authorized by this plan.

## Compute expectation

Seven fresh sequential references are estimated at 2–3 L4 GPU-hours. This is a
planning estimate, not a pass/fail gate. The first two scenes provide measured
runtime bounds before committing compute to the remaining five.
