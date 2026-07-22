# Generic Experiment Runner and Decision Engine Design

**Date:** 2026-07-22

**Status:** Approved design for Module 3; implementation remains staged and
test-gated.

**Parent specification:**
`docs/superpowers/specs/2026-07-22-scene-specific-optimization-program-design.md`

## 1. Purpose

Module 3 is an experiment-integrity layer, not a second trainer. It executes
one scene, one candidate, and one stage per invocation through the existing
`run_training.py`, then validates artifacts and makes deterministic paired
decisions.

It solves four operational risks:

1. unfair B0/candidate comparisons;
2. accidental overwrite between 7k and 30k evidence;
3. excessive checkpoint storage;
4. expensive 30k runs for candidates that already fail at 7k.

## 2. Why 30k, and what it does not prove

Thirty thousand steps are not a mathematical convergence guarantee and are
not claimed to be the globally optimal stopping time. They are the locked
finite-budget comparison horizon for three evidence-backed reasons:

1. The official 3D Gaussian Splatting implementation defaults to 30,000
   iterations, schedules position learning rate over 30,000 steps, and stops
   densification at 15,000 steps:
   https://github.com/graphdeco-inria/gaussian-splatting/blob/main/arguments/__init__.py
2. AbsGS evaluates its mechanism by stopping densification at 15k and training
   at 30k on a single V100:
   https://arxiv.org/abs/2404.10484
3. Repository evidence for HCM0181 shows that extending B0 from 7k to 30k
   improved PSNR `21.733 -> 22.692`, SSIM `0.768 -> 0.805`, LPIPS
   `0.160 -> 0.111`, and local Score50 by `+3.646`. The 30k run took
   `2.8965 h` and peaked at `11.79 GB`. This is one-scene/one-seed evidence,
   so it justifies confirmation at 30k but cannot prove that 30k is optimal or
   that 40k would be better.

The densification horizon separates two regimes:

```text
steps 0..15k: optimize parameters and change representation capacity
steps 15k..30k: optimize the fixed Gaussian population
```

For AbsGrad, the mechanism under test primarily acts in the first regime. The
second regime is still necessary to measure whether the altered Gaussian
population can be optimized into a better held-out solution under the same
budget as B0.

## 3. Compute policy

The runner uses successive filtering:

| Stage | Horizon | Checkpoint policy | Purpose |
|---|---:|---|---|
| `reference` | 7k | none | Paired B0 authority for cheap screening |
| `screen` | 7k | none | Reject weak/unsafe mechanisms cheaply |
| `confirm` | 30k | one rolling recovery file | Fair final validation against fresh B0 |
| `production` | 30k | one rolling recovery file | Train the locked seven-scene cohort |

Confirmation records validation snapshots at 15k and 30k. A 15k snapshot may
reject a run for integrity, OOM, non-finite state, or resource failure, but it
cannot accept a candidate. Quality acceptance is based on paired 30k results.
This avoids a biased rule where a candidate stops at its most favorable
intermediate point while B0 is judged at another horizon.

For scene `s`, define the paired time ratio:

```text
rho_time(s) = T_candidate,30k(s) / T_B0,30k(s)
```

Every report must contain wall time, median observed step time, peak VRAM,
peak/final Gaussian count, and `rho_time` when a pair exists. The pre-registered
engineering budget is `rho_time <= 1.25` and peak VRAM `< 23 GB`. The 1.25
limit is a project cost constraint, not a scientific constant.

Absolute runtime is estimated, not used as a quality gate because scene image
count, resolution, and Gaussian population differ. Current planning bounds are:

- seven fresh B0 7k references: approximately 2--3 sequential L4 GPU-hours;
- B1 screen, two scenes by two candidates: approximately 1.2 GPU-hours using
  the HCM0181 7k proxy, before operational overhead;
- one paired 30k confirmation: approximately 3.7--5.8 GPU-hours, using the
  measured seven-scene B0 mean at the lower end and HCM0181 at the upper end;
- seven-scene production: measured B0 approximately 13.1 sequential L4
  GPU-hours; an accepted policy at the 1.25 time ceiling is approximately
  16.4 hours.

These bounds must be replaced by measured per-scene values as runs complete.

## 4. Output and invocation contract

```text
runs/scene_opt_v1/
├── experiment.json
├── reference/<scene_id>/
├── screen/<scene_id>/<candidate_id>/
├── confirm/<scene_id>/<candidate_id>/
├── production/<scene_id>/<candidate_id>/
├── decisions/<scene_id>.json
└── cohort_decision.json
```

The CLI accepts exactly one `stage`, `scene_id`, and `candidate_id`. It fails
before GPU allocation for an unknown identity, illegal stage/candidate pair,
non-empty destination, mismatched manifest/holdout/backend hash, or an
unsupported setting.

Stage authority is fixed:

- `reference` accepts only `B0-reference` and 7k;
- `screen` accepts an executable non-B0 candidate and 7k;
- `confirm` accepts `B0-reference` or a recorded scene winner and 30k;
- `production` accepts only the policy authorized by `cohort_decision.json`.

## 5. Module boundaries

```text
src/bts_nvs/experiments/experiment.py
src/bts_nvs/experiments/commands.py
src/bts_nvs/experiments/artifacts.py
src/bts_nvs/experiments/decisions.py
src/bts_nvs/experiments/run_experiment.py
```

- `experiment.py` owns schema validation, seven-scene authority, stage
  horizons, paths, and resource budgets.
- `commands.py` builds one argument vector for the existing training entry
  point; it never launches a shell or duplicates training logic.
- `artifacts.py` validates reports, renders, provenance, recovery policy, and
  resource measurements, and writes a durable failure ledger.
- `decisions.py` compares paired reports using pre-registered gates and
  deterministic tie-breaks.
- `run_experiment.py` is a thin Python CLI that composes these modules and
  invokes `run_training.py` once.

`run_training.py` receives only minimal generic additions: candidate/stage
identity, internal-holdout report generation for generic experiments, no-save
7k policy, and a stop-step control distinct from the locked 30k schedule
horizon so 15k validation does not change the optimizer schedule or config
identity.

## 6. Confirmation resume and snapshot semantics

A confirmation config always pins `max_steps=30000`. Invocation one may stop at
15,000 through `stop_step=15000`; invocation two resumes the same config and
stops at 30,000. `stop_step` is execution control, not a model hyperparameter,
and therefore does not alter the candidate config hash.

At 15k, reports are preserved under a snapshot subdirectory while the only
model state is `checkpoints/recovery.pt`. At 30k, final reports are written at
the run root. The rolling recovery file is atomically overwritten every 3k
steps and at the current stop step. No milestone model checkpoint is retained.

## 7. Decision contract

A candidate is eligible only if all paired identities match and:

```text
delta_local_composite > 0
delta_LPIPS <= 0
delta_hard_stratum_local_composite >= 0
not (missing_edge worsens and spurious_edge worsens)
rho_time <= 1.25
peak_vram_mb < 23 * 1024
```

Any missing render, non-finite metric/loss, invalid Gaussian state, OOM,
uncontrolled primitive growth, hash mismatch, or recovery mismatch rejects the
run. Tie-break order is larger local-composite gain, lower LPIPS, lower
symmetric edge distance, then lower peak Gaussian count.

The local composite remains diagnostic because grader details are not fully
known. No official test RGB or official per-scene score is used for selection.

## 8. Acceptance criteria

Module 3 passes only when tests prove:

- stage-first paths preserve 7k and 30k evidence independently;
- illegal stage/candidate/horizon combinations fail before process launch;
- B0 and candidate commands pin identical non-candidate settings;
- 7k commands cannot save checkpoints;
- 30k commands use one rolling recovery file and preserve 15k/30k reports;
- resume retains the 30k schedule/config identity;
- artifact validation rejects incomplete, stale, non-finite, or mismatched
  evidence;
- decisions implement every metric and resource gate exactly;
- failure ledger writes atomically before non-zero exit;
- existing Module 1, Module 2, trainer, and evaluator tests remain green.

Passing Module 3 authorizes Stage A reference execution. It does not itself
authorize Stage B1, confirmation, or production runs.
