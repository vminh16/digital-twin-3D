# Project documentation status

## Current position — 2026-07-26

The closed submission baseline remains:

```text
B0-submission-q99-v1
Score 70.98330
HCM0644 HCM0674 HCM0540 HCM0539 HCM0421 chair bonsai
```

The project has moved from the old global Phase 3/4 and C1 workflows to the
scene-specific seven-scene optimization program.

Completed:

1. baseline training, inference, submission validation and official evaluation;
2. the historical C1 AbsGrad × revised-opacity experiment;
3. Module 1: held-out validation, detail metrics and pose strata;
4. Module 2: generic candidate registry and AbsGrad-capable hooks;
5. Module 3: stage-first runner, artifact validation, recovery contract and
   deterministic decisions;
6. local unit gates and user-reported NVIDIA L4 smoke gates;
7. Stage A: seven valid `B0-reference` internal-holdout runs at 7,000 steps.
8. Stage B1: AbsGrad t04 screening on HCM0539 and HCM0421.
9. Deadline compute-first production: retained 30k AbsGrad checkpoints for
   HCM0539 and HCM0421.

Current execution boundary:

- Stage A passed with exactly seven reference directories, 174/174 validation
  renders, no missing report, no non-finite metric and no model checkpoint;
- scene-balanced Stage A means are Score50 `64.654`, PSNR `21.050`, SSIM
  `0.7267` and LPIPS `0.2444`;
- total measured Stage A runtime was about 95.6 minutes on NVIDIA L4, with
  peak VRAM between 1.35 GB and 6.16 GB;
- the existing 30k common-config production baseline remains the closed
  `B0-submission-q99-v1` official result at Score `70.98330`;
- Stage C paired 30k confirmation was not executed and must not be inferred
  from the full-data production checkpoints;
- the active scope is hybrid inference: B0 for five scenes and AbsGrad for
  HCM0539/HCM0421, followed by exact output validation and packaging.

## Canonical documents

Read these in order:

1. [Scene-specific optimization program](superpowers/specs/2026-07-22-scene-specific-optimization-program-design.md) —
   umbrella scientific and engineering contract.
2. [Generic experiment runner](superpowers/specs/2026-07-22-generic-experiment-runner-design.md) —
   executable stage, artifact, recovery and decision contract.
3. [Stage B1 density screen](superpowers/plans/2026-07-24-stage-b1-density-screen.md) —
   completed screening evidence and compute-first exception context.
4. [Completed Stage A execution plan](superpowers/plans/2026-07-23-stage-a-seven-scene-b0-references.md) —
   historical execution evidence.
5. [Repository README](../README.md) — installation, closed baseline and
   legacy baseline operations.
6. [Repository rules](../AGENTS.md) — data, output, metric and reproducibility
   constraints.

If these documents conflict, `AGENTS.md` governs repository constraints and
the scene-specific program governs experiment policy.

## Why older documents were removed

The deleted Markdown files remain recoverable through Git history. They were
removed from the active tree because leaving multiple executable-looking plans
caused ambiguity about the authorized experiment.

| Removed group | Why it is no longer active |
|---|---|
| Phase 3 specifications, smoke runbook and implementation plans | Training-engine construction and the HCM0181 smoke gate are complete. Their commands include obsolete numbered checkpoints and pre-baseline milestones. |
| Phase 4.1–4.8 plans and designs | They built the closed B0 pipeline. Some target 13/18-scene cohorts, old `runs/phase4` layouts, legacy Bash runners or assumptions that are not the current seven-scene optimization authority. |
| C1 AbsGrad × revised-opacity design | C1 was executed and superseded. Revised opacity is deliberately absent from the current candidate registry; reusing this plan would mix retired candidate IDs and Phase A/B/C semantics with the generic runner. |
| JPEG submission converter plan/design | The q99 submission is closed and its essential codec contract is now in `AGENTS.md` and `README.md`. It is not an optimization-stage authority. |
| Module 1–3 implementation plans | Their code and tests are complete. Durable outcomes are summarized here; detailed task-by-task evidence is available from commits and Git history. |

Removal does not remove code, experiment outputs, research reports or Git
evidence. `docs/research/` is user-owned and was not modified by this cleanup.

## Next action

Canonicalize the interrupted HCM0539 metric trace through its final recovery
checkpoint, then run one atomic five-BTS inference with per-scene run-directory
overrides for HCM0539 and HCM0421. Render chair and bonsai through the unchanged
auxiliary baseline path, validate all exact filenames/formats/dimensions, and
package the hybrid candidate separately from `B0-submission-q99-v1`.

```bash
test ! -d runs/scene_opt_v1/production_mvp/scenes/HCM0539/validation_renders \
  || rmdir -- runs/scene_opt_v1/production_mvp/scenes/HCM0539/validation_renders
bash scripts/run_absgrad_mvp_production.sh HCM0539

BTS_OUTPUT_ROOT="$PWD/outputs_bts_absgrad_mvp_q99" \
BTS_INFERENCE_REPORT="$PWD/runs/scene_opt_v1/mvp_inference_report.json" \
bash scripts/run_phase4_inference.sh \
  --skip_prepare \
  --jpeg_quality 99 \
  --scene_ids HCM0644 HCM0674 HCM0540 HCM0539 HCM0421 \
  --run_dir "HCM0421=$PWD/runs/scene_opt_v1/production_mvp/scenes/HCM0421" \
  --run_dir "HCM0539=$PWD/runs/scene_opt_v1/production_mvp/scenes/HCM0539"
```

`rmdir` intentionally succeeds only when the stale holdout directory is empty;
it does not recursively delete artifacts.
