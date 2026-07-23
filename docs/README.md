# Project documentation status

## Current position — 2026-07-23

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
6. local unit gates and user-reported NVIDIA L4 smoke gates.

Current execution boundary:

- Stage A is authorized but has not been executed in this workspace;
- `runs/scene_opt_v1` is absent locally as of this audit;
- the next permitted GPU work is seven fresh or contract-valid
  `B0-reference` runs at 7,000 steps;
- Stage B1 candidate screening, 15k/30k confirmation and production remain
  unauthorized until Stage A evidence is reviewed.

## Canonical documents

Read these in order:

1. [Scene-specific optimization program](superpowers/specs/2026-07-22-scene-specific-optimization-program-design.md) —
   umbrella scientific and engineering contract.
2. [Generic experiment runner](superpowers/specs/2026-07-22-generic-experiment-runner-design.md) —
   executable stage, artifact, recovery and decision contract.
3. [Stage A execution plan](superpowers/plans/2026-07-23-stage-a-seven-scene-b0-references.md) —
   the only active execution plan.
4. [Repository README](../README.md) — installation, closed baseline and
   legacy baseline operations.
5. [Repository rules](../AGENTS.md) — data, output, metric and reproducibility
   constraints.

If these documents conflict, `AGENTS.md` governs repository constraints, the
scene-specific program governs experiment policy, and the Stage A plan governs
the next execution.

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

Execute Stage A sequentially:

```text
HCM0539 → HCM0421 → HCM0644 → chair → bonsai → HCM0674 → HCM0540
```

Begin with the single HCM0539 canary command in the Stage A plan. Review its
runtime, VRAM, Gaussian growth, finite metrics, complete renders and absence of
model checkpoints before launching HCM0421. Do not launch candidate screening
until all seven B0 references pass the locked contract and the Stage A audit is
approved.
