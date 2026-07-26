# Documentation status

## Current position — 2026-07-26

- Closed baseline: `B0-submission-q99-v1`, official Score `70.98330`, 7/7
  scenes.
- Stage A completed seven deterministic `B0-reference` 7k holdouts.
- Stage B1 completed on `HCM0421/HCM0539`.
- Their AbsGrad production checkpoints are deadline-exception MVP artifacts,
  not paired-confirmed research winners.
- The five-scene 7k screen is complete:

```text
HCM0644 -> retain B0
HCM0674 -> antialiased rejected; retain B0
HCM0540 -> antialiased rejected; retain B0
chair   -> local sharpness-weighted loss selected
bonsai  -> SH4 selected
```

- Deadline-exception full-data 30k production is authorized for chair and
  bonsai. It is operational MVP evidence, not paired 30k confirmation.

## Read order

Only these documents are required for active work:

1. [Repository rules](../AGENTS.md) — immutable data, output, metric and
   reproducibility constraints.
2. [Five-scene MVP authority](superpowers/specs/2026-07-26-five-scene-mvp.md) —
   scope, candidates, gates and completion.
3. [Five-scene screen plan](superpowers/plans/2026-07-26-five-scene-mvp-screen.md) —
   implementation and VM execution order.
4. [Repository README](../README.md) — installation and operational commands.

## Historical documents

The two 2026-07-22 specs and completed Stage A/B1 plans remain in the tree only
for provenance. Their durable constraints are consolidated in the active MVP
authority. Old deleted Phase 3/4, C1 and module plans remain recoverable through
Git history; they are not executable authorities.

## Execution boundary

- Local workstation: edit code, run unit tests and build the deployment bundle.
- NVIDIA L4 VM: sequential full-data 30k production for chair and bonsai.
- Existing `runs/scene_opt_v1/reference/` reports remain the paired B0
  authority.
- New runs and decisions use `runs/scene_opt_v2/`.
- Only the deterministic chair and bonsai winners may enter production.
- No hidden-test render is used for tuning.
