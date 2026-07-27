# Documentation status

## Current position — 2026-07-27

- `B0-submission-q99-v1` is closed at official Score `70.98330`.
- `MVP-hybrid-4scene-q99-v1` is submitted and closed at official Score
  `71.2124`, PSNR `24.629191`, SSIM `0.807208`, LPIPS `0.194533`, 7/7 scenes.
- All Stage A, Stage B1, five-scene MVP, production, rerender and assembly work
  through submission v2 is historical.
- The only active research scope is deep optimization of `chair` and `bonsai`.
- Five other scenes are frozen until this plan finishes.

## Active read order

Only these documents are required:

1. [Repository rules](../AGENTS.md)
2. [Active chair/bonsai authority](superpowers/specs/2026-07-27-chair-bonsai-deep-optimization.md)
3. [Active execution plan](superpowers/plans/2026-07-27-chair-bonsai-deep-optimization.md)
4. [Repository operations](../README.md)

For prior decisions, read only the
[phase closure](superpowers/history/2026-07-27-optimization-phase-closure.md).

## Active harness

New evidence writes to:

```text
runs/scene_opt_v3/research/<scene>/<candidate>/
```

The `research` stage is locked to `chair/bonsai`, uses a 30k optimizer
schedule stopped at 15k, internal holdout and no model checkpoint.

```bash
bash scripts/run_chair_bonsai_research.sh <chair|bonsai> <candidate-id>
```

Chair E3 and E4 are registered research-only candidates. E3 completed its
paired L4 control; E4 is the next executable AbsGrad screen. Unregistered
candidate IDs in the active spec remain reserved.

## Historical documents

The 2026-07-22 through 2026-07-26 specs/plans are provenance only. Their
durable conclusions are consolidated in the phase-closure record. They must
not be treated as executable authority, and their v1/v2 artifacts must not be
migrated or overwritten.
