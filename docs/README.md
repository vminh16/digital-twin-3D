# Documentation status

## Current position — 2026-07-29

- `B0-submission-q99-v1` is closed at official Score `70.98330`.
- `MVP-hybrid-4scene-q99-v1` is submitted and closed at official Score
  `71.2124`, PSNR `24.629191`, SSIM `0.807208`, LPIPS `0.194533`, 7/7 scenes.
- All Stage A, Stage B1, five-scene MVP, production, rerender and assembly work
  through submission v2 is historical.
- The only active research scope is deep optimization of `chair` and `bonsai`.
- Five other scenes are frozen until this plan finishes.
- Chair E3 repaired the research observation mapping; E4 and E5 were rejected.
  E6 was rejected as an under-densified implementation, not as a valid
  falsification of Perceptual-GS. E7 corrected perceptual ADC completed 15k
  and passed its mechanism gate, but failed quality, compute and tail gates.
  It must not resume to 30k. E8 3D shape-aware spectral splitting is the final
  active chair candidate. Its implementation and local tests pass; one fresh
  L4 15k research gate is now authorized.

## Active read order

Only these documents are required:

1. [Repository rules](../AGENTS.md)
2. [Consolidated diagnosis and experiment history](superpowers/history/2026-07-29-project-diagnosis-and-experiment-report.md)
3. [Active chair/bonsai authority](superpowers/specs/2026-07-27-chair-bonsai-deep-optimization.md)
4. [Active execution plan](superpowers/plans/2026-07-27-chair-bonsai-deep-optimization.md)
5. [Repository operations](../README.md)

The older
[phase closure](superpowers/history/2026-07-27-optimization-phase-closure.md)
is retained as provenance; the consolidated report supersedes it as the
default historical context.

## Active harness

New evidence writes to:

```text
runs/scene_opt_v3/research/<scene>/<candidate>/
```

The `research` stage is locked to `chair/bonsai`, uses a 30k optimizer
schedule normally stopped at 15k and an internal holdout. Locked wrappers may
enable rolling recovery for staged candidates.

E3 remains the corrected research control; E4 through E7 are closed as
rejected. E8 is registered and its wrapper performs the required CUDA smoke
before the fresh 15k run. It is not authorized for 30k resume or production.
Other unregistered candidate IDs remain reserved.

## Historical documents

The 2026-07-22 through 2026-07-26 specs/plans are provenance only. Their
durable conclusions are consolidated in the 2026-07-29 diagnosis report. They
must not be treated as executable authority, and their v1/v2 artifacts must
not be migrated or overwritten.
