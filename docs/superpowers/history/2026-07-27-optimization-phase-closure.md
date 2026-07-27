# Optimization phase closure through submission v2

**Status:** historical summary; not an execution authority.

This record replaces the old documents as default context. Git history retains
their full implementation detail.

## Closed artifacts

| Phase | Durable result | Scientific status |
|---|---|---|
| B0 production | `B0-submission-q99-v1`, Score `70.98330` | closed baseline |
| Module 1 | detail metrics and pose strata | implemented |
| Module 2 | candidate registry and trainer hooks | implemented |
| Module 3 | stage-first runner, artifact validation and decisions | implemented |
| Stage A | seven 7k B0 internal-holdout references | complete |
| Stage B1 | AbsGrad/scale screens on HCM0539/HCM0421 | complete |
| AbsGrad production | HCM0539/HCM0421 full-data 30k | deadline exception, not paired-confirmed |
| Five-scene screen | AA rejected; chair local-Laplacian and bonsai SH4 selected | complete |
| Auxiliary production | chair local-Laplacian and bonsai SH4 full-data 30k | deadline exception |
| Hybrid submission | `MVP-hybrid-4scene-q99-v1`, Score `71.2124` | submitted and closed |

## Preserved decisions

```text
HCM0421 -> AbsGrad production override
HCM0539 -> AbsGrad production override
chair   -> local-Laplacian production override
bonsai  -> SH4 production override
HCM0644 -> B0
HCM0674 -> B0; antialiasing rejected
HCM0540 -> B0; antialiasing rejected
```

The official v2 aggregate is:

```text
Score          71.2124
PSNR           24.629191
SSIM           0.807208
LPIPS          0.194533
matched scenes 7/7
```

No official per-scene or per-image metrics were available. These values cannot
be retroactively used to prove that one override won.

## Durable harness invariants

- Scene is the unit of training and selection.
- Historical `reference/screen` stages remain 7k and immutable.
- Confirmation and production remain 30k.
- Research and confirmation use internal holdout; production uses full data.
- Test RGB is never a tuning signal.
- Every output filename, pose, intrinsics and dimension comes from
  `test_poses.csv`.
- Old `runs/scene_opt_v1` and `runs/scene_opt_v2` artifacts are not migrated or
  overwritten.
- New work writes to `runs/scene_opt_v3`.

## Why the next plan is narrower

The post-submission full-image audit found that most BTS output is acceptable
and the dominant visible tail risk is concentrated in chair and bonsai.
Chair fails at close-mesh projected scale; bonsai fails at reflective planar
geometry under local pose gaps. The active plan therefore freezes five scenes
and opens only these two auxiliary scenes.

Active authority:
`../specs/2026-07-27-chair-bonsai-deep-optimization.md`.
