# Historical five-scene MVP authority

**Status:** CLOSED. Full detail and locked rerender commands remain in Git
history and the repository README.

## Scope and result

The MVP screened:

```text
HCM0644 HCM0674 HCM0540 chair bonsai
```

Candidates:

- antialiased rasterization on HCM0674/HCM0540;
- bounded local-Laplacian L1 weighting on chair/bonsai;
- SH degree 4 on bonsai.

Screen results:

| Scene | Candidate | ΔScore50 | ΔLPIPS | Decision |
|---|---|---:|---:|---|
| chair | local Laplacian | +0.5688 | -0.00683 | production exception |
| bonsai | local Laplacian | +0.2869 | -0.00213 | not selected |
| bonsai | SH4 | +0.5963 | -0.00694 | production exception |
| HCM0674 | antialiased | -4.9568 | +0.04888 | reject |
| HCM0540 | antialiased | -4.7899 | +0.04347 | reject |
| HCM0644 | no screen | — | — | retain B0 |

Chair and bonsai received fresh full-data 30k production under a deadline
exception. They were not fresh paired 30k-confirmed research winners.

## Closed hybrid

`MVP-hybrid-4scene-q99-v1` used:

```text
HCM0421 HCM0539 -> AbsGrad
chair            -> local Laplacian
bonsai           -> SH4
HCM0644 HCM0674 HCM0540 -> byte-identical B0 folders
```

It was submitted and closed with official Score `71.2124`, PSNR `24.629191`,
SSIM `80.7208`, LPIPS `19.4533`, 7/7 matched scenes.

The official result has no per-scene breakdown and is not a tuning signal.

## Preserved invariants

- JPEG Q99, 4:4:4, optimized, non-progressive;
- exact `test_poses.csv` names, poses, intrinsics and dimensions;
- no re-encoding of B0 fallback folders;
- no hidden-test RGB selection;
- no silent artifact overwrite.

Current authority:
`2026-07-27-chair-bonsai-deep-optimization.md`.
