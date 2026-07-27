# Historical Stage A summary

**Status:** COMPLETED 2026-07-24.

Stage A created one deterministic, full-resolution, seed-0, 7k
internal-holdout `B0-reference` for each submission scene. All seven reports
passed artifact validation with 174/174 validation renders, no model
checkpoint and no NaN/Inf. Total measured L4 runtime was about 95.6 minutes.

Scene-balanced diagnostics were:

```text
Score50 64.654
PSNR    21.050
SSIM     0.7267
LPIPS    0.2444
```

These are local 7k diagnostics and are not comparable to the official hidden
test. Artifacts remain under `runs/scene_opt_v1/reference/<scene>`.

Current history summary:
`../history/2026-07-27-optimization-phase-closure.md`.
