# AGENTS.md — BTS Digital Twin: Novel View Synthesis

## Project overview
This project builds a pipeline that reconstructs the implicit 3D structure of telecom BTS (base transceiver station) sites from calibrated multi-view RGB imagery, and synthesizes photorealistic RGB renders at camera poses that were never captured. It supports a Digital Twin use case: high-fidelity 3D replicas of telecom infrastructure for remote monitoring, inspection, maintenance, and installation planning. Domain: 3D Vision, Neural Rendering, Novel View Synthesis (NVS).

## Task definition
- **Input** (per scene): 150–300 calibrated RGB images, camera intrinsics, camera poses, and a COLMAP sparse reconstruction (sparse point cloud).
- **Output** (per scene): photorealistic RGB renders at 40–70 held-out target poses, matching the requested geometry, object placement, and exact image dimensions.
- **Unit of work is a scene.** Training and (for most NVS methods) inference happen per scene — there is no cross-scene generalization requirement. Plan compute budget and orchestration around `N scenes × per-scene train time`, not a single global model.

**Recommended technical direction:** 3D Gaussian Splatting or a NeRF-family method (Instant-NGP, Nerfacto, Zip-NeRF). `points3D.bin` already provides a usable point cloud for Gaussian/NeRF initialization — do not re-run SfM/COLMAP from scratch; it wastes compute and the provided reconstruction is the intended starting point.

## Data contract
```
scene_XXX/
├── train/
│   ├── images/              # 150–300 RGB images
│   └── sparse/0/            # COLMAP sparse reconstruction — use as-is
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
└── test/
    └── test_poses.csv       # 40–70 target poses to render
```

`test_poses.csv` columns, in this exact order:
```
image_name, qw, qx, qy, qz, tx, ty, tz, fx, fy, cx, cy, width, height
```
- `qw,qx,qy,qz` / `tx,ty,tz`: rotation (quaternion) + translation in **COLMAP convention**. Before batch-rendering the test set, sanity-check the world↔camera direction by re-rendering one *train* image with known ground truth — do not assume the convention without verifying.
- `fx,fy,cx,cy,width,height`: intrinsics and output dimensions — renders **must match exactly**.

## Output contract
```
outputs/
├── scene_001/
│   ├── 0001.png   # filename MUST match image_name in test_poses.csv
│   └── ...
└── scene_002/...
```
- Image dimensions must exactly match `width`/`height` per row in `test_poses.csv`.
- Every pose in every scene must have a corresponding render — a missing scene or pose invalidates the entire evaluation run, not just that scene.
- Renders must be produced entirely by the pipeline; no manual post-processing (see Engineering constraints).

## Evaluation metrics
Final metric is a weighted composite, matching standard NVS benchmarking (Mip-NeRF / 3DGS-style evaluation):

```
Score = 0.4 × (1 − LPIPS) + 0.3 × SSIM + 0.3 × PSNR_norm
PSNR_norm = clamp(PSNR / PSNR_max, 0, 1)
```

| Metric | Direction | Definition | Reference |
|---|---|---|---|
| LPIPS | lower is better | Perceptual similarity via deep features | Zhang et al., CVPR 2018 |
| SSIM | higher is better | Structural similarity (luminance/contrast/structure) | Wang et al., IEEE TIP 2004 |
| PSNR | higher is better | Pixel-level error, normalized by a fixed `PSNR_max` before averaging | Wang et al., IEEE TIP 2004 |

**Implementation details that change the score and must be pinned down before trusting local evaluation numbers:**
- LPIPS score depends on the backbone (`alex` vs `vgg` — the `lpips` package defaults to AlexNet, but VGG is also common in NVS papers and gives different absolute values). Confirm which backbone the grading harness uses; mismatched backbones make local LPIPS numbers not comparable to the leaderboard.
- SSIM depends on window size and kernel (Gaussian vs uniform) and whether it's computed per-channel-then-averaged or on luminance only. Use the same implementation/config as the grading harness if specified, otherwise default to the standard `skimage`/`scikit-image` or `torchmetrics` Gaussian-window SSIM (11×11, σ=1.5) as the closest common default.
- PSNR is sensitive to color space and value range ([0,255] vs [0,1]) and to `PSNR_max`, which is not fixed by convention — confirm its value (see Open questions).
- Because LPIPS carries the largest weight (0.4) and is a *perceptual* metric, optimizing purely for pixel-space error (PSNR/SSIM) — e.g. via over-smoothing — can hurt LPIPS and lower the total score. Track all three metrics during development, not PSNR alone.
- Final score is the mean across all scenes in the test set.

## Engineering constraints
- **No external data.** Use only the imagery and reconstruction provided per scene. Do not source additional images, video, or 3D assets of the same site/object, and do not collect supplementary field data for the given scenes.
- **No ground-truth leakage.** Do not attempt to access or infer held-out test images through any channel outside the documented data contract.
- **Fully automated output.** Every rendered image must come directly from the pipeline — no manual compositing, retouching, or per-pose manual intervention.
- **Reproducibility is a deliverable, not an afterthought.** Track: training/inference code, exact configs, dependency versions, checkpoints, and training logs, from the start — not reconstructed retroactively. Fix random seeds where applicable.

## Data availability / milestones
| | |
|---|---|
| Train images per scene | 150–300 |
| Target poses per scene | 40–70 |
| Data volume per scene | 200–300 MB |
| Test data released | 2026-07-02 |
| Submission deadline | 2026-07-30 |

## Open questions — confirm before implementing, don't assume
- Exact value of `PSNR_max` used for normalization, and the LPIPS backbone / SSIM window config used by the grading harness.
- Whether the dev/validation dataset matches the held-out test set in structure and distribution, or only approximately.
- Actual available compute (GPU type/count, VRAM) — 3DGS/NeRF training is per-scene, so total scene count × per-scene training time is likely the real bottleneck, not model quality alone.
- Whether any baseline/starter implementation already exists for this project, before building a pipeline from scratch.