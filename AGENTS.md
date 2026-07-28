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
│   ├── <exact image_name from test_poses.csv>
│   └── ...
└── scene_002/...
```
- `image_name` trong `test_poses.csv` là nguồn sự thật tuyệt đối cho filename,
  extension và case. Không tự đổi suffix.
- Payload phải khớp extension: `.jpg`/`.jpeg` là JPEG, `.png` là PNG.
- Inference CLI mặc định JPEG quality 98. Baseline đã đóng
  `B0-submission-q99-v1` dùng quality 99, 4:4:4, optimized, non-progressive;
  không được âm thầm thay đổi cấu hình của baseline đã đóng.
- Image dimensions must exactly match `width`/`height` per row in `test_poses.csv`.
- Every pose in every scene must have a corresponding render — a missing scene or pose invalidates the entire evaluation run, not just that scene.
- Renders must be produced entirely by the pipeline; no manual post-processing (see Engineering constraints).

`SceneManifest.test_output_names` là field legacy của schema v1 và không phải
submission authority. Inference/validator mới phải dùng `test_image_names` để
không làm đổi manifest hash của checkpoint đã train.

## Current baseline closure

Submission cohort đã xác nhận gồm đúng:

```text
HCM0644 HCM0674 HCM0540 HCM0539 HCM0421 chair bonsai
```

- Baseline ID: `B0-submission-q99-v1`; trạng thái: **CLOSED**.
- Evaluator chính thức: Score `70.98330`, PSNR `24.611499`, SSIM `80.4805`,
  LPIPS `19.8195`, `matched_scenes=7/7`.
- Submission dùng JPEG quality 99, 4:4:4, optimized, non-progressive; ZIP cuối
  335 MB dưới giới hạn 350 MB.
- Chưa chạy benchmark local có ground truth. Không được mô tả số liệu evaluator
  chính thức là kết quả local hoặc dùng official test data làm tín hiệu tuning.
- Canonical research pool 18 BTS scene vẫn giữ riêng; hai auxiliary scene không
  được nhập vào pool đó hoặc dùng để suy luận cross-scene generalization.
- Mọi thay đổi model, loss, optimizer, densification, renderer hoặc codec phải
  tạo candidate/baseline ID mới.
- Modules 1–3, Stage A, Stage B1, five-scene screen, bốn production override,
  rerender và submission v2 đã hoàn tất. Durable history nằm tại
  `docs/superpowers/history/2026-07-27-optimization-phase-closure.md`.
- Research authority hiện tại là
  `docs/superpowers/specs/2026-07-27-chair-bonsai-deep-optimization.md`.
  Chỉ `chair` và `bonsai` được mở cho `scene_opt_v3`; năm scene còn lại bị
  freeze trong plan này.
- Stage `research` dùng optimizer horizon 30k, dừng ở 15k, internal holdout,
  không checkpoint và chỉ chấp nhận `chair/bonsai`. Production vẫn là fresh
  full-data 30k không holdout.
- Phase 1 dùng artifact riêng `holdout_research_v3.json`; không được thay hoặc
  ghi đè `holdout.json`. Hai incumbent L4 15k và artifact contract đã được
  validate. Candidate-theory gate và local SfM audit nằm tại
  `docs/superpowers/research/2026-07-27-chair-bonsai-candidate-validation.md`;
  Phase 2 chỉ mở candidate sau khi code, contract và test được merge.
- Các candidate chưa có trong registry vẫn là reserved. Hai candidate chair
  `E3-chair-observation-scale-v1` và
  `E4-chair-observation-scale-absgrad-v1` đã được triển khai chỉ cho
  `chair/research`; confirm và production phải từ chối cả hai. Không được dùng
  tên trong docs để bypass preflight.
- E4 đã bị paired holdout bác bỏ tại 15k. Candidate
  `E5-chair-observation-scale-mcmc-v1` là ngoại lệ research full-horizon 30k:
  cap 2M, relocation tới 25k, rolling recovery mỗi 3k và internal holdout.
  Chạy bằng `scripts/run_chair_mcmc_research.sh`. E3-30k chỉ được chi compute
  sau khi E5-30k vượt rõ lower-bound E3-15k; E5 vẫn bị cấm ở confirm/production
  cho tới khi có decision artifact mới.

## Closed hybrid submission

Submission ID `MVP-hybrid-4scene-q99-v1` đã nộp và **CLOSED**. Evaluator chính
thức: Score `71.2124`, PSNR `24.629191`, SSIM `80.7208`, LPIPS `19.4533`,
`matched_scenes=7/7`. Bốn checkpoint override và ba fallback scene là:

| Scene | Render source |
|---|---|
| HCM0421, HCM0539 | `runs/scene_opt_v1/production_mvp/scenes/<scene>` |
| chair, bonsai | `runs/scene_opt_v2/production_mvp/scenes/<scene>` |
| HCM0644, HCM0674, HCM0540 | byte-identical folders từ `B0-submission-q99-v1` |

Không dùng kết quả aggregate này để suy ra per-scene winner. Mọi submission
sau nó phải có ID mới. Các lệnh rerender lịch sử vẫn nằm trong README nhưng
không còn là research authority.

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
- `PSNR_max=50` đã được xác nhận bằng đại số từ số evaluator chính thức:
  `PSNR_max=49.99983` sau sai số làm tròn và recomputed Score `70.9832494`
  khớp Score công bố `70.98330`.
- LPIPS score depends on the backbone (`alex` vs `vgg` — the `lpips` package defaults to AlexNet, but VGG is also common in NVS papers and gives different absolute values). Confirm which backbone the grading harness uses; mismatched backbones make local LPIPS numbers not comparable to the leaderboard.
- SSIM depends on window size and kernel (Gaussian vs uniform) and whether it's computed per-channel-then-averaged or on luminance only. Use the same implementation/config as the grading harness if specified, otherwise default to the standard `skimage`/`scikit-image` or `torchmetrics` Gaussian-window SSIM (11×11, σ=1.5) as the closest common default.
- PSNR remains sensitive to color space and value range ([0,255] vs [0,1]); the repository harness uses RGB `[0,1]` and `data_range=1`.
- Because LPIPS carries the largest weight (0.4) and is a *perceptual* metric, optimizing purely for pixel-space error (PSNR/SSIM) — e.g. via over-smoothing — can hurt LPIPS and lower the total score. Track all three metrics during development, not PSNR alone.
- Final score is the mean across all scenes in the test set.

## Engineering constraints
- **No external data.** Use only the imagery and reconstruction provided per scene. Do not source additional images, video, or 3D assets of the same site/object, and do not collect supplementary field data for the given scenes.
- **No ground-truth leakage.** Do not attempt to access or infer held-out test images through any channel outside the documented data contract.
- **Fully automated output.** Every rendered image must come directly from the pipeline — no manual compositing, retouching, or per-pose manual intervention.
- **Reproducibility is a deliverable, not an afterthought.** Track: training/inference code, exact configs, dependency versions, checkpoints, and training logs, from the start — not reconstructed retroactively. Fix random seeds where applicable.
- **Preserve production module boundaries.** Do not place method-specific
  geometry, diagnostics, artifact validation and orchestration in one file.
  Keep each module focused on one responsibility, expose a small typed API,
  and move reusable policy or math into a named module before merging. A
  passing test suite does not justify leaving an oversized catch-all module.

## Data availability / milestones
| | |
|---|---|
| Train images per scene | 150–300 |
| Target poses per scene | 40–70 |
| Data volume per scene | 200–300 MB |
| Test data released | 2026-07-02 |
| Submission deadline | 2026-07-30 |

## Open questions — confirm before implementing, don't assume
- LPIPS backbone and SSIM window/channel/aggregation details used by the official grading harness.
- Whether the dev/validation dataset matches the held-out test set in structure and distribution, or only approximately.
- Whether future compute remains one NVIDIA L4 with 23 GB VRAM; Stage A was
  measured on that hardware, but later resource availability must not be assumed.
