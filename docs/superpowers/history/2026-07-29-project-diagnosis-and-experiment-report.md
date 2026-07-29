# Báo cáo tổng thể: chẩn đoán nguyên nhân và lịch sử thí nghiệm

**Ngày chốt:** 2026-07-29
**Phạm vi:** toàn bộ vòng đời repository, từ contract nền tảng, pilot,
qualification, baseline B0, các screen/MVP, submission v2 đến research
`scene_opt_v3`.
**Trạng thái:** nguồn tổng hợp lịch sử và chẩn đoán hiện tại. Tài liệu này
không thay thế execution authority của E7 trong
`../specs/2026-07-27-chair-bonsai-deep-optimization.md`.

## 1. Kết luận điều hành

Pipeline đã hoàn thành đúng hợp đồng submission và hai lần được evaluator xác
nhận đủ `7/7` scene. Bản hybrid v2 cải thiện thật so với B0 ở aggregate:

| Submission | Score | PSNR | SSIM | LPIPS | Matched |
|---|---:|---:|---:|---:|---:|
| `B0-submission-q99-v1` | 70.98330 | 24.611499 | 0.804805 | 0.198195 | 7/7 |
| `MVP-hybrid-4scene-q99-v1` | 71.2124 | 24.629191 | 0.807208 | 0.194533 | 7/7 |
| Thay đổi | **+0.2291** | **+0.017692** | **+0.002403** | **-0.003662** | — |

Kết quả này chứng minh toàn bộ tổ hợp submission v2 tốt hơn B0, nhưng không
chứng minh riêng từng override tốt hơn trên hidden test vì evaluator không trả
per-scene hoặc per-image metrics.

Chẩn đoán trung tâm của dự án hiện tại:

1. Chất lượng không fail đồng đều trên bảy scene. Năm scene BTS nhìn chung ổn
   hơn; rủi ro thị giác nặng tập trung ở `chair` và `bonsai`.
2. `chair` fail chủ yếu ở cụm pose nhìn gần qua lưới ghế: Gaussian quá lớn hoặc
   quá kéo dài xuất hiện trong một khoảng góc nhìn hẹp, tạo màn sương, ghosting
   và smear. Motion blur trong một số ảnh train là yếu tố phụ, không giải thích
   được các frame catastrophic.
3. `bonsai` fail chủ yếu ở mặt bàn phản chiếu và foreground khi target pose xa
   các pose train lân cận. Đây là thiếu ràng buộc geometry/depth kết hợp
   view-dependent reflection; SH4 chỉ tăng appearance capacity nên không chữa
   được hình học.
4. Tăng step đơn thuần không phải lời giải đã được chứng minh. Production đã
   chạy 30k; densification mặc định dừng ở 15k. Train lâu hơn chỉ tiếp tục tối
   ưu các Gaussian hiện có, không tự sửa topology bị thiếu hoặc đặt sai.
5. Trong selector research của `chair` có một defect thật: observation COLMAP
   nằm ở `1080x1920`, ảnh dùng train là `720x1280`; phép suy ra scale cũ làm
   tròn `1.5` thành `2`. E3 sửa defect này và tăng Score50 nội bộ `+0.9631`,
   nhưng đây là sửa độ tin cậy của research holdout, không phải production
   candidate vì full-data production dùng màu sẵn trong `points3D.bin`.
6. E4 AbsGrad và E5 MCMC không qua gate. E6 cho thấy PSNR tăng nhưng LPIPS giảm
   rõ vì implementation vô tình thay standard ADC bằng các event perceptual
   thưa, chỉ còn khoảng `297k` Gaussian. E6 bị bác, nhưng không phải bằng chứng
   bác bỏ phương pháp Perceptual-GS công bố.
7. E7 đã chạy xong 15k. Nó sửa đúng lỗi cơ chế E6 và vượt population gate,
   nhưng fail quality gate: Score50 chỉ `+0.0647` so với E3, SSIM và
   spurious-edge xấu hơn, scale/radius tail tăng mạnh. E7 bị reject và không
   được resume 30k.

## 2. Quy tắc đọc bằng chứng

Ba mức bằng chứng được dùng xuyên suốt:

- **L — local evidence:** đo từ input, internal holdout hoặc artifact của repo.
- **O — official evidence:** số aggregate do evaluator chính thức trả.
- **P — published evidence:** kết quả từ bài báo trên dataset của bài báo.

Không mức nào cho phép suy ra hidden test per-scene khi không có ground-truth.
Đặc biệt:

- metric 7k/15k local không so trực tiếp với official metric;
- hai run khác holdout hash hoặc khác horizon không phải paired causal test;
- ảnh test render chỉ cho visual diagnosis, không phải tuning target;
- train convergence trên một camera không thay thế holdout evaluation.

Repo dùng local composite:

```text
Score50 = 0.6 * PSNR + 30 * SSIM + 40 * (1 - LPIPS)
```

Tương đương công thức official khi `PSNR_max=50`. LPIPS càng thấp càng tốt.
Khi so hai candidate, hằng số 40 triệt tiêu nên:

```text
delta Score50 = 0.6 * delta PSNR + 30 * delta SSIM - 40 * delta LPIPS
```

## 3. Hợp đồng kỹ thuật đã chốt

Đơn vị train và inference là từng scene. Mỗi scene dùng ảnh RGB, camera COLMAP
và `points3D.bin`; không có yêu cầu generalization chéo scene.

Baseline chung:

| Nhóm | Giá trị khóa |
|---|---|
| Model | 3D Gaussian Splatting qua `gsplat` |
| Init | sparse COLMAP, normalized scene coordinates |
| Loss | L1 + DSSIM, `lambda_dssim=0.2` |
| Optimizer | Adam, FP32, seed 0 |
| Appearance | SH3 nếu candidate không override |
| ADC | refine mỗi 100 step, bắt đầu 500, dừng 15k |
| Density | `grow_grad2d=0.0002`, `grow_scale3d=0.01` |
| Prune/reset | `prune_opa=0.005`, reset mỗi 3k |
| Production | full resolution, 30k, toàn bộ train, không holdout |
| Research | internal holdout; v3 dùng horizon 30k và stop 15k |
| Submission | tên/đuôi/kích thước đúng `test_poses.csv` |
| JPEG | Q99, 4:4:4, optimized, non-progressive |

Các module chính hiện có:

| Thư mục | Trách nhiệm |
|---|---|
| `src/bts_nvs/cameras` | quaternion, intrinsics, world/camera transforms |
| `src/bts_nvs/data` | COLMAP readers, manifest, dataset, holdout |
| `src/bts_nvs/models` | Gaussian initialization và renderer-facing state |
| `src/bts_nvs/rendering` | gsplat adapter và inference test-pose |
| `src/bts_nvs/training` | trainer, optimizer, checkpoint/recovery |
| `src/bts_nvs/evaluation` | PSNR/SSIM/LPIPS và detail/tail diagnostics |
| `src/bts_nvs/experiments` | registry, policy, runner, reports, decisions |
| `src/bts_nvs/submission` | output validation và JPEG packaging |
| `scripts` | wrapper khóa stage/candidate và orchestration |
| `runs/scene_opt_v1..v3` | artifact theo thế hệ harness |

## 4. Lịch sử dự án theo phase

### 4.1 Nền tảng dữ liệu và geometry — 2026-07-13

Các contract camera, COLMAP source, manifest, dataset diagnostics, benchmark
skeleton và submission validator được xây trước. Quyết định đúng ở phase này
là dùng reconstruction được cung cấp, không chạy lại SfM.

Kết quả bền:

- camera convention và output dimensions có validator;
- manifest/hash cho phép checkpoint gắn đúng scene/data;
- exact output filename đến từ `test_poses.csv`;
- benchmark và submission contract được tách khỏi trainer.

### 4.2 Baseline 3DGS và pilot — 2026-07-14 đến 2026-07-15

Repo bổ sung normalized Gaussian initialization, gsplat renderer, masked loss,
optimizer, adaptive density và training engine. HCM0181 được dùng làm pilot ở
factor 4/2/1 và 500/3k/7k step để kiểm tra:

- loss hữu hạn và render không blank;
- density control thực sự thay đổi Gaussian population;
- full-resolution chạy được trên NVIDIA L4;
- headless OpenCV không phụ thuộc UI/`libGL`.

Các checkpoint pilot còn trong `runs/HCM0181`, nhưng log/summary đầy đủ của
pilot không còn ở current tree. Chúng là engineering smoke, không phải
submission evidence.

### 4.3 Phase 4: inventory, holdout và qualification — 2026-07-16

Phase 4 thêm:

- inventory feasibility gate;
- `pose_fps_guard2_v1` leakage-controlled holdout;
- cached/uncached input profiling;
- qualification report và resource gates;
- 30k full-length dry run.

Qualification sáu scene đã so B0-reference với compact density:

| Candidate | PSNR | SSIM | LPIPS | Mean Gaussian | VRAM peak | Tổng thời gian |
|---|---:|---:|---:|---:|---:|---:|
| B0-reference | 21.9454 | 0.73480 | 0.18993 | 3.485M | 7.20 GB | 5555.6 s |
| B0-compact | 21.8357 | 0.72738 | 0.21125 | 1.927M | 4.21 GB | 3631.8 s |

Compact tiết kiệm compute nhưng giảm chất lượng trên cả sáu calibration scene;
decision chọn B0-reference. Đây là bằng chứng đầu tiên rằng giảm density quá
mạnh không phù hợp mục tiêu score.

### 4.4 30k dry run HCM0181 — 2026-07-16

Thông số:

```text
max_steps=30000
full resolution=1320x989
internal train/holdout=169/25
refine stop=15000
seed=0, FP32, lambda_dssim=0.2
```

Resource và convergence:

| Thuộc tính | Giá trị |
|---|---:|
| Thời gian | 10,427.47 s = 2.90 h |
| Gaussian cuối | 6,861,805 |
| VRAM peak | 11.79 GB |
| Final train-camera PSNR | 27.7467 dB |
| Final train-camera SSIM | 0.93182 |

Internal validation 7k→30k được research report cũ ghi:

| Mốc | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| 7k | 21.733 | 0.768 | 0.160 |
| 30k | 22.692 | 0.805 | 0.111 |

`ΔScore50=+3.646`. Kết luận đúng: 7k đủ để screen mechanism nhưng không phải
MVP production. Kết luận không được phép: mọi scene sẽ tiếp tục tăng nếu kéo
tới 70k.

### 4.5 Backend acceleration — 2026-07-17

Ba run 1k trên HCM0181:

| Backend | Precision | Time | Median CUDA step | Final G | VRAM |
|---|---|---:|---:|---:|---:|
| reference | FP32 Adam | 38.19 s | 33.51 ms | 551,388 | 958.7 MB |
| fused | FP32 Adam-fused | 36.51 s | 32.61 ms | 550,698 | 958.2 MB |
| amp | AMP FP16 + fused | 36.36 s | 31.72 ms | 556,116 | 966.9 MB |

Các run xác nhận backend path chạy hữu hạn. Chênh lệch 1k nhỏ; không dùng bảng
này để tuyên bố AMP cải thiện quality.

### 4.6 Full-training, inference và B0 submission — 2026-07-17 đến 2026-07-19

Sequential full-training runner, selected-scene mode, test-pose inference,
benchmark CLI, JPEG converter và submission validator được hoàn thiện.

`B0-submission-q99-v1` đóng trên:

```text
HCM0644 HCM0674 HCM0540 HCM0539 HCM0421 chair bonsai
```

Official result:

```text
Score 70.98330
PSNR 24.611499
SSIM 0.804805
LPIPS 0.198195
matched 7/7
```

Đây là baseline duy nhất được evaluator xác nhận trước tối ưu.

### 4.7 Research thiết kế ban đầu — 2026-07-19

Research ban đầu xếp density control cao hơn appearance rewrite:

1. AbsGrad/revised opacity;
2. MCMC cap 2M/3M;
3. frequency regularization;
4. Pixel-GS;
5. antialiasing chỉ khi camera audit chứng minh sampling shift;
6. NeRF/2DGS/WildGaussians rewrite bị hoãn vì deadline.

Giá trị của phase này là tạo giả thuyết và compute ladder. Nó không phải bằng
chứng local rằng tất cả candidate sẽ thắng.

### 4.8 Harness thí nghiệm scene-specific — 2026-07-22

Ba module được triển khai:

1. detail metrics, edge diagnostics và pose strata;
2. candidate registry, provenance và density hooks;
3. stage-first runner, resource/artifact validation, deterministic decision.

Harness tách `reference`, `screen`, `confirm`, `production` và sau đó thêm
`research`. Các artifact v1/v2 không bị overwrite bởi v3.

### 4.9 Stage A: bảy B0 reference 7k — 2026-07-23/24

Tất cả 174/174 holdout render hợp lệ. Tổng L4 khoảng 95.6 phút.

| Scene | Internal train | Holdout | Guard | Resolution |
|---|---:|---:|---:|---:|
| HCM0644 | 169 | 24 | 47 | 1320×989 |
| HCM0674 | 168 | 28 | 44 | 1320×989 |
| HCM0540 | 170 | 27 | 43 | 1320×989 |
| HCM0539 | 170 | 25 | 45 | 1320×989 |
| HCM0421 | 169 | 26 | 45 | 1320×989 |
| chair | 145 | 20 | 40 | 720×1280 |
| bonsai | 176 | 24 | 48 | 1920×1080 |

| Scene | PSNR | SSIM | LPIPS | Score50 | Final G | VRAM MB | Holdout |
|---|---:|---:|---:|---:|---:|---:|---:|
| HCM0644 | 21.9553 | 0.76431 | 0.17369 | **69.1547** | 3,496,390 | 6,310.8 | 24 |
| HCM0539 | 21.9355 | 0.75427 | 0.17232 | 68.8966 | 3,469,365 | 6,230.9 | 25 |
| HCM0421 | 22.0800 | 0.74343 | 0.17868 | 68.4036 | 3,450,171 | 6,181.8 | 26 |
| HCM0540 | 21.1385 | 0.71197 | 0.20597 | 65.8036 | 3,451,638 | 6,177.4 | 27 |
| HCM0674 | 18.9517 | 0.69066 | 0.23163 | 62.8255 | 3,219,830 | 5,868.1 | 28 |
| chair | 20.9518 | 0.68758 | 0.35594 | 58.9608 | 1,092,849 | 2,144.3 | 20 |
| bonsai | 20.3325 | 0.73455 | 0.39267 | 58.5293 | 536,453 | 1,379.9 | 24 |

Scene-balanced mean là Score50 `64.654`, PSNR `21.050`, SSIM `0.7267`,
LPIPS `0.2444`.

Hai auxiliary scene có LPIPS xấu nhất; trong BTS, HCM0674 yếu nhất. Đây là
screen local 7k, không phải official per-scene ranking.

### 4.10 Stage B1: HCM0421/HCM0539 — 2026-07-24/25

Hai density candidate:

| Candidate | Khác B0 |
|---|---|
| `E1-density-absgrad-t04-v1` | `absgrad=true`, `grow_grad2d=0.0004` |
| `E1-density-scale005-v1` | `grow_scale3d=0.005` |

Kết quả:

| Scene/candidate | Score50 | ΔScore | ΔLPIPS | Δ hard Score | Time ratio | Decision |
|---|---:|---:|---:|---:|---:|---|
| HCM0421 AbsGrad | 69.7672 | +1.3636 | -0.02193 | +0.2340 | 1.347x | screen reject theo time; production exception |
| HCM0421 scale005 | 68.2647 | -0.1390 | +0.00022 | -1.2303 | — | reject |
| HCM0539 AbsGrad | 69.7820 | +0.8853 | -0.01558 | -0.3789 | 1.361x | screen reject theo time/tail; production exception |
| HCM0539 scale005 | 69.0881 | +0.1915 | -0.00497 | -0.2125 | — | reject |

Điểm cần giữ đúng:

- JSON decision nghiêm ngặt vẫn chọn B0;
- deadline cho phép chạy fresh full-data 30k AbsGrad;
- đây là operational override, không phải paired 30k-confirmed winner.

Production artifact còn local:

| Scene | Steps | Time | Final G | VRAM | Candidate |
|---|---:|---:|---:|---:|---|
| HCM0421 | 30k | 969.3 s | 5,450,847 | 6.94 GB | AbsGrad |
| HCM0539 | 30k | 6,389.6 s | 5,886,580 | 10.16 GB | AbsGrad |

### 4.11 Five-scene screen và auxiliary production — 2026-07-26

Candidate:

| Candidate | Thông số riêng |
|---|---|
| `E2-raster-aa-v1` | rasterizer `antialiased` |
| `E2-loss-local-laplacian-v1` | pixel weight local Laplacian, patch 31, floor 0.5 |
| `E2-appearance-sh4-v1` | `max_sh_degree=4` |

Kết quả 7k:

| Scene/candidate | ΔScore50 | ΔLPIPS | Kết luận |
|---|---:|---:|---|
| chair local-Laplacian | +0.5688 | -0.00683 | chọn production exception |
| bonsai local-Laplacian | +0.2869 | -0.00213 | dương nhưng thua SH4 |
| bonsai SH4 | +0.5963 | -0.00694 | chọn production exception |
| HCM0540 antialiased | **-4.7899** | **+0.04347** | reject mạnh |
| HCM0674 antialiased | **-4.9568** | **+0.04888** | reject mạnh |
| HCM0644 | — | — | không screen, giữ B0 |

AA làm xấu cả pixel và perceptual metrics. Do đó vấn đề của HCM0540/HCM0674
không thể quy giản thành aliasing do sampling-rate shift.

Chair và bonsai được train fresh full-data 30k. Current checkout không giữ
`runs/scene_opt_v2/production_mvp/scenes/{chair,bonsai}`, nên report không bịa
resource/final-G metrics cho hai production run này.

Console log được lưu trong lịch sử trao đổi xác nhận riêng chair:

```text
initial Gaussians=79,828
steps=30,000
train-camera PSNR delta=+10.424276
train-camera SSIM delta=+0.493750
non_blank=true
```

Đây chỉ là convergence log; validator fail xảy ra sau train vì stale
`validation_renders`. Nó không phải holdout hoặc official quality metric.

### 4.12 Rerender và hybrid submission v2 — 2026-07-26/27

Mapping đã nộp:

| Scene | Nguồn render |
|---|---|
| HCM0421, HCM0539 | production AbsGrad |
| chair | production local-Laplacian |
| bonsai | production SH4 |
| HCM0644, HCM0674, HCM0540 | folder B0 byte-identical |

Ba incident harness đã được xử lý trong quá trình production/rerender:

1. Production chair train xong nhưng validator thấy
   `validation_renders/` cũ từ holdout và từ chối. Đây là stale workspace
   artifact, không phải train không lưu checkpoint. Fix sau đó dọn/reject đúng
   holdout artifact trong production workspace.
2. Một thư mục bonsai non-empty chỉ có config/environment/metrics/previews,
   thiếu `summary.json`, nên recovery validator từ chối. Đây là interrupted
   run không có valid recovery state.
3. Một checkpoint HCM0539 từng bị inference từ chối vì `metrics.jsonl` không
   tăng đơn điệu tại step 12001, thường do resume nối log trùng/out-of-order.
   Đây là artifact-integrity failure; không được bypass validator để render.

Các lỗi trên thuộc orchestration/provenance, không phải nguyên nhân thị giác
của chair/bonsai.

## 5. Chẩn đoán bộ output/submission hiện tại

### 5.1 Điều đã được chứng minh

- Evaluator xác nhận submission v2 đủ 7/7 scene.
- Aggregate Score/SSIM/LPIPS tốt hơn B0.
- Không có official per-scene metrics.
- Full-image audit trước đó bao phủ 453 train image và 86 render của
  chair/bonsai, không chỉ đọc summary artifact.

### 5.2 Khoảng trống artifact trong current checkout

Tại thời điểm report:

- `submission_round1/` rỗng;
- `outputs/` chỉ có 5 folder BTS, mỗi folder 60 PNG, tổng 300 ảnh và khoảng
  616.5 MB;
- không có chair/bonsai trong `outputs/`;
- `runs/submission/jpeg_report_q99.json` ghi 5 scene, 300 JPEG,
  `299,158,611` bytes;
- production directories chair/bonsai không có trong current checkout;
- E7 có đầy đủ 15k research artifact, checkpoint recovery, validation render
  và diagnostics trong `runs/scene_opt_v3/research/chair/`.

Inventory của `outputs/`:

| Folder | Images | Bytes |
|---|---:|---:|
| HCM0421 | 60 | 126,028,286 |
| HCM0539 | 60 | 123,586,308 |
| HCM0540 | 60 | 125,350,931 |
| HCM0644 | 60 | 121,368,009 |
| HCM0674 | 60 | 120,174,076 |

Vì B0 official được ghi là 7 scene/335 MB, artifact q99 5 scene hiện tại không
phải bản archive đầy đủ của submission chính thức. Đây là **provenance/archive
gap**, không phải bằng chứng evaluator đã chấm thiếu scene.

### 5.3 Vấn đề thật theo scene

| Scene | Tình trạng output | Nguyên nhân có bằng chứng | Điều chưa biết |
|---|---|---|---|
| HCM0644 | B0 được giữ; local B0 cao nhất | chưa có candidate nào chứng minh cần đổi | không có official per-scene |
| HCM0539 | BTS nhìn chung ổn; AbsGrad mean/LPIPS local tốt hơn | gradient collision/detail allocation có khả năng; hard-tail hơi xấu | override có thắng hidden test riêng scene hay không |
| HCM0421 | AbsGrad có local gain mạnh nhất B1 | density signal cải thiện mean và LPIPS; compute tăng | chưa paired-confirm 30k |
| HCM0540 | B0; AA làm xấu mạnh | lỗi không phải aliasing đơn giản | root cause cụ thể chưa được visual/pose audit sâu |
| HCM0674 | BTS yếu nhất Stage A; ảnh mẫu có tearing/floaters ở background xa | sparse/far geometry và coverage là giả thuyết phù hợp; AA đã bị bác | chưa có paired seed/depth candidate |
| chair | 5/58 output catastrophic; 4 frame trong cụm 855–895 | projected-scale/occlusion instability, giant/elongated Gaussian; E7 làm tail lớn hơn | phương pháp nào sửa shape/allocation mà không bão hòa sensitivity |
| bonsai | 10/28 major/catastrophic | pose gap + thiếu geometry mặt bàn phản chiếu; SH4 không đủ | c2f AbsGrad SH4 chưa chạy |

### 5.4 Chair: chẩn đoán chi tiết

Quan sát:

- cụm fail gần lưới ghế có train image lân cận vẫn sharp;
- pose distance đơn thuần không dự báo severity;
- disagreement giữa source views dự báo severity;
- bảy ảnh train motion-blur chỉ giải thích softness nhẹ;
- E2 15k có 8,572 Gaussian có projected radius trên 128 px;
- xóa toàn bộ tail có thể sửa một frame nhưng phá view bình thường.

Kết luận nguyên nhân:

```text
occlusion + projected-scale thay đổi nhanh
        -> gradient của Gaussian lớn bị trộn/cancel
        -> primitive sai hình hoặc sai vị trí tồn tại
        -> ở một khoảng pose hẹp, primitive che phủ vùng ảnh lớn
        -> veil / blur / ghosting catastrophic
```

Không nên dùng global hard-pruning vì tail chứa cả Gaussian hữu ích. Cần phân
bổ lại topology theo contribution/perceptual importance hoặc geometry-aware
split.

### 5.5 Bonsai: chẩn đoán chi tiết

Quan sát:

- nearest-train normalized pose gap tương quan LPIPS `r=0.647`;
- mọi output có normalized gap ít nhất 2 đều không clear;
- pot thường ổn nhưng table/legs/carpet/foreground vỡ;
- frame train holdout 390 sharp nhưng vẫn fail;
- tabletop ROI frame 390 chỉ có `103/1,289` SfM observations: mật độ tương đối
  `0.523` so với toàn ảnh;
- E2 SH4 15k có 21,088 Gaussian radius trên 128 px;
- xóa tail làm mất khoảng 6.8 dB PSNR trung bình.

Kết luận:

```text
pose gap + tabletop thiếu SfM support + reflection view-dependent
        -> depth/normal mặt bàn không được xác định tốt
        -> Gaussian dùng alpha/appearance để giải thích nhiều view mâu thuẫn
        -> novel pose tạo mặt bàn translucent, legs/foreground warp
```

Higher SH chữa một phần hướng nhìn nhưng không tạo geometry bị thiếu. Candidate
hợp lý đầu tiên vẫn là coarse-to-fine + AbsGrad + SH4; sparse COLMAP depth chỉ
được thêm sau khi curriculum ổn định tail.

## 6. Toàn bộ deep-research `scene_opt_v3`

Holdout v3:

```text
algorithm=targeted_pose_guard2_v3
chair train/holdout=146/20
bonsai train/holdout=176/25
optimizer horizon=30000
ordinary research stop=15000
diagnostic scale threshold=0.1
diagnostic radius threshold=128 px
```

Không so số tuyệt đối của v3 với v1/v2 nếu holdout set khác.

### 6.1 Bảng kết quả chair/bonsai

| Run | Step | PSNR | SSIM | LPIPS | Score50 | Final G | VRAM | Time | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| bonsai E2 SH4 | 15k | 20.0981 | 0.70672 | 0.38927 | 57.6896 | 1,027,717 | 2.77 GB | 1605 s | incumbent; geometry vẫn fail |
| chair E2 local-Lap | 15k | 21.2579 | 0.68148 | 0.32756 | 60.0966 | 1,939,240 | 3.54 GB | 1560 s | incumbent v3 |
| chair E3 obs-scale | 15k | 21.7996 | 0.68931 | 0.31748 | **61.0597** | 1,973,807 | 3.61 GB | 1579 s | selector fix/control |
| chair E4 E3+AbsGrad | 15k | 20.8202 | 0.66638 | 0.32241 | 59.5872 | 3,260,453 | 5.85 GB | 2491 s | reject |
| chair E5 MCMC | 30k | 22.2456 | 0.68522 | 0.30793 | **61.5870** | 2,000,000 | 3.55 GB | 3992 s | reject lower-bound gate |
| chair E6 perceptual | 15k | 22.1637 | 0.69074 | 0.34227 | 60.3297 | 297,081 | 0.82 GB | 907 s | reject implementation |
| chair E7 corrected | 15k | 21.7500 | 0.68455 | 0.31155 | 61.1244 | 1,662,104 | 3.32 GB | 2185 s | reject quality/tail gate |

### 6.2 E3 — continuous observation mapping

Defect:

```text
stored COLMAP points2D scale ~= 1.5 * current projection
legacy inference = ceil(1.495.., 1.497..) = 2
sample position = stored / 2 thay vì stored / 1.5
```

Audit trên 474,038 observations:

| Mapping | Mean RGB MAE | Median | p90 |
|---|---:|---:|---:|
| correct 1.5 | 33.6835 | 32.5890 | 58.4306 |
| legacy 2.0 | 77.5610 | 66.5674 | 157.7571 |

Implementation preflight fit scale `(1.50001973, 1.50001935)`, giảm mapped
reprojection p95 từ `329.57 px` xuống `1.78 px`.

Paired E3 vs E2:

```text
ΔPSNR   +0.54174 dB
ΔSSIM   +0.007829
ΔLPIPS  -0.010079
ΔScore  +0.96308
```

Decision: giữ E3 làm corrected research control. Không promote production vì
full-data production dùng colors trong `points3D.bin`, không rebuild sparse
color bằng holdout images.

### 6.3 E4 — E3 + AbsGrad

Thông số:

```text
absgrad=true
grow_grad2d=0.0004
các thông số khác giữ E3
```

So với E3:

```text
ΔPSNR   -0.97943 dB
ΔSSIM   -0.022929
ΔLPIPS  +0.004924
ΔScore  -1.47249
Gaussian +65%
```

E4 tạo 3.26M Gaussian, tăng spurious-edge và chỉ thắng Score ở 2/20 frame.
Decision: reject. Bằng chứng cho thấy tăng density đồng đều theo AbsGrad không
giải quyết đúng allocation của chair.

### 6.4 E5 — 3DGS-MCMC full 30k

Thông số khóa:

```text
density_strategy=mcmc
cap=2,000,000
noise_lr=500,000
opacity_reg=0.001
scale_reg=0.01
relocation/refine_stop=25,000
rolling recovery=3,000
```

Raw comparison với E3-15k:

```text
ΔPSNR   +0.44595 dB
ΔSSIM   -0.004085
ΔLPIPS  -0.009556
ΔScore  +0.52727
```

Đây là 30k-vs-15k, không phải paired causal comparison. Predeclared lower-bound
gate yêu cầu `ΔScore>=+0.75`, LPIPS/hard/sentinel không regression. E5 không
qua gate; worst-decile edge distance xấu hơn và scale/radius opacity mass tăng.

Decision: reject, không chi thêm E3-30k.

### 6.5 E6 — Perceptual-GS core port

Thông số:

```text
sensitivity map: gamma=1.5, Sobel=0.05, avg pool=5x5, threshold=0.3
BCE weight=0.1
sensitivity LR=0.05
HD/MD sensitivity threshold=0.9/0.3
HD/MD contribution=25/10
HD/MD interval=1000/1500
opacity exponent=1.2
cap=2.1M
```

So với E3:

```text
ΔPSNR   +0.36410 dB
ΔSSIM   +0.001429
ΔLPIPS  +0.024783
ΔScore  -0.73000
```

Bootstrap đã ghi nhận:

```text
Score Δ CI95  [-1.454, -0.155]
LPIPS Δ CI95  [+0.0149, +0.0356]
PSNR Δ CI95   [+0.080, +0.711]
frame wins    7/20; regressions 13/20
hard wins     6/6; easy wins 0/7
```

E6 kết thúc chỉ 297k Gaussian, Laplacian variance mean `263.7` so với E3
`348.9`, không frame nào sharp hơn E3. Scale p99 `0.1224` so với `0.0350`;
radius p99 `163 px` so với `81 px`.

Root implementation mismatch:

- E6 override `_grow_gs` đã thay ADC 100-step;
- chỉ còn event HD/MD mỗi 1000/1500 step;
- perceptual candidate còn bị gradient gate lần hai;
- official Perceptual-GS giữ normal ADC và OR perceptual mask.

Decision: không resume E6 tới 30k. Reject E6 implementation, không coi đây là
falsification của Perceptual-GS.

### 6.6 E7 — corrected perceptual ADC

E7 đã chạy fresh tới 15k trên cùng holdout, seed và optimizer horizon với E3.
Khác E6:

1. giữ standard ADC mỗi 100 step;
2. HD/MD mask được OR vào clone/split, không gradient gate lần hai;
3. contribution lấy max trên toàn bộ internal-train views;
4. cap 2.1M ưu tiên baseline ADC;
5. giữ loss, SH3, E3 mapping và holdout;
6. không thêm scene-adaptive depth reinitialization.

Mechanism gate:

| Kiểm tra | Kết quả |
|---|---|
| Final population ít nhất 1.2M | pass: 1,662,104 |
| ADC diagnostics tồn tại | pass: 1,458,502 clone; 361,535 split |
| Perceptual diagnostics tồn tại | pass: 442,384 clone; 289,563 split |
| Cap 2.1M | không chạm cap |
| Run/artifact hữu hạn | pass |

Paired E7−E3:

| Metric | Delta | Bootstrap CI95 | Kết luận |
|---|---:|---:|---|
| Score50 | +0.0647 | [-0.3276, +0.3598] | không có gain rõ |
| PSNR | -0.0496 dB | [-0.2657, +0.1399] | không khác rõ |
| SSIM | -0.004755 | [-0.007828, -0.002238] | regression rõ |
| LPIPS | -0.005929 | [-0.010804, -0.000392] | improvement rõ |
| Spurious-edge | +0.002720 | [+0.001238, +0.004114] | regression rõ |
| Symmetric edge distance | +0.001614 | [-0.000046, +0.004274] | nghiêng xấu |

E7 thắng Score50 ở 13/20 frame, nhưng gain bị một regression nặng ở
`frame_000020` (`-2.8751`) triệt tiêu. Theo strata:

```text
easy   Score -0.1127, LPIPS -0.00285
medium Score +0.0526, LPIPS -0.00491
hard   Score +0.2859, LPIPS -0.01070
```

Sentinel:

| Frame | ΔScore50 | ΔLPIPS | Δ symmetric-edge |
|---|---:|---:|---:|
| 525 | -0.1571 | -0.00052 | -0.00038 |
| 870 | -0.1298 | -0.00173 | +0.02177 |
| 885 | +0.5310 | -0.02067 | +0.00965 |
| 260 | +0.5594 | -0.01822 | -0.00122 |

E7 dùng 0.842× số Gaussian cuối và 0.919× VRAM của E3, nhưng mất 1.384× thời
gian. Tail geometry xấu rõ:

| Diagnostic | E3 | E7 | Ratio |
|---|---:|---:|---:|
| projected radius p99 | 81 px | 124 px | 1.53× |
| radius >128 px count | 7,367 | 18,293 | 2.48× |
| radius-tail opacity fraction | 0.00375 | 0.01168 | 3.12× |
| scale3d p99 | 0.03499 | 0.07621 | 2.18× |
| scale >0.1 count | 4,022 | 11,858 | 2.95× |
| scale-tail opacity fraction | 0.00228 | 0.00810 | 3.56× |

Sensitivity không còn chọn lọc: mean learned sensitivity là `0.96193` và
`95.98%` Gaussian nằm trong high bin; medium chỉ `0.30%`. Full-view max
contribution kết hợp với broad sensitivity làm perceptual additions ưu tiên
quá rộng. Visual audit xác nhận E7 tạo một blur blob lớn mới ở frame 20, trong
khi veil ở 870/885 vẫn còn.

Decision: E7 pass mechanism gate nhưng fail mean-gain, SSIM, spurious-edge,
compute và scale/radius-tail gates. Không resume E7 tới 30k và không chạy
E3-30k confirmation.

## 7. Thông số và quyết định của mọi candidate

| ID | Scene/stage | Thay đổi chính | Horizon/stop | Trạng thái |
|---|---|---|---|---|
| `B0-reference` | 7 scene/reference | baseline SH3/ADC | 7k/7k | closed |
| `E1-density-absgrad-t04-v1` | 0421/0539 screen+prod | AbsGrad, grad 0.0004 | 7k; prod 30k | production override closed |
| `E1-density-scale005-v1` | 0421/0539 screen | grow scale 0.005 | 7k | reject |
| `E2-raster-aa-v1` | 0540/0674 screen | antialiased rasterizer | 7k | reject |
| `E2-loss-local-laplacian-v1` | chair/bonsai | patch31, floor0.5 | 7k; chair prod30k | chair incumbent |
| `E2-appearance-sh4-v1` | bonsai | SH4 | 7k; prod30k; research15k | bonsai incumbent |
| `E3-chair-observation-scale-v1` | chair/research | continuous observation mapping | 30k/15k | corrected control |
| `E4-chair-observation-scale-absgrad-v1` | chair/research | E3 + AbsGrad | 30k/15k | reject |
| `E5-chair-observation-scale-mcmc-v1` | chair/research | MCMC cap2M | 30k/30k | reject |
| `E6-chair-observation-scale-perceptual-v1` | chair/research | perceptual-only growth port | 30k/15k | reject implementation |
| `E7-chair-perceptual-adc-corrected-v1` | chair/research | standard ADC + perceptual OR | 30k/15k | reject at 15k; no resume |
| `E3-bonsai-c2f-absgrad-sh4-v1` | bonsai/research | 4x→1x by 5k + AbsGrad + SH4 | 30k/15k | reserved, not implemented |
| `E4-bonsai-c2f-absgrad-sh4-sparse-depth-v1` | bonsai/research | previous + COLMAP sparse depth | 30k/15k | reserved |

## 8. Các hướng nghiên cứu đã đánh giá

| Phương pháp | Phù hợp nguyên nhân | Bằng chứng/giới hạn | Quyết định hiện tại |
|---|---|---|---|
| AbsGS | gradient collision | tốt trên 0421/0539, fail chair E4 | không dùng độc lập cho chair |
| Spec-Gaussian c2f+AbsGrad | floater sớm, detail muộn | phù hợp bonsai, chưa local-run | bonsai candidate kế tiếp |
| MCMC | relocation, cap population | E5 30k không qua gate | reject chair hiện tại |
| Perceptual-GS | allocation theo contribution | E7 sửa ADC nhưng sensitivity bão hòa và tail tăng | reject local E7 port |
| Pixel-GS | pixel/depth-aware growth | exact participation không có trong gsplat API | defer, không dùng proxy dưới tên Pixel-GS |
| Sparse COLMAP depth | anchor geometry ít quan sát | table interior bonsai thiếu point | chỉ sau c2f pass |
| 2DGS/PGSR | planar geometry/depth/normal | renderer/loss contract lớn, NVS có trade-off | post-deadline fallback |
| GaussianShader/ASG/HMGS | reflection appearance | cần geometry/normal đủ tốt | không chạy trước geometry fix |
| Spectral-GS | shape-aware split/filter | đúng giant/elongated tail nhưng port sâu | research fallback |
| GaussianSpa | post-density sparsification | chỉ hợp lý sau khi có winner topology | chưa kết hợp |

## 9. Vì sao chưa chạy 70k

Evidence ủng hộ 30k hơn 7k trên HCM0181, nhưng không ủng hộ 70k:

- production chair/bonsai đã 30k;
- ADC dừng 15k, nên 30k→70k không tạo topology mới;
- đặt `max_steps=70000` làm thay đổi toàn bộ learning-rate trajectory, không
  phải continuation so sánh được với run 30k;
- E5 đã chạy đủ 30k vẫn không qua gate;
- E6 topology bị under-densified và đã frozen; thêm step không chữa cơ chế;
- visual catastrophic failure có tính pose-local, phù hợp topology/occlusion
  hơn global non-convergence.

Chỉ cân nhắc 5–10k low-LR polish sau một winner 30k nếu holdout curve
20k/25k/30k còn tăng, LPIPS/worst-tail không quay đầu và Gaussian topology ổn
định.

## 10. Trạng thái từng scene và quyết định tiếp theo

| Scene | Checkpoint/submission hiện hành | Research decision |
|---|---|---|
| HCM0644 | B0 | freeze; không có bằng chứng cần retrain |
| HCM0674 | B0 | AA reject; chỉ mở lại khi có sparse/far-geometry audit paired |
| HCM0540 | B0 | AA reject; chưa đủ bằng chứng cho candidate khác |
| HCM0539 | AbsGrad production | closed; không suy ra official per-scene win |
| HCM0421 | AbsGrad production | closed; local evidence dương nhưng compute gate fail |
| chair | local-Laplacian production | E7 reject; không có research winner được promote |
| bonsai | SH4 production | candidate c2f+AbsGrad+SH4 vẫn chưa triển khai/chạy |

Hệ quả quyết định E7:

- không promote E5, E6 hoặc E7;
- không chi E3-30k vì không candidate nào qua lower-bound gate;
- E3 vẫn là corrected research control, không phải production method;
- production fallback hợp lệ của chair vẫn là local-Laplacian đã nộp.

## 11. Rủi ro và nợ bằng chứng còn lại

1. Không có official per-scene/per-image metric.
2. LPIPS backbone và SSIM implementation của evaluator chưa được xác nhận.
3. Final 7-scene submission archive và hai production checkpoint auxiliary
   không nằm đầy đủ trong current checkout.
4. `runs/submission/jpeg_report_q99.json` hiện chỉ đại diện 5 BTS scene.
5. V3 veil flag đánh dấu toàn bộ holdout ở nhiều run; threshold này hữu ích
   cho provenance nhưng không phân biệt candidate tốt. Không dùng veil count
   đơn độc để chọn winner.
6. E5-vs-E3 là 30k-vs-15k lower-bound test, không phải paired causal estimate.
7. Bonsai chưa có deep candidate mới ngoài incumbent SH4.
8. E7 sensitivity bị bão hòa và scale/radius tail tăng; metric collapse flag
   không bắt được regression thị giác lớn ở frame 20.
9. HCM0540/HCM0674 có candidate bị bác nhưng chưa có root-cause audit sâu như
   chair/bonsai.

## 12. Source of truth sau khi hợp nhất

Đọc theo thứ tự:

1. `AGENTS.md` — data/output/evaluation/engineering contract.
2. Tài liệu này — lịch sử, chẩn đoán và toàn bộ kết quả.
3. `../specs/2026-07-27-chair-bonsai-deep-optimization.md` — execution
   authority và trạng thái closure E7 hiện tại.
4. `../plans/2026-07-27-chair-bonsai-deep-optimization.md` — checklist chạy.
5. `../../../README.md` — lệnh vận hành.

Các spec/plan ngày 2026-07-22 đến 2026-07-26 chỉ còn là provenance. Không dùng
chúng để mở lại candidate đã reject hoặc thay đổi baseline ID.

## 13. Chỉ mục provenance

### 13.1 Artifact chính

| Evidence | Vị trí |
|---|---|
| HCM0181 pilot checkpoints | `runs/HCM0181/` |
| Backend qualification | `runs/phase4/backend_qualification/` |
| Stage A/B1 artifacts | `runs/scene_opt_v1/` |
| Five-scene screen | `runs/scene_opt_v2/screen/` |
| Deep research E2–E7 | `runs/scene_opt_v3/research/` |
| B0 five-BTS JPEG report còn local | `runs/submission/jpeg_report_q99.json` |
| Initial optimization research | `docs/research/2026-07-19-baseline-optimization/` |
| Theory/local SfM audit | `../research/2026-07-27-chair-bonsai-candidate-validation.md` |

### 13.2 Mốc commit

| Mốc | Commit tiêu biểu |
|---|---|
| Data/camera/manifest/submission contracts | `6de7c00` đến `676b141` |
| Gaussian init, renderer, trainer, density | `640738c` đến `f6eb970` |
| Phase 4 holdout/profile/qualification | `f234fd5` đến `b1d67e9` |
| 30k dry run và accelerated backend | `aa55a21` đến `b2bdff0` |
| Full training/inference/JPEG/B0 closure | `558c500` đến `dd130c0` |
| Scene-specific experiment harness | `c78025c` đến `58ca887` |
| Stage A và Stage B1 | `ea6a6ce` đến `52dcc82` |
| AbsGrad production | `bf715e2` đến `a4f7c36` |
| Five-scene screen/hybrid/aux production | `f1ea7b4` đến `4fd0ae9` |
| Deep chair/bonsai research E2–E4 | `ada72f9` đến `ca44804` |
| MCMC E5 | `159cfd5` |
| Perceptual E6 | `8294883` |
| Corrected perceptual ADC E7 | `17c516b` |

### 13.3 Primary-method references đã dùng trong research

- [AbsGS](https://arxiv.org/abs/2404.10484)
- [Spec-Gaussian](https://proceedings.neurips.cc/paper_files/paper/2024/file/708e0d691a22212e1e373dc8779cbe53-Paper-Conference.pdf)
- [Pixel-GS](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)
- [3DGS-MCMC](https://proceedings.neurips.cc/paper_files/paper/2024/hash/93be245fce00a9bb2333c17ceae4b732-Abstract-Conference.html)
- [Perceptual-GS](https://arxiv.org/html/2506.12400)
- [DNGaussian](https://openaccess.thecvf.com/content/CVPR2024/html/Li_DNGaussian_Optimizing_Sparse-View_3D_Gaussian_Radiance_Fields_with_Global-Local_Depth_CVPR_2024_paper.html)
- [PGSR](https://arxiv.org/abs/2406.06521)
- [2DGS](https://arxiv.org/abs/2403.17888)
- [GaussianShader](https://openaccess.thecvf.com/content/CVPR2024/html/Jiang_GaussianShader_3D_Gaussian_Splatting_with_Shading_Functions_for_Reflective_Surfaces_CVPR_2024_paper.html)
