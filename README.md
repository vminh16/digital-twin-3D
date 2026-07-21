# BTS Digital Twin — Novel View Synthesis

Baseline 3D Gaussian Splatting theo từng scene, khởi tạo từ COLMAP sparse point
cloud và render tại các camera trong `test/test_poses.csv`. Mỗi scene có một mô
hình riêng; đây không phải mô hình tổng quát hóa chéo scene.

## Trạng thái baseline

Baseline `B0-submission-q99-v1` đã **CLOSED** trên đúng 7 scene:

```text
HCM0644 HCM0674 HCM0540 HCM0539 HCM0421 chair bonsai
```

| Kết quả evaluator chính thức | Giá trị |
|---|---:|
| Score | 70.98330 |
| PSNR | 24.611499 |
| SSIM | 80.4805 |
| LPIPS | 19.8195 |
| Matched scenes | 7/7 |

Submission dùng JPEG quality 99, 4:4:4, optimized, non-progressive; ZIP cuối
335 MB, dưới giới hạn 350 MB. Đây là kết quả từ evaluator chính thức, **không
phải benchmark local**; cấu hình nội bộ của LPIPS/SSIM và `PSNR_max` chưa được
xác nhận. Mọi thay đổi training, rendering hoặc codec sau mốc này phải dùng
baseline/candidate ID mới.

### Chi phí train baseline 30k

Các số dưới đây lấy từ `summary.json` của đúng bảy model tạo submission. Đây là
thời gian training tuần tự trên NVIDIA L4, không gồm chuẩn bị dữ liệu, inference
và đóng gói.

| Scene | Thời gian | Peak VRAM |
|---|---:|---:|
| HCM0644 | 2.29 giờ | 9.32 GB |
| HCM0674 | 2.24 giờ | 8.79 GB |
| HCM0540 | 2.30 giờ | 10.04 GB |
| HCM0539 | 2.45 giờ | 9.88 GB |
| HCM0421 | 1.99 giờ | 8.94 GB |
| chair | 0.99 giờ | 3.50 GB |
| bonsai | 0.86 giờ | 1.84 GB |
| **Tổng** | **13.13 L4 GPU-giờ** | **10.04 GB lớn nhất** |

## Tiến độ nghiên cứu

| Ngày | Mốc | Trạng thái | Kết quả ngắn gọn |
|---|---|---|---|
| 2026-07-19 | Đóng baseline `B0-submission-q99-v1` | Hoàn thành | Official Score 70.98330, đủ 7/7 scene |
| 2026-07-20 | C1 Phase A: `AbsGrad × revised_opacity`, 2 scene × 2 candidate × 7k | **PASS** | Khóa `C1-absgrad-t08-revopacity-v1` |
| 2026-07-21 | C1 Phase B: winner trên 4 scene 7k còn lại | **PASS** | 5/6 scene và 122/130 ảnh tốt hơn; mean delta Score50 `+0.823` |
| Hiện tại | C1 Phase C: fresh 30k trên HCM0181 có holdout | Sẵn sàng chạy | So sánh với B0 30k cùng holdout; một rolling recovery checkpoint |
| Cuối cùng | C1 Phase D: production 7 scene × 30k | Chờ Phase C | Tạo candidate submission mới, không sửa B0 đã đóng |

### C1 Phase A — AbsGrad và revised opacity

Mục tiêu C1 là giảm blur/missing edge tại antenna, dây và lattice BTS bằng
AbsGrad densification; `revised_opacity` được thử như cơ chế hạn chế haze,
halo và cạnh giả sau clone/split. Phase A dùng cùng seed 0, full resolution,
7,000 step và paired internal holdout với B0.

| Candidate | Mean delta Score50 trên 2 scene | Quyết định |
|---|---:|---|
| `C1-absgrad-t08-v1` | +0.614 | Không chọn |
| `C1-absgrad-t08-revopacity-v1` | **+1.197** | **Winner Phase A** |

Winner tăng Score50 trên cả HCM0421 (`+1.000`) và HCM1439 (`+1.394`), cải
thiện 35/36 ảnh validation, giảm HF-L1 khoảng 2.0–2.2% và missing-edge khoảng
2.4–3.3%. Spurious-edge tăng khoảng 3%, nên noise vẫn là metric giám sát ở
Phase B. Một outlier HCM0421 không tái tạo được bầu trời; Phase A vẫn pass nhưng
không được giả định rằng chỉ tăng từ 7k lên 30k sẽ tự sửa lỗi background này.

Bốn run candidate Phase A tốn tổng cộng **2,568.39 giây = 42.81 phút = 0.713
L4 GPU-giờ**. B0 controls được tái sử dụng, không train lại. Phase B dự kiến thêm
khoảng 45–54 phút training tuần tự cho bốn scene còn lại.

### C1 Phase B — robustness trên sáu scene

`C1-absgrad-t08-revopacity-v1` pass gate đã đăng ký trước: mean delta Score50
`+0.823`, median `+0.920`, LPIPS giảm `0.01060`, SSIM tăng `0.00842` và PSNR
tăng `0.244 dB`. Năm trong sáu scene và 122/130 ảnh validation cải thiện. Tổng
thời gian sáu run C1 là 68.62 phút, nhanh hơn 25.9% so với B0 paired; peak VRAM
lớn nhất 3.70 GB. Exact sign test theo scene có `p=0.21875`, nên đây là bằng
chứng screening chứ chưa phải tuyên bố tổng quát hóa.

Ba outlier lớn tại `hcm0031`, `HCM0421` và `HNI0131` đều liên quan đến vùng
bầu trời tối/đen. C1 giảm missing-edge aggregate nhưng spurious-edge tăng nhẹ;
Phase C giữ nguyên candidate để kiểm tra hiệu ứng tại 30k, không âm thầm thêm
loss hoặc xử lý background.

### C1 Phase C — confirmation 30k trên HCM0181

Phase C train fresh đúng `C1-absgrad-t08-revopacity-v1`, seed 0, factor 1,
30,000 step và internal holdout của B0 30k. Runner đọc backend đã được qualify,
chỉ giữ `runs/c1/phase_c/HCM0181/checkpoints/recovery.pt` mỗi 3,000 step và tự
resume khi run chưa hoàn tất. B0 lịch sử là Adam/FP32 và backend qualification
hiện tại cũng chọn Adam/FP32; runner fail trước training nếu backend này thay
đổi, thay vì tạo một so sánh không paired. Không resume checkpoint 7k.

B0 30k reference đã bị bỏ khỏi HEAD để giảm dung lượng worktree nhưng vẫn nằm
trong lịch sử tại `411c8de`. Trên một checkout sạch, materialize artifact này
một lần rồi chạy Phase C:

```bash
git archive 411c8de runs/phase4/dry_run_30k/HCM0181 | tar -x

python src/bts_nvs/training/run_c1_confirmation.py \
  --repo_root "$PWD" \
  --scenes_root "$PWD/data/bts_scenes" \
  --manifests_root "$PWD/runs/manifests" \
  --backend_root "$PWD/runs/phase4/backend_qualification" \
  --baseline_root "$PWD/runs/phase4/dry_run_30k/HCM0181" \
  --phase_b_root "$PWD/runs/c1/phase_b" \
  --output_root "$PWD/runs/c1/phase_c"
```

Chạy lại cùng lệnh sẽ resume từ rolling recovery nếu có; một run có report hoàn
chỉnh chỉ được tái sử dụng sau khi checkpoint, config, manifest, report và hash
của từng validation render khớp nhau. Phase C chỉ pass khi Score50 tăng, LPIPS
không tăng, missing-edge và spurious-edge không đồng thời xấu đi, integrity pass
và peak VRAM dưới đúng 23 GB. `phase_c_decision.json` lưu provenance; ledger lưu
trạng thái hoặc lỗi. Runner không tự khởi động Phase D. Mốc chi phí dự kiến từ
B0 là khoảng 2.9 L4 GPU-giờ.

## Cài đặt không dùng Docker

Python 3.10–3.12 và NVIDIA CUDA/PyTorch tương thích GPU được yêu cầu. Trên VM
headless, repo dùng `opencv-python-headless`, không cần `libGL` hay UI.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Cài PyTorch CUDA phù hợp với driver trước, sau đó:
pip install -r requirements.txt
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python tests/smoke_test.py
```

Trên Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
```

## Cấu trúc dữ liệu

```text
data/
├── bts_scenes/<scene_id>/       # canonical BTS pool
└── auxiliary/
    ├── chair/
    └── bonsai/

<scene_id>/
├── train/images/
├── train/sparse/0/{cameras.bin,images.bin,points3D.bin}
└── test/test_poses.csv
```

Không chạy lại COLMAP. Pose test, intrinsics, tên và resolution đầu ra lấy từ
CSV; COLMAP registration chỉ cung cấp kiểm tra pose/calibration và distortion.

## 1. Chuẩn bị artifact

Canonical pool:

```bash
bash scripts/prepare_scene_manifests.sh
```

Artifact cho `chair` và `bonsai` được tạo tự động bởi script train ở bước dưới.
Hai scene này dùng COLMAP `SIMPLE_PINHOLE`, được ánh xạ thành pinhole với
`fx = fy` và distortion bằng zero.

## 2. Qualification backend

Một checkout mới phải có qualification artifact trước production training:

```bash
bash scripts/qualify_training_backend.sh
```

Runner production đọc backend/precision đã được chấp nhận từ
`runs/phase4/backend_qualification`; không tự chọn lại cho từng scene.

## 3. Train baseline

Năm BTS scene cần nộp:

```bash
bash scripts/train_scene_cohort.sh \
  --scene_ids HCM0644 HCM0674 HCM0540 HCM0539 HCM0421
```

Hai scene còn thiếu:

```bash
bash scripts/run_submission_auxiliary_training.sh chair bonsai
```

Script auxiliary dùng cùng baseline full-resolution, 30k steps, seed 0,
backend qualification và rolling `recovery.pt`. Nếu checkpoint recovery hợp lệ
đã tồn tại, script resume; một run hoàn chỉnh sẽ không train lại 30k steps.

## 4. Inference đúng định dạng CSV

Inference dùng `test_image_names` làm nguồn sự thật và giữ nguyên extension/case:

- `.jpg`/`.jpeg` (không phân biệt hoa thường): payload JPEG;
- `.png`: payload PNG;
- width/height: đúng từng dòng CSV;
- `test_output_names` trong manifest schema v1 chỉ là field legacy để bảo toàn
  hash của checkpoint cũ, không dùng cho submission mới.

Mặc định JPEG là quality 98, 4:4:4, optimize và non-progressive. Có thể đổi bằng
`--jpeg_quality 1..100`; không khuyến nghị Q100 vì 7 scene sẽ vượt 350 MB.

Render 5 BTS scene vào một output root mới:

```bash
BTS_OUTPUT_ROOT="$PWD/outputs_bts" \
BTS_INFERENCE_REPORT="$PWD/runs/phase4/inference_bts.json" \
bash scripts/render_scene_cohort.sh \
  --jpeg_quality 98 \
  --scene_ids HCM0644 HCM0674 HCM0540 HCM0539 HCM0421
```

Render hai auxiliary scene. `--allow_noncanonical_scenes` chỉ mở khóa các scene
được nêu rõ; `--skip_prepare` tránh wrapper ép kiểm tra canonical pool 18 scene:

```bash
BTS_SCENES_ROOT="$PWD/data/auxiliary" \
BTS_MANIFESTS_ROOT="$PWD/runs/manifests_auxiliary" \
BTS_FULL_ROOT="$PWD/runs/phase4/auxiliary_training" \
BTS_OUTPUT_ROOT="$PWD/outputs_auxiliary" \
BTS_INFERENCE_REPORT="$PWD/runs/phase4/inference_auxiliary.json" \
bash scripts/render_scene_cohort.sh \
  --skip_prepare \
  --allow_noncanonical_scenes \
  --jpeg_quality 98 \
  --scene_ids chair bonsai
```

Mỗi inference run tự validate toàn bộ output subset trước khi atomic rename.
Output root và report phải chưa tồn tại để tránh trộn artifact cũ.

## 5. Ghép và nén submission

Việc ghép chỉ di chuyển các folder scene, không encode ảnh lần hai:

```bash
mkdir -p submission_final
cp -a outputs_bts/. submission_final/
cp -a outputs_auxiliary/. submission_final/
test "$(find submission_final -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 7
(cd submission_final && zip -r -9 ../submission_final.zip .)
du -h submission_final.zip
```

ZIP phải chứa trực tiếp 7 folder scene, không có tầng `submission_final/` bên
ngoài. Mục tiêu an toàn là khoảng 310–330 MB; giới hạn cứng là 350 MB.

## 6. Local benchmark

Chỉ benchmark khi có reference RGB hợp lệ và không phải official held-out test:

```bash
python -m bts_nvs.evaluation.run_benchmark \
  --outputs_root outputs_bts \
  --reference_root /path/to/references \
  --scenes_root data/bts_scenes \
  --manifests_root runs/manifests \
  --scene_ids HCM0644 HCM0674 HCM0540 HCM0539 HCM0421 \
  --psnr_max 40 --lpips_backbone alex --device cuda \
  --report_path runs/phase4/local_benchmark.json
```

Không dùng test RGB cho training, tuning hoặc chọn checkpoint.
