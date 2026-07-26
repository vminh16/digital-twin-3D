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
phải benchmark local**. Từ bốn số evaluator có thể giải ngược
`PSNR_max = 49.99983` sau sai số làm tròn, vì vậy repo khóa `PSNR_max=50`.
Backbone LPIPS và chi tiết kernel/aggregation SSIM của evaluator chính thức vẫn
chưa được xác nhận. Mọi thay đổi training, rendering hoặc codec sau mốc này
phải dùng baseline/candidate ID mới.

## Trạng thái tối ưu hóa hiện tại

Chương trình scene-specific đã hoàn thành Module 1–3: validation/detail audit,
candidate registry và generic stage-first runner. NVIDIA L4 smoke gates đã
được người dùng xác nhận pass. Stage A đã hoàn tất trên đủ bảy scene với
`174/174` validation render hợp lệ, không checkpoint, không NaN/Inf, tổng
runtime khoảng 95,6 phút trên NVIDIA L4. Trung bình scene-balanced của
`B0-reference` 7k là Score50 `64.654`, PSNR `21.050`, SSIM `0.7267`, LPIPS
`0.2444`.

`B0-submission-q99-v1` 30k là production baseline chung đã đạt 70.98330 điểm;
`B0-reference` 7k là paired internal-holdout authority để chọn candidate. Hai
artifact có vai trò khác nhau và không thay thế nhau. Stage B1 đã hoàn tất:
AbsGrad t04 cải thiện local Score50 trên HCM0539 và HCM0421 nhưng vượt time
gate 1.25x. Do deadline, hai scene này được duyệt theo hướng compute-first MVP:
train production 30k trực tiếp, không mô tả là paired-confirmed winner. Xem
[documentation status](docs/README.md) và
[five-scene MVP authority](docs/superpowers/specs/2026-07-26-five-scene-mvp.md).
Vòng hiện tại giữ HCM0644 ở B0; screen antialiasing cho HCM0674/HCM0540,
local sharpness weighting cho chair/bonsai và SH4 riêng cho bonsai.

Production 30k hiện đã hoàn tất cho bốn scene override. Candidate submission
được khóa là `MVP-hybrid-4scene-q99-v1`: HCM0421/HCM0539 dùng AbsGrad,
chair dùng local-Laplacian, bonsai dùng SH4; HCM0644/HCM0674/HCM0540 giữ
nguyên folder Q99 từ baseline đã đóng.

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
bash scripts/prepare_phase4_artifacts.sh
```

Artifact cho `chair` và `bonsai` được tạo tự động bởi script train ở bước dưới.
Hai scene này dùng COLMAP `SIMPLE_PINHOLE`, được ánh xạ thành pinhole với
`fx = fy` và distortion bằng zero.

## 2. Qualification backend

Một checkout mới phải có qualification artifact trước production training:

```bash
bash scripts/run_phase4_backend_qualification.sh
```

Runner production đọc backend/precision đã được chấp nhận từ
`runs/phase4/backend_qualification`; không tự chọn lại cho từng scene.

## 3. Train baseline

Năm BTS scene cần nộp:

```bash
bash scripts/run_phase4_full_training.sh \
  --scene_ids HCM0644 HCM0674 HCM0540 HCM0539 HCM0421
```

Hai scene còn thiếu:

```bash
bash scripts/run_submission_auxiliary_training.sh chair bonsai
```

Script auxiliary dùng cùng baseline full-resolution, 30k steps, seed 0,
backend qualification và rolling `recovery.pt`. Nếu checkpoint recovery hợp lệ
đã tồn tại, script resume; một run hoàn chỉnh sẽ không train lại 30k steps.

### Train AbsGrad compute-first MVP

Script sau chỉ hỗ trợ `HCM0421` và `HCM0539`. Trước mỗi production run, nó
validate lại B0-reference và AbsGrad screen 7k, gồm config
`internal_holdout=true`, manifest/holdout hash, validation renders và reports.
Production 30k luôn dùng `internal_holdout=false` để học toàn bộ train images:

```bash
bash scripts/run_absgrad_mvp_production.sh
```

Có thể chạy hoặc resume riêng một scene:

```bash
bash scripts/run_absgrad_mvp_production.sh HCM0421
bash scripts/run_absgrad_mvp_production.sh HCM0539
```

Output mặc định:

```text
runs/scene_opt_v1/production_mvp/scenes/<scene_id>/
```

Script tự bỏ qua run hoàn chỉnh, resume duy nhất từ rolling
`checkpoints/recovery.pt`, và từ chối thư mục partial không có recovery hợp lệ.

## 4. Inference đúng định dạng CSV

Inference dùng `test_image_names` làm nguồn sự thật và giữ nguyên extension/case:

- `.jpg`/`.jpeg` (không phân biệt hoa thường): payload JPEG;
- `.png`: payload PNG;
- width/height: đúng từng dòng CSV;
- `test_output_names` trong manifest schema v1 chỉ là field legacy để bảo toàn
  hash của checkpoint cũ, không dùng cho submission mới.

Mặc định JPEG là quality 98, 4:4:4, optimize và non-progressive. Có thể đổi bằng
`--jpeg_quality 1..100`; không khuyến nghị Q100 vì 7 scene sẽ vượt 350 MB.

### Rerender chính xác bốn scene MVP

Tái sử dụng `scripts/run_phase4_inference.sh` cho từng scene. Mỗi output root
phải mới vì inference chỉ publish bằng atomic rename và không ghi nối vào một
root đã tồn tại.

HCM0421:

```bash
BTS_SCENES_ROOT="$PWD/data/bts_scenes" \
BTS_MANIFESTS_ROOT="$PWD/runs/manifests" \
BTS_OUTPUT_ROOT="$PWD/outputs/HCM0421_mvp_q99" \
BTS_INFERENCE_REPORT="$PWD/runs/scene_opt_v1/inference_HCM0421_mvp_q99.json" \
bash scripts/run_phase4_inference.sh \
  --skip_prepare \
  --jpeg_quality 99 \
  --scene_ids HCM0421 \
  --run_dir "HCM0421=$PWD/runs/scene_opt_v1/production_mvp/scenes/HCM0421"
```

HCM0539:

```bash
BTS_SCENES_ROOT="$PWD/data/bts_scenes" \
BTS_MANIFESTS_ROOT="$PWD/runs/manifests" \
BTS_OUTPUT_ROOT="$PWD/outputs/HCM0539_mvp_q99" \
BTS_INFERENCE_REPORT="$PWD/runs/scene_opt_v1/inference_HCM0539_mvp_q99.json" \
bash scripts/run_phase4_inference.sh \
  --skip_prepare \
  --jpeg_quality 99 \
  --scene_ids HCM0539 \
  --run_dir "HCM0539=$PWD/runs/scene_opt_v1/production_mvp/scenes/HCM0539"
```

chair:

```bash
BTS_SCENES_ROOT="$PWD/data/auxiliary" \
BTS_MANIFESTS_ROOT="$PWD/runs/manifests_auxiliary" \
BTS_FULL_ROOT="$PWD/runs/phase4/auxiliary_training" \
BTS_OUTPUT_ROOT="$PWD/outputs/chair_mvp_q99" \
BTS_INFERENCE_REPORT="$PWD/runs/scene_opt_v2/inference_chair_mvp_q99.json" \
bash scripts/run_phase4_inference.sh \
  --skip_prepare \
  --allow_noncanonical_scenes \
  --jpeg_quality 99 \
  --scene_ids chair \
  --run_dir "chair=$PWD/runs/scene_opt_v2/production_mvp/scenes/chair"
```

bonsai:

```bash
BTS_SCENES_ROOT="$PWD/data/auxiliary" \
BTS_MANIFESTS_ROOT="$PWD/runs/manifests_auxiliary" \
BTS_FULL_ROOT="$PWD/runs/phase4/auxiliary_training" \
BTS_OUTPUT_ROOT="$PWD/outputs/bonsai_mvp_q99" \
BTS_INFERENCE_REPORT="$PWD/runs/scene_opt_v2/inference_bonsai_mvp_q99.json" \
bash scripts/run_phase4_inference.sh \
  --skip_prepare \
  --allow_noncanonical_scenes \
  --jpeg_quality 99 \
  --scene_ids bonsai \
  --run_dir "bonsai=$PWD/runs/scene_opt_v2/production_mvp/scenes/bonsai"
```

Ánh xạ checkpoint:

| Nhóm | Scene | Candidate | Run directory |
|---|---|---|---|
| scene-opt v1 | HCM0421, HCM0539 | `E1-density-absgrad-t04-v1` | `runs/scene_opt_v1/production_mvp/scenes/<scene>` |
| scene-opt v2 | chair | `E2-loss-local-laplacian-v1` | `runs/scene_opt_v2/production_mvp/scenes/chair` |
| scene-opt v2 | bonsai | `E2-appearance-sh4-v1` | `runs/scene_opt_v2/production_mvp/scenes/bonsai` |

Output scene folders:

```text
outputs/HCM0421_mvp_q99/HCM0421/
outputs/HCM0539_mvp_q99/HCM0539/
outputs/chair_mvp_q99/chair/
outputs/bonsai_mvp_q99/bonsai/
```

Mỗi output root và report tương ứng phải chưa tồn tại. Mỗi invocation load đúng
rolling checkpoint 30k, đọc world-to-camera pose, intrinsics, width, height và
case-sensitive `image_name` từ manifest được tạo từ
`<scene>/test/test_poses.csv`, rồi tự validate toàn bộ subset trước atomic
rename. Không dùng internal holdout, train camera hay `test_output_names` để
render test.

Render 5 BTS scene vào một output root mới:

```bash
BTS_OUTPUT_ROOT="$PWD/outputs_bts" \
BTS_INFERENCE_REPORT="$PWD/runs/phase4/inference_bts.json" \
bash scripts/run_phase4_inference.sh \
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
bash scripts/run_phase4_inference.sh \
  --skip_prepare \
  --allow_noncanonical_scenes \
  --jpeg_quality 98 \
  --scene_ids chair bonsai
```

Mỗi inference run tự validate toàn bộ output subset trước khi atomic rename.
Output root và report phải chưa tồn tại để tránh trộn artifact cũ.

## 5. Ghép và nén hybrid submission

Đặt `B0_SUBMISSION_ROOT` tới thư mục giải nén của
`B0-submission-q99-v1`. Tạo staging mới từ đúng ba B0 fallback và bốn output
MVP; không encode ảnh lần hai:

```bash
B0_SUBMISSION_ROOT="/path/to/extracted/B0-submission-q99-v1"
FINAL_ROOT="$PWD/submission_mvp_hybrid_4scene_q99_v1"

test ! -e "$FINAL_ROOT"
mkdir "$FINAL_ROOT"
cp -a "$B0_SUBMISSION_ROOT/HCM0644" "$FINAL_ROOT/"
cp -a "$B0_SUBMISSION_ROOT/HCM0674" "$FINAL_ROOT/"
cp -a "$B0_SUBMISSION_ROOT/HCM0540" "$FINAL_ROOT/"
cp -a outputs/HCM0421_mvp_q99/HCM0421 "$FINAL_ROOT/"
cp -a outputs/HCM0539_mvp_q99/HCM0539 "$FINAL_ROOT/"
cp -a outputs/chair_mvp_q99/chair "$FINAL_ROOT/"
cp -a outputs/bonsai_mvp_q99/bonsai "$FINAL_ROOT/"

test "$(find "$FINAL_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 7
(cd "$FINAL_ROOT" && zip -r -9 ../MVP-hybrid-4scene-q99-v1.zip .)
du -h MVP-hybrid-4scene-q99-v1.zip
```

ZIP phải chứa trực tiếp đúng 7 folder scene, không có tầng staging bên ngoài.
Kiểm tra kích thước cuối không vượt giới hạn cứng 350 MB. Không suy ra rằng
hybrid vẫn đúng 335 MB; bốn render mới có thể làm kích thước thay đổi.

## 6. Local benchmark

Chỉ benchmark khi có reference RGB hợp lệ và không phải official held-out test:

```bash
python -m bts_nvs.evaluation.run_benchmark \
  --outputs_root outputs_bts \
  --reference_root /path/to/references \
  --scenes_root data/bts_scenes \
  --manifests_root runs/manifests \
  --scene_ids HCM0644 HCM0674 HCM0540 HCM0539 HCM0421 \
  --psnr_max 50 --lpips_backbone alex --device cuda \
  --report_path runs/phase4/local_benchmark.json
```

Không dùng test RGB cho training, tuning hoặc chọn checkpoint.
