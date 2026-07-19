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
baseline/candidate ID mới. Hướng cải tiến tiếp theo chưa được chốt.

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
