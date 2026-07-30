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

Submission `MVP-hybrid-4scene-q99-v1` đã nộp và **CLOSED**:

| Kết quả evaluator chính thức | Giá trị |
|---|---:|
| Score | 71.2124 |
| PSNR | 24.629191 |
| SSIM | 80.7208 |
| LPIPS | 19.4533 |
| Matched scenes | 7/7 |

Nó dùng AbsGrad cho HCM0421/HCM0539, local-Laplacian cho chair, SH4 cho
bonsai và giữ B0 cho HCM0644/HCM0674/HCM0540. Đây là aggregate hidden-test;
không có per-scene metric để dùng làm tín hiệu tuning.

Plan active hiện chỉ tối ưu sâu `chair` và `bonsai`. Năm scene còn lại bị
freeze. Authority và execution plan:

- [consolidated project diagnosis and experiment history](docs/superpowers/history/2026-07-29-project-diagnosis-and-experiment-report.md)
- [chair/bonsai deep-optimization authority](docs/superpowers/specs/2026-07-27-chair-bonsai-deep-optimization.md)
- [active execution plan](docs/superpowers/plans/2026-07-27-chair-bonsai-deep-optimization.md)

Research thông thường giữ lịch optimizer 30k nhưng dừng ở 15k, dùng internal
holdout và không lưu checkpoint:

```bash
bash scripts/run_chair_bonsai_research.sh <chair|bonsai> <candidate-id>
```

Chair MCMC là ngoại lệ full-horizon vì relocation còn chạy tới 25k. Wrapper
khóa candidate, cap 2M và rolling recovery 3k:

```bash
bash scripts/run_chair_mcmc_research.sh
```

Chair E6 bị reject vì implementation thay standard ADC và chỉ kết thúc với
khoảng 297k Gaussian. E7 đã sửa ADC và chạy fresh tới 15k, nhưng cũng bị
reject: Score50 chỉ `+0.0647` so với E3, SSIM/spurious-edge và scale/radius
tail xấu hơn. Không resume E6/E7 và không chi E3-30k confirmation.

E8 spectral split đã bị paired gate 15k bác bỏ: Score50 `-0.1372`, SSIM
`-0.00405` và hard Score50 `-0.2238` so với E3-15k. Không resume E8.

Do compute không còn là ràng buộc quyết định, E5 MCMC được mở lại để kiểm định
causal bằng một E3 control cùng horizon. Chạy fresh E3 30k, internal holdout
và rolling recovery bằng:

```bash
bash scripts/run_chair_e3_30k_control.sh
```

Đây là research control, không phải MVP production. E5 vẫn chưa được phép
production cho tới khi paired E5-30k so với E3-30k được chốt.

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
