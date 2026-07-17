# Specification: Phase 4 — Qualification, Full Training and Benchmark

## 1. Mục tiêu

Phase 4 biến baseline 3DGS của Phase 3 thành một quy trình đánh giá và huấn
luyện đa-scene có thể vận hành an toàn trên **một NVIDIA L4 24 GB**.

Phase này phải trả lời lần lượt bốn câu hỏi:

1. Baseline có tổng quát hóa sang camera chưa tham gia optimization không?
2. Cấu hình nào cho chất lượng tốt nhất trong giới hạn VRAM, thời gian và disk?
3. Cấu hình đã khóa có thể huấn luyện tuần tự toàn bộ scene một cách tái lập không?
4. Checkpoint cuối có render đủ và đúng mọi test pose theo output contract không?

Luồng phụ thuộc:

```text
Scene inventory
→ Leakage-controlled internal holdout
→ Input/GPU profiling
→ Bounded hyperparameter qualification
→ One-scene 30k qualification
→ Freeze B0
→ Sequential full training
→ Test rendering and final validation
```

Không được bắt đầu một phiên khi acceptance criteria của phiên trước chưa pass.
Mỗi phiên có commit riêng và bắt đầu bằng red review các artifact/code mà phiên
đó phụ thuộc.

## 2. Phạm vi và non-goals

### 2.1. Trong phạm vi

- Audit toàn bộ scene trước khi phân bổ compute.
- Tạo internal holdout deterministic từ ảnh train vật lý.
- Ngăn RGB validation đi vào loss, model selection trong cùng run và sparse
  color initialization.
- Đo PSNR, SSIM và LPIPS trên internal holdout.
- Profile và loại bỏ CPU preprocessing bottleneck mà không đổi toán học B0.
- Thử một tập hyperparameter nhỏ, định trước và có control.
- Chạy qualification 30,000 steps trên một scene đại diện.
- Khóa config, dependency và artifact schema trước multi-scene training.
- Huấn luyện tuần tự từng scene trên một L4; không train nhiều scene đồng thời.
- Render đúng test pose, distortion domain, filename và resolution.
- Tổng hợp benchmark, resource report và submission validation.

### 2.2. Ngoài phạm vi

- Distributed training trong một scene.
- Multi-GPU orchestration.
- Pose optimization, appearance embedding, depth loss, LPIPS training loss.
- AMP, MCMC, AbsGS hoặc đổi representation trong baseline đầu tiên.
- Nâng `gsplat` trong lúc qualification; Phase 4 giữ `gsplat==1.4.0`.
- Dùng official test pose, test filename hoặc test image để chọn split,
  hyperparameter, checkpoint hay early stopping.
- Chạy lại feature extraction, matching hoặc SfM/COLMAP.
- Tuyên bố metric nội bộ tương đương leaderboard khi grading harness chưa khóa
  `PSNR_max`, LPIPS backbone và SSIM implementation.

## 3. Cơ sở nghiên cứu và quyết định baseline

Baseline tiếp tục dùng cấu hình gốc của 3D Gaussian Splatting:

- 30,000 optimization steps;
- position LR từ `1.6e-4` xuống `1.6e-6` trong 30,000 steps;
- densification từ step 500 đến trước step 15,000, interval 100;
- opacity reset interval 3,000;
- densification gradient threshold `2e-4`;
- DSSIM weight `0.2`;
- SH degree tối đa 3.

Các giá trị này khớp implementation chính thức của 3DGS. Repository gốc cũng
khuyến nghị ảnh có chiều rộng khoảng 1–1.6K; ảnh HCM0181 rộng 1320 pixel nằm
trong miền này. Nguồn: [3DGS paper and official code][3dgs-paper],
[official training arguments][3dgs-code].

Mip-NeRF 360 dùng một ảnh trên tám làm test set, tương đương khoảng 12.5% và
phủ đều trajectory. Nerfstudio cung cấp cả fraction 90/10 và interval 8. Đây là
tham chiếu cho kích thước holdout, không phải thuật toán split của dự án vì
filename/frame interval có thể đặt hai camera gần trùng vào hai split khác nhau.
Nguồn: [Mip-NeRF 360][mipnerf360], [Nerfstudio dataparser][ns-parser].

3DGS/gsplat tối ưu trên full image với batch một camera. `gsplat` packed mode
được giữ lại vì mục tiêu là hiệu quả bộ nhớ và rasterization, không phải lấp đầy
VRAM. Nerfstudio cũng cache ảnh đã preprocessing cho Splatfacto thay vì decode
và undistort lại ở mỗi step. Nguồn: [gsplat][gsplat],
[Nerfstudio Splatfacto][splatfacto].

Trong tài liệu này, “optimal” nghĩa là tốt nhất trong candidate set đã định
trước dưới giới hạn một L4, không có nghĩa là hyperparameter tối ưu phổ quát.

## 4. Threat model chống leakage

### 4.1. Dữ liệu cấm

Official `test/test_poses.csv`, test names và mọi test RGB không được đọc bởi:

- holdout selector;
- calibration/confirmation cohort selector;
- hyperparameter search;
- early stopping hoặc checkpoint selection;
- training evaluator.

Test camera chỉ được đọc sau khi baseline đã khóa, trong test-rendering stage.

### 4.2. Internal validation RGB

Một ảnh internal validation không được:

- xuất hiện trong train sampler;
- đóng góp vào loss hoặc gradient;
- dùng để tính sparse point color;
- dùng làm train preview;
- dùng để chọn checkpoint trong chính training run.

Metric validation chỉ chạy tại step cố định đã khai báo trước: 7,000 cho bounded
qualification và 30,000 cho full qualification. Không early-stop theo validation.

### 4.3. COLMAP geometry disclosure

Sparse point coordinates và camera poses là geometry do dataset cung cấp. COLMAP
reconstruction có thể đã sử dụng observations từ toàn bộ physical train set.
Phase 4 không chạy lại SfM, do đó internal holdout được mô tả chính xác là:

> photometric holdout không leakage RGB, có điều kiện trên shared provided SfM
> geometry.

Không được tuyên bố geometry hoàn toàn độc lập với validation. Để loại leakage
màu, point color phải được tính lại chỉ từ observations thuộc internal train.

## 5. Public contracts mới

### 5.1. Scene inventory

```python
@dataclass(frozen=True)
class SceneInventory:
    scene_id: str
    train_image_count: int
    test_pose_count: int
    sparse_point_count: int
    trajectory_nn_p90: float
    distortion_abs_max: float
    native_widths: tuple[int, ...]
    native_heights: tuple[int, ...]

def build_scene_inventory(manifest: SceneManifest) -> SceneInventory: ...
```

Inventory chỉ dùng train metadata và sparse geometry. `test_pose_count` được
phép dùng cho capacity/output completeness, nhưng test poses/names không được
đưa vào feature chọn cohort hoặc split.

### 5.2. Holdout split

```python
@dataclass(frozen=True)
class HoldoutSplit:
    schema_version: int
    scene_id: str
    manifest_sha256: str
    algorithm: str
    train_image_names: tuple[str, ...]
    validation_image_names: tuple[str, ...]
    guard_image_names: tuple[str, ...]

def build_pose_holdout(manifest: SceneManifest) -> HoldoutSplit: ...
def save_holdout_split(split: HoldoutSplit, path: Path) -> None: ...
def load_holdout_split(path: Path, manifest: SceneManifest) -> HoldoutSplit: ...
```

Schema đầu tiên là `1`; algorithm identifier là
`"pose_fps_guard2_v1"`. JSON dùng UTF-8, `allow_nan=False`, key deterministic và
atomic write. Manifest hash mismatch hoặc unsupported schema là fatal.

### 5.3. Subset dataset

```python
SceneDataset(
    manifest,
    scene_root,
    *,
    image_names: tuple[str, ...] | None = None,
    undistort: bool = True,
    resize: tuple[int, int] | None = None,
    cache_images: bool = False,
)
```

`image_names=None` giữ behavior Phase 3. Khi có names, dataset phải bảo toàn
thứ tự đó, reject duplicate/unknown name và không expose sample ngoài subset.

### 5.4. Qualification report

```python
@dataclass(frozen=True)
class QualificationReport:
    schema_version: int
    scene_id: str
    config_sha256: str
    holdout_sha256: str
    step: int
    image_count: int
    psnr_db_mean: float
    ssim_mean: float
    lpips_mean: float
    train_psnr_db_mean: float
    peak_vram_mb: float
    peak_gaussians: int
    total_time_seconds: float
```

Metric được tính per-image rồi mới scene-average. Raw PSNR/SSIM/LPIPS được dùng
cho qualification; composite score không dùng để chọn config khi `PSNR_max` và
grading harness chưa được xác nhận.

## 6. Thuật toán internal holdout

### 6.1. Pose distance

Với normalized camera center `C_i` và optical axis thế giới
`v_i = R_cw @ [0, 0, 1]`, định nghĩa:

```text
theta(i,j) = arccos(clamp(v_i · v_j, -1, 1))
D(i,j) = ||C_i - C_j||_2 + 0.25 * theta(i,j) / pi
```

Camera centers đã ở normalized world domain, nên translation term là
dimensionless và nhất quán giữa scene. Tie luôn break bằng exact image name.

### 6.2. Validation selection

1. `target = max(8, floor(N / 8 + 0.5))`; quy tắc half-up này tránh khác biệt
   do cách làm tròn mặc định của ngôn ngữ.
2. Chọn seed đầu là camera gần mean normalized center nhất.
3. Lặp farthest-point sampling: chọn camera có khoảng cách nhỏ nhất tới tập đã
   chọn lớn nhất.
4. Với mỗi validation camera, đưa hai camera non-validation gần nhất theo `D`
   vào guard set.
5. Internal train là phần còn lại.
6. Nếu train còn ít hơn `max(120, ceil(0.70 * N))`, bỏ validation seed được
   chọn cuối cùng, tính lại guard và lặp.
7. Nếu không thể giữ ít nhất 8 validation images thì scene fail qualification.

Guard images không train và không tính validation metric. Chúng chỉ tạo khoảng
cách pose để giảm near-duplicate leakage.

### 6.3. Split invariants

- Train, validation và guard pairwise disjoint.
- Hợp ba tập bằng exact physical train image names trong manifest.
- Không có official test name trong ba tập.
- Tối thiểu 70% physical train images.
- Tối thiểu 8 validation images.
- Cùng manifest luôn tạo cùng JSON byte-for-byte.
- Đổi thứ tự input manifest nhưng giữ cùng camera/name mapping không đổi split.

### 6.4. Split-specific sparse initialization

- Chỉ giữ point có ít nhất một observation thuộc internal train.
- Guard và validation observations không được dùng để tính color.
- Mỗi train observation sample RGB tại raw distorted image coordinate bằng
  bilinear interpolation; invalid/out-of-frame observation bị bỏ.
- Dữ liệu hiện tại có `points2D.xy` trong `images.bin` ở domain ảnh gốc trong
  khi physical train image và camera calibration đã downsample. Builder suy ra
  integer scale X/Y từ percentile 99.9 của toàn bộ physical-train observation
  geometry, chuyển tọa độ về physical image domain trước bounds check và
  fail-fast nếu scale vượt 64. Không hardcode factor 4 theo scene.
- Point color là channel-wise median của valid train observation colors, round
  và cast `uint8`.
- Point không còn valid train color observation bị loại.
- XYZ tiếp tục dùng provided COLMAP geometry và giữ `float64` trước khi chuyển
  sang trainer.

## 7. Chọn calibration scenes

Pool chuẩn gồm 18 BTS scenes dưới `data/bts_scenes`. Sáu calibration scenes được
khóa trước khi đọc metric: `hcm0031`, `HCM0181`, `HCM0421`, `HCM1439`,
`HNI0131`, `HNI0265`. Tập này bao phủ nguồn public/private/new, distortion
thấp/cao và scene 103/200/205/240 ảnh. `bonsai` và `chair` thuộc
`data/auxiliary`, khác domain và không tham gia qualification.

Phase 4 baseline không dùng confirmation stage riêng. Sau candidate decision,
config thắng đi qua one-scene 30k qualification rồi được dùng để train độc lập
toàn bộ 18 BTS scenes.

## 8. Baseline và bounded hyperparameter search

### 8.1. Các giá trị cố định

| Parameter | Giá trị |
|---|---:|
| Resolution | factor 1, native output domain |
| Steps | 30,000 |
| Batch | 1 full image |
| Seed | 0 |
| Packed mode | true |
| SH max degree | 3 |
| Mean LR | `1.6e-4 → 1.6e-6` trong 30k |
| Scale LR | `5e-3` |
| Quaternion LR | `1e-3` |
| Opacity LR | `5e-2` |
| SH0 LR | `2.5e-3` |
| SHN LR | `1.25e-4` |
| DSSIM weight | `0.2` |
| Refine start/every/stop | `500 / 100 / 15,000` |
| Opacity reset | `3,000` |
| Prune opacity | `0.005` |
| AMP / MCMC / AbsGrad | off |

### 8.2. Candidate set

Chỉ thử hai candidate ở 7,000 steps trên sáu calibration scenes:

| Candidate | `grow_grad2d` | Mục đích |
|---|---:|---|
| `B0-reference` | `0.0002` | Research baseline |
| `B0-compact` | `0.0003` | Giảm Gaussian growth và resource cost |

Không grid-search learning rate, loss weight hoặc reset schedule trong Phase 4.
Candidate compact chỉ thắng nếu đồng thời:

- mean PSNR giảm không quá `0.25 dB`;
- mean SSIM giảm không quá `0.005`;
- mean LPIPS tăng không quá `0.01`;
- và giảm ít nhất 15% một trong hai: peak Gaussian count hoặc total wall time.

Nếu không thỏa toàn bộ, khóa `B0-reference`. Không chọn theo một scene riêng lẻ.

## 9. Tận dụng một NVIDIA L4

Mục tiêu là tăng useful compute throughput, không phải tăng allocated VRAM.

### 9.1. Input cache

- Decode, undistort và resize mỗi train image đúng một lần mỗi process.
- Cache image dưới dạng contiguous `uint8`, mask dưới dạng `bool` trong pageable
  host RAM; không cache float32.
- Chỉ dùng pinned-memory ring buffer hai samples cho H2D transfer; không pin cả
  scene để tránh làm VM lag.
- Convert `[0,255] uint8 → [0,1] float32` trên GPU.
- Cache phải trả pixel, mask và intrinsics giống uncached path bit-for-bit trước
  tensor conversion.
- Trước khi allocate, estimate cache bytes và require ít nhất 4 GiB system RAM
  còn trống sau allocation; nếu thiếu thì fail rõ hoặc dùng uncached path.

### 9.2. Profiling

- CUDA stages đo bằng CUDA events; wall time đo bằng `perf_counter` sau
  synchronize tại boundary của profiling window.
- Warm-up 50 steps, đo 500 steps cùng scene/config/seed.
- So sánh uncached và cached path trên cùng process setup.
- Cached path phải nhanh hơn ít nhất 10% median step time hoặc giảm CPU
  preprocessing xuống dưới 10% wall step time.
- Cached/uncached phải dùng cùng sampled-image index sequence trong toàn bộ 500
  measured steps. Loss trace phải allclose với `rtol=5e-4, atol=1e-6` và
  Gaussian count phải bằng nhau trên các step trước density refinement đầu tiên.
  Topology delta từ refinement được report riêng vì threshold có thể khuếch đại
  nondeterminism CUDA nhỏ; không được dùng quy tắc này để bỏ qua divergence xảy
  ra trước refinement.
- Không đặt acceptance theo phần trăm VRAM sử dụng hay một mẫu `nvidia-smi`.

### 9.3. Resource limits

- Peak PyTorch allocated VRAM `< 20 GiB` để chừa headroom CUDA/driver.
- OOM, host swap hoặc system RAM còn dưới 4 GiB là fatal.
- Trước atomic checkpoint, free disk phải lớn hơn `2 × estimated_checkpoint_size
  + 10 GiB`.
- Training không tự động xóa artifact nếu không có explicit retention policy.

## 10. Lộ trình triển khai tuần tự

### Phiên 4.1 — Scene inventory và feasibility gate

**Mục đích:** chứng minh dữ liệu, RAM, disk và output capacity đủ trước khi dùng
GPU cho qualification.

**Mục tiêu triển khai riêng:** tạo production inventory API/CLI và report
deterministic cho toàn bộ scene.

**Deliverables:**

- `SceneInventory` và JSON report tổng;
- kiểm tra manifest availability, physical train images, native resolutions;
- estimate cache RAM, Gaussian/checkpoint disk và render output disk;
- deterministic six-scene calibration matrix trên pool 18 BTS scenes.

**Acceptance:**

- mọi scene có manifest hợp lệ và ít nhất 150 physical train images trước split;
- inventory không đọc test pose values/names để tạo features;
- scene count khác expected 13 làm cohort lock fail;
- report byte-deterministic và không có NaN/Infinity;
- real-data smoke pass trên năm scene hiện có, nhưng trạng thái vẫn ghi
  `incomplete_cohort` cho tới khi đủ 13.

**Điểm dừng:** red review inventory math và capacity estimates; chưa sửa trainer.

### Phiên 4.2 — Leakage-controlled holdout

**Mục đích:** tạo bằng chứng novel-view nội bộ đáng tin hơn train-camera metric.

**Mục tiêu triển khai riêng:** triển khai `pose_fps_guard2_v1`, subset dataset và
split-specific sparse colors.

**Deliverables:**

- holdout build/save/load/validation;
- dataset subset contract;
- train-only sparse point support/color initialization;
- split visualization notebook: camera centers với màu train/guard/validation,
  nearest-pose distance histogram và image contact sheet.

**Acceptance:**

- mọi split invariant ở Mục 6.3 pass;
- thay validation pixels không thể thay initial Gaussian colors;
- validation/guard image index không bao giờ được train sampler trả về;
- same manifest tạo same split/hash trên Windows và Linux;
- notebook trực quan hóa rõ coverage và near-duplicate guard;
- synthetic leakage tests và một real-scene split pass.

**Điểm dừng:** red review split, sparse color và visualization trước GPU run.

### Phiên 4.3 — Input cache và GPU profiling

**Mục đích:** giảm GPU idle do lặp JPEG decode/undistort mà không đổi baseline.

**Mục tiêu triển khai riêng:** thêm cache host-memory tối thiểu và profiler có
timing đúng.

**Deliverables:**

- batch script tạo/validate `manifest.json`, `arrays.npz` và `holdout.json` cho
  mọi scene hiện có, với strict 13-scene mode khi cohort đầy đủ;
- optional `cache_images` dataset path;
- two-sample pinned transfer ring;
- 50-step warm-up + 500-step controlled profiler report;
- RAM/VRAM/disk preflight.

**Acceptance:**

- cached/uncached samples bằng nhau;
- profile sampler chỉ dùng internal-train; guard và validation không được train;
- optimization trace bằng nhau trong tolerance đã khóa;
- performance gate Mục 9.2 pass trên HCM0181 factor 1;
- VM không swap và terminal vẫn responsive;
- không đổi batch size, optimizer, strategy hoặc dependency.

**Điểm dừng:** red review profiling evidence; cache bị loại nếu không có lợi ích.

### Phiên 4.4 — Bounded hyperparameter qualification

**Mục đích:** chọn resource/quality trade-off mà không overfit validation hoặc
biến Phase 4 thành open-ended HPO.

**Mục tiêu triển khai riêng:** chạy đúng hai candidate trên sáu calibration
scenes đã khóa trước khi đọc metric, với split, seed và 7k horizon cố định.

**Deliverables:**

- twelve run artifacts;
- per-image và per-scene PSNR/SSIM/LPIPS reports;
- Gaussian/VRAM/time comparison;
- machine-readable candidate decision với rule Mục 8.2.

**Runbook tối thiểu:**

```bash
./scripts/prepare_phase4_artifacts.sh
./scripts/run_phase4_qualification.sh
```

Script đầu yêu cầu đúng 18 scene dưới `data/bts_scenes` và sinh manifest cùng
holdout vào `runs/manifests`. Script thứ hai gọi lại bước chuẩn bị một cách
idempotent, smoke-test LPIPS rồi chạy ma trận 6 scene × 2 candidate vào
`runs/phase4/qualification`. Run đã có `qualification_report.json` được bỏ qua;
thư mục run dở dang làm script dừng và không bị tự động xóa. Có thể đổi root
trên VM bằng `PYTHON_BIN`, `BTS_SCENES_ROOT`, `BTS_MANIFESTS_ROOT` và
`BTS_QUALIFICATION_ROOT`.

**Acceptance:**

- mọi run có 7,000 finite steps và phục hồi sau reset step 6,000;
- không candidate nào được thêm sau khi nhìn validation result;
- decision code tái tạo đúng từ raw reports;
- sáu scene cố định là `hcm0031`, `HCM0181`, `HCM0421`, `HCM1439`, `HNI0131`
  và `HNI0265`;
- không có confirmation stage riêng trong baseline; candidate thắng chuyển sang
  one-scene 30k qualification.

**Điểm dừng:** red review candidate decision; chưa chạy 30k.

### Phiên 4.5 — Full-length 30k qualification

**Mục đích:** kiểm tra toàn bộ densification window và hậu densification trước
multi-scene spend.

**Mục tiêu triển khai riêng:** chạy config đã chọn 30k trên calibration scene gần
feature centroid nhất, dùng internal holdout cố định.

**Deliverables:**

- complete 30k training artifacts;
- validation renders và metrics tại step 30k;
- Gaussian/VRAM/time curves qua mốc 15k;
- resume test từ recovery checkpoint trên một short continuation branch.

**Runbook:**

```bash
./scripts/run_phase4_30k_dry_run.sh

# Chỉ dùng sau khi một run dở đã có checkpoints/recovery.pt
BTS_RESUME=1 ./scripts/run_phase4_30k_dry_run.sh
```

Runner chỉ giữ một rolling atomic checkpoint, không tạo chuỗi checkpoint theo
mốc. Internal validation được đo tại initialization và 30k nhưng chỉ lần 30k
ghi render; public test RGB không được đọc trong phase này.

**Acceptance:**

- 30,000 ordered finite metric/timing records;
- không OOM, swap, NaN/Inf hoặc corrupted checkpoint;
- peak VRAM `< 20 GiB` và peak Gaussians `< 10,000,000`;
- validation PSNR tăng `> 3 dB`, SSIM tăng `> 0.05`, LPIPS giảm `> 0.05` so với
  initialization;
- mean train–validation PSNR gap `< 8 dB`;
- final renders đúng pose, không blank/collapse và không severe floater;
- resume hash/RNG validation pass.

**Điểm dừng:** red review toán học, quality và resource growth.

### Phiên 4.6 — Freeze baseline B0

**Mục đích:** ngăn config drift sau khi bắt đầu production cohort.

**Mục tiêu triển khai riêng:** tạo immutable baseline bundle và run contract.

**Deliverables:**

- qualification ba backend `adam/fp32`, `adam-fused/fp32` và
  `adam-fused/amp-fp16` bằng 1,000 full-resolution steps trên HCM0181;
- gradient audit xác nhận `means2d.grad` đã được unscale trước density strategy;
- canonical `phase4_baseline.yaml`;
- SHA-256 của config, dependency lock, code commit và holdout algorithm;
- qualification summary và explicit grading-harness caveats;
- compact inference artifact schema chứa Gaussian parameters, active SH degree,
  normalization/config/manifest hashes, không chứa optimizer states.

**Acceptance:**

- fused Adam chỉ được chọn nếu nhanh hơn reference ít nhất 10%; AMP chỉ được
  chọn nếu đúng objective và nhanh thêm ít nhất 5% so với fused FP32;
- Gaussian parameters và Adam states luôn là `float32`; AMP scaler state phải
  round-trip qua recovery checkpoint;
- ba backend dùng cùng sample trace; loss trước topology all-close, final
  Gaussian count lệch không quá 1% và rolling-100 loss lệch không quá 2%;
- thay một config scalar làm hash đổi;
- trainer/resume/render reject hash mismatch;
- compact artifact render khớp full checkpoint trong `atol=1/255`;
- baseline file không chứa scene-specific tuned value;
- mọi open question leaderboard được ghi, không silently assumed.

**Điểm dừng:** baseline freeze commit; mọi thay đổi sau đó phải tạo baseline ID mới.

### Phiên 4.7 — Sequential multi-scene full training

**Mục đích:** huấn luyện toàn bộ scene an toàn trên một L4 mà một scene lỗi không
làm mất trạng thái cohort.

**Mục tiêu triển khai riêng:** CLI orchestration tuần tự với preflight, resume và
deterministic status ledger.

Production run dùng **toàn bộ physical train images**, không dùng internal
holdout; holdout chỉ dành cho qualification. Config/hyperparameter vẫn giữ
nguyên baseline đã khóa.

**Deliverables:**

- sorted scene queue;
- ledger trạng thái `pending/running/complete/failed`;
- rolling atomic recovery checkpoint cho scene đang chạy;
- final compact model và per-scene training report;
- bounded checkpoint retention, không lưu checkpoint dày đặc.

**Acceptance:**

- scene chạy đúng thứ tự scene ID, một process GPU tại một thời điểm;
- rerun bỏ qua scene complete chỉ khi hashes và artifact validation pass;
- failed scene không đánh dấu các scene khác complete;
- disk preflight chạy trước mỗi scene và fail trước `torch.save` nếu thiếu;
- mỗi scene complete có 30k records, compact model, config/environment/hash và
  non-blank fixed train preview;
- peak VRAM/RAM/Gaussians nằm trong limits đã khóa.

**Điểm dừng:** red review từng scene; không render official test từ model fail.

### Phiên 4.8 — Test rendering, benchmark và final audit

**Mục đích:** tạo output đúng contract và kiểm tra completeness trước submission.

**Mục tiêu triển khai riêng:** batch render từ compact model qua canonical test
camera path và dùng Phase 2 submission validator.

Training diễn ra trong undistorted pinhole domain. Với `SIMPLE_RADIAL`, output
phải quay lại distorted native domain:

1. Render pinhole tại `K` và native width/height.
2. Với mỗi distorted destination pixel, chuyển về normalized distorted point.
3. Invert radial distortion để tìm normalized undistorted source point.
4. Bilinear sample pinhole render; `PINHOLE` dùng identity mapping.
5. Encode RGB PNG bằng exact `test_output_names`.

**Deliverables:**

- per-scene canonical PNG outputs;
- internal qualification benchmark report;
- submission validation report;
- final cohort resource/quality notebook;
- checksums cho compact models và PNG outputs.

**Acceptance:**

- đúng một output cho mọi test pose của mọi scene;
- exact filename, case, PNG RGB và original resolution;
- không symlink, missing/extra scene hoặc file;
- renderer không đọc internal validation RGB hoặc official test RGB;
- deterministic render: same model/camera tạo same PNG bytes trong cùng pinned
  environment;
- report JSON standard-compliant, không NaN/Infinity;
- Phase 2 submission validator pass toàn bộ cohort.

**Điểm dừng:** chỉ artifact pass validator mới được coi là submission candidate.

## 11. Artifact layout

```text
runs/phase4/
├── inventory.json
├── cohort.json
├── baseline/
│   ├── phase4_baseline.yaml
│   ├── hashes.json
│   └── qualification_summary.json
├── holdouts/<scene_id>.json
├── qualification/<scene_id>/<candidate_id>/...
├── production/<scene_id>/
│   ├── config.yaml
│   ├── environment.json
│   ├── hashes.json
│   ├── checkpoints/recovery.pt
│   ├── model/final.pt
│   ├── metrics.jsonl
│   ├── timing.json
│   └── summary.json
├── outputs/<scene_id>/*.png
├── ledger.json
└── final_report.json
```

`final.pt` ở layout trên là compact inference artifact, không phải full optimizer
checkpoint. Atomic recovery write cần temporary sibling và rename.

## 12. Phase 4 exit criteria

Phase 4 chỉ complete khi:

- đủ expected 18 BTS scenes và mọi manifest pass;
- holdout algorithm, split artifacts và leakage tests pass;
- một 30k qualification pass quality/resource gates;
- baseline bundle đã freeze và hash;
- mọi production scene complete ở 30k hoặc có explicit approved exclusion;
- mọi test pose có canonical output;
- submission validator pass toàn bộ output tree;
- code commit, dependencies, configs, model/output hashes và reports đầy đủ;
- không dùng official test data cho tuning hoặc optimization.

Nếu grading harness cung cấp `PSNR_max`, LPIPS backbone hoặc SSIM config khác,
chỉ evaluator/report config được cập nhật. Baseline training không được âm thầm
thay đổi sau khi đã freeze.

## References

[3dgs-paper]: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
[3dgs-code]: https://github.com/graphdeco-inria/gaussian-splatting
[mipnerf360]: https://arxiv.org/abs/2111.12077
[ns-parser]: https://docs.nerf.studio/reference/api/data/dataparsers.html
[gsplat]: https://github.com/nerfstudio-project/gsplat
[splatfacto]: https://docs.nerf.studio/nerfology/methods/splat.html
