# Phase 3.6 — NVIDIA L4 smoke-test runbook

## 1. Build and verify the GPU image

Run from the repository root on a Linux VM with an NVIDIA L4 and NVIDIA Container Toolkit:

```bash
docker build -t bts-nvs:phase3 .
docker run --rm --gpus all \
  -v "$PWD/data:/workspace/digital-twin-3D/data" \
  -v "$PWD/runs:/workspace/digital-twin-3D/runs" \
  bts-nvs:phase3 python tests/smoke_test.py
```

This command must fail if CUDA or the compiled gsplat backend is unavailable. Before continuing, verify inside the container:

```bash
python -c "import torch, gsplat; print(torch.__version__, torch.version.cuda, gsplat.__version__); print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Required result: CUDA is `True`, device is NVIDIA L4, capability is `(8, 9)`, and the synthetic gsplat forward/backward smoke exits with code 0.

## 2. Run the real 10-step integration gate

```bash
docker run --rm --gpus all \
  -v "$PWD/data:/workspace/digital-twin-3D/data" \
  -v "$PWD/runs:/workspace/digital-twin-3D/runs" \
  bts-nvs:phase3 pytest tests/integration/test_smoke_hcm0181.py -q -s
```

It must run with real CUDA; a skip is not a pass on the VM.

## 3. Run A — stability gate

```bash
docker run --rm --gpus all \
  -v "$PWD/data:/workspace/digital-twin-3D/data" \
  -v "$PWD/runs:/workspace/digital-twin-3D/runs" \
  bts-nvs:phase3 python src/bts_nvs/training/run_training.py \
    --scene_dir data/bts_scenes/HCM0181 \
    --manifest_dir runs/manifests/HCM0181 \
    --output_dir runs/HCM0181/run_a_factor4_500 \
    --resize_factor 4 --max_steps 500 --checkpoint_every 100 --seed 0
```

Run A passes when:

- exit code is 0 with no CUDA, OOM, NaN, or Inf error;
- `metrics.jsonl` contains exactly 500 ordered records;
- final checkpoint and `summary.json` exist;
- `max_vram_mb < 23000`;
- reference, step-0, and step-500 PNG files are visually aligned and final render is not black/empty;
- `convergence.json.final_render_non_blank` is `true`;
- PSNR and SSIM do not materially regress. A small fluctuation at the first density update is acceptable, but inspect the preview before Run B.

## 4. Run B — quality gate

```bash
docker run --rm --gpus all \
  -v "$PWD/data:/workspace/digital-twin-3D/data" \
  -v "$PWD/runs:/workspace/digital-twin-3D/runs" \
  bts-nvs:phase3 python src/bts_nvs/training/run_training.py \
    --scene_dir data/bts_scenes/HCM0181 \
    --manifest_dir runs/manifests/HCM0181 \
    --output_dir runs/HCM0181/run_b_factor2_7000 \
    --resize_factor 2 --max_steps 7000 --checkpoint_every 500 --seed 0
```

Run B passes when all Run A resource/integrity checks pass and:

- `metrics.jsonl` contains exactly 7,000 ordered finite records;
- `convergence.json.quality_improved` is `true`;
- fixed train-view PSNR improves by at least 1 dB;
- fixed train-view SSIM improves by at least 0.01;
- final preview remains geometrically aligned with the reference and is neither blank nor severely blurred.

These thresholds validate engine convergence, not leaderboard comparability.

## 5. Resume an interrupted Run B

Use the same scene, manifest, resize, seed, max steps, and output directory:

```bash
docker run --rm --gpus all \
  -v "$PWD/data:/workspace/digital-twin-3D/data" \
  -v "$PWD/runs:/workspace/digital-twin-3D/runs" \
  bts-nvs:phase3 python src/bts_nvs/training/run_training.py \
    --scene_dir data/bts_scenes/HCM0181 \
    --manifest_dir runs/manifests/HCM0181 \
    --output_dir runs/HCM0181/run_b_factor2_7000 \
    --resize_factor 2 --max_steps 7000 --checkpoint_every 500 --seed 0 \
    --resume runs/HCM0181/run_b_factor2_7000/checkpoints/step_000003000.pt
```

## 6. Phase 4.6 — accelerated backend qualification

PyTorch CUDA phải nhận đúng NVIDIA L4 (`compute capability 8.9`). Trước run
1,000 bước, chạy gradient smoke thật cho cả ba backend:

```bash
BTS_RUN_CUDA_BACKEND_SMOKE=1 python -m pytest \
  tests/integration/test_cuda_backend_smoke.py -q
```

Sau đó chạy ba job full-resolution tuần tự:

```bash
bash scripts/run_phase4_backend_qualification.sh
cat runs/phase4/backend_qualification/backend_qualification.json
```

Điều kiện pass:

- `accepted` là `true`;
- mọi gradient audit finite và density strategy nhận projected gradient đã
  unscale;
- sample traces và density-event schedule giống nhau;
- fused FP32 chỉ được chọn khi speedup tối thiểu 10%;
- AMP FP16 chỉ được chọn khi nhanh thêm tối thiểu 5% so với fused FP32;
- peak VRAM nhỏ hơn 20 GiB;
- candidate không đạt gate phải quay về reference trong report, không silently
  fallback trong training.

L4 hỗ trợ FP16/BF16/TF32 Tensor Cores, nhưng gate này chỉ thay đổi FP16 AMP để
speedup và sai số có một nguyên nhân duy nhất. Tham chiếu:
[PyTorch fused Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html),
[PyTorch AMP với nhiều optimizer](https://docs.pytorch.org/docs/stable/notes/amp_examples.html),
[NVIDIA L4 specifications](https://www.nvidia.com/en-gb/data-center/l4/).

Changing resize, seed, manifest, or optimization horizon must be rejected by checkpoint validation.

## 7. Phase 4.7 — full training 18 scenes without Docker

Activate the installed virtual environment, then start the sequential cohort:

```bash
source .venv/bin/activate
bash scripts/run_phase4_full_training.sh
```

Optional path overrides are limited to runtime locations:

```bash
BTS_SCENES_ROOT="$PWD/data/bts_scenes" \
BTS_MANIFESTS_ROOT="$PWD/runs/manifests" \
BTS_BACKEND_ROOT="$PWD/runs/phase4/backend_qualification" \
BTS_FULL_ROOT="$PWD/runs/phase4/full_training" \
PYTHON_BIN="$PWD/.venv/bin/python" \
bash scripts/run_phase4_full_training.sh
```

The wrapper regenerates/audits manifest artifacts, validates the exact canonical
18-scene pool, reads the accepted backend decision, and runs one GPU process at a
time. Monitor durable cohort state with:

```bash
cat runs/phase4/full_training/ledger.json
df -h "$PWD/runs/phase4/full_training"
nvidia-smi
```

If the process is interrupted, execute the same wrapper again. A scene with a
valid `checkpoints/recovery.pt` resumes automatically; a fully validated scene is
skipped. A non-zero scene exit stops the queue and records `failed`. `Ctrl+C`
leaves the current scene as `running`, so its recovery artifact can be inspected
and resumed. Do not delete or rename partial run directories.

Only one rolling recovery checkpoint is retained per scene. A ledger status of
`trained` means the 30k training artifacts passed integrity checks; compact model
export and official test rendering are separate later steps.
