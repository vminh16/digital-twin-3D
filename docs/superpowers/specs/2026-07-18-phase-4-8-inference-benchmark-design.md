# Phase 4.8 Inference and Local Benchmark Design

## Goal

Render canonical PNG outputs from trained per-scene Gaussian checkpoints at all
manifest test poses, validate the output contract, and provide a separate local
benchmark CLI that only runs when an explicit RGB reference root is supplied.

## Separation of responsibilities

Inference and evaluation are separate commands. `run_phase4_inference.sh` never
loads reference RGB or computes PSNR, SSIM, LPIPS, or a composite score. The
benchmark CLI only reads already-rendered outputs and an explicit reference
root; it never loads checkpoints or renders images.

## Inference interface

```bash
bash scripts/run_phase4_inference.sh \
  --scene_ids HCM0644 HCM0674 HCM0540 HCM0539 HCM0421
```

The default output root is `<repo>/outputs`, producing the exact layout:

```text
outputs/
├── HCM0644/<manifest.test_output_names>.png
├── HCM0674/<manifest.test_output_names>.png
└── ...
```

Path environment overrides follow the existing scripts: `BTS_SCENES_ROOT`,
`BTS_MANIFESTS_ROOT`, `BTS_FULL_ROOT`, `BTS_OUTPUT_ROOT`, and `PYTHON_BIN`.
Omitting `--scene_ids` selects all canonical scenes. Supplied IDs are ordered,
case-sensitive, canonical, non-empty, and duplicate-free.

## Inference camera and model contract

- Load `runs/phase4/full_training/scenes/<scene>/checkpoints/recovery.pt` only
  after validating the trained run, step 30,000, manifest hash, and config hash.
- Instantiate `GaussianParameters` from the checkpoint Gaussian state and use
  the checkpoint `active_sh_degree` on CUDA under inference mode.
- Convert raw test W2C into the normalized world domain by preserving rotation
  and transforming only the camera center, identically to training.
- Render a native-resolution undistorted pinhole RGB image with the manifest K.
- For `SIMPLE_RADIAL`, map each distorted destination pixel through
  `undistort_normalized_points`, then bilinear-sample the pinhole render. For
  `PINHOLE`, preserve pixels exactly.
- Clamp finite RGB to `[0,1]`, quantize deterministically to RGB uint8 and save
  PNG using `manifest.test_output_names`.
- Load and release one scene model at a time.

## Output transaction and validation

The requested batch renders into a temporary sibling of `outputs`. The complete
temporary tree is checked by the Phase 2 submission validator with only the
selected manifests, then atomically renamed to `outputs`. The final output root
must not already exist; no old output is overwritten or adopted. Failure or
interrupt cleans the temporary tree. `inference_report.json` records selected
scenes, per-scene image counts, checkpoint and manifest hashes, elapsed time,
and contains no quality metric.

## Local benchmark CLI

```bash
python -m bts_nvs.evaluation.run_benchmark \
  --outputs_root outputs \
  --reference_root data/local_references \
  --manifests_root runs/manifests \
  --scenes_root data/bts_scenes \
  --scene_ids HCM0644 HCM0674 HCM0540 HCM0539 HCM0421 \
  --psnr_max 40 \
  --lpips_backbone alex \
  --report_path runs/phase4/local_benchmark.json
```

The reference root must mirror `outputs/<scene>/<test_output_name>.png`. Exact
filenames, RGB mode, and native resolution are checked before metrics. The CLI
uses the existing Phase 2 evaluator, requires explicit `psnr_max`, records the
metric configuration, and rejects missing/extra/non-finite data. It never
searches scene `test/` directories for RGB and therefore cannot manufacture an
official-test score when ground truth is unavailable.
