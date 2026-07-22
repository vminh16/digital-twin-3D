# Validation Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the CPU-only validation foundation for scene-specific experiments: high-frequency detail diagnostics, deterministic held-out pose strata, and one provenance-checked experiment report.

**Architecture:** Keep the existing holdout and evaluator unchanged. Add three focused modules under `bts_nvs.evaluation`; each consumes existing manifest, holdout, dataset, or validation-report contracts and exposes deterministic plain-dict JSON artifacts. No training, renderer, candidate, CLI, or GPU behavior changes in this plan.

**Tech Stack:** Python 3.10+, NumPy, OpenCV, Pillow, pytest, existing `bts_nvs` manifest/holdout/dataset APIs.

## Global Constraints

- Implement directly on `main`; do not create a new branch or worktree.
- Preserve `B0-submission-q99-v1` and existing evaluation semantics.
- Reuse `pose_fps_guard2_v1`, `SceneManifest`, `HoldoutSplit`, and existing validation image names.
- Use no external data, pretrained model, test RGB, or BTS segmentation.
- High-frequency metrics are explanatory and veto-only, not official score components.
- Do not modify trainer, renderer, qualification runner, inference, codec, or submission code.
- Use TDD: each production module is written only after its focused test fails because that module is missing.
- Commit each task separately and stage only files listed by that task.

---

### Task 1: High-Frequency Detail Metrics

**Files:**
- Create: `src/bts_nvs/evaluation/detail_metrics.py`
- Create: `tests/unit/test_detail_metrics.py`

**Interfaces:**
- Consumes: dataset samples exposing `image`, `valid_mask`, and `image_name`; a dataset exposing `manifest.scene_id`; PNG validation renders.
- Produces: `detail_metrics(prediction, target, valid_mask=None) -> dict[str, float]` and `evaluate_detail_directory(dataset, render_dir) -> dict[str, object]`.

- [ ] **Step 1: Write tests for identity, blur, noise, shifted edges, invalid arrays, masks, and render-directory identity**

```python
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from bts_nvs.evaluation.detail_metrics import (
    detail_metrics,
    evaluate_detail_directory,
)


def _step_image(offset: int = 16) -> np.ndarray:
    image = np.zeros((32, 32, 3), dtype=np.float64)
    image[:, offset:] = 1.0
    return image


def test_identical_images_have_zero_detail_error():
    image = _step_image()
    assert detail_metrics(image, image) == {
        "hf_l1": 0.0,
        "missing_edge": 0.0,
        "spurious_edge": 0.0,
        "symmetric_edge_distance": 0.0,
    }


def test_blur_noise_and_shift_activate_expected_diagnostics():
    target = _step_image()
    blurred = cv2.GaussianBlur(target, (9, 9), 2.0)
    noisy = target.copy()
    noisy[:8, :8] = (np.indices((8, 8)).sum(axis=0) % 2)[..., None]
    shifted = _step_image(20)
    assert detail_metrics(blurred, target)["missing_edge"] > 0.0
    assert detail_metrics(noisy, target)["spurious_edge"] > 0.0
    assert detail_metrics(shifted, target)["symmetric_edge_distance"] > 0.0
```

Add focused tests that reject non-RGB, mismatched, non-finite, out-of-range, and invalid-mask inputs. Add a tiny fake dataset test proving invalid pixels are replaced by reference values, output names are canonical PNG names, missing files fail, extra files fail, and `.JPG`/`.png` name collisions fail.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest -q tests/unit/test_detail_metrics.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'bts_nvs.evaluation.detail_metrics'`.

- [ ] **Step 3: Implement the minimal detail module**

Implement fixed luminance, 3x3 Sobel magnitude, 3x3 Laplacian, target top-10-percent edge mask, target bottom-50-percent flat mask, and a symmetric edge distance normalized by image diagonal. Both-empty edge masks return `0.0`; exactly-one-empty returns `1.0`. Restrict aggregates to valid pixels and replace invalid prediction pixels with target pixels.

`evaluate_detail_directory` must enumerate every dataset sample, derive `Path(image_name).with_suffix(".png").name`, reject case-insensitive collisions, reject missing/extra files, require RGB and exact resolution, and return:

```python
{
    "schema_version": 1,
    "scene_id": dataset.manifest.scene_id,
    "image_count": len(dataset),
    "hf_l1_mean": ...,
    "missing_edge_mean": ...,
    "spurious_edge_mean": ...,
    "symmetric_edge_distance_mean": ...,
    "images": {sample.image_name: detail_metrics(...)},
}
```

- [ ] **Step 4: Run focused and adjacent metric tests**

Run:

```bash
pytest -q tests/unit/test_detail_metrics.py tests/unit/test_metrics.py tests/unit/test_evaluator.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/bts_nvs/evaluation/detail_metrics.py tests/unit/test_detail_metrics.py
git commit -m "feat: add validation detail diagnostics"
```

### Task 2: Deterministic Held-Out Pose Strata

**Files:**
- Create: `src/bts_nvs/evaluation/pose_strata.py`
- Create: `tests/unit/test_pose_strata.py`

**Interfaces:**
- Consumes: `SceneManifest`, validated `HoldoutSplit`, and `manifest_pose_distance_matrix()` from `bts_nvs.data.holdout`.
- Produces: `assign_pose_strata(distances: Mapping[str, float]) -> dict[str, str]`, `build_pose_strata(manifest, split) -> dict[str, object]`, and `save_pose_strata(report, path) -> None`.

- [ ] **Step 1: Write tests for deterministic tertiles and nearest retained-train pose metadata**

```python
def test_assign_pose_strata_is_order_independent_and_exhaustive():
    distances = {f"v{i}.JPG": float(i) for i in range(8)}
    forward = assign_pose_strata(distances)
    reverse = assign_pose_strata(dict(reversed(tuple(distances.items()))))
    assert forward == reverse
    assert tuple(forward.values()).count("easy") == 3
    assert tuple(forward.values()).count("medium") == 3
    assert tuple(forward.values()).count("hard") == 2


def test_build_pose_strata_records_nearest_retained_train_camera():
    manifest = synthetic_manifest_with_linear_camera_centers()
    split = build_pose_holdout(manifest)
    report = build_pose_strata(manifest, split)
    assert report["algorithm"] == "nearest_train_tertiles_v1"
    assert set(report["images"]) == set(split.validation_image_names)
    assert {item["stratum"] for item in report["images"].values()} == {
        "easy", "medium", "hard"
    }
```

Also test duplicate/non-finite distance rejection, deterministic tie-breaking by filename, canonical JSON round-trip, and rejection when a split does not match the manifest.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest -q tests/unit/test_pose_strata.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'bts_nvs.evaluation.pose_strata'`.

- [ ] **Step 3: Implement deterministic pose strata**

Use existing combined pose distance to select the nearest retained training camera. Record separate normalized center distance and optical-axis angle in degrees. Sort validation records by `(pose_distance, image_name)`, partition ordered indices with `numpy.array_split(..., 3)`, and assign labels in `easy`, `medium`, `hard` order.

Return:

```python
{
    "schema_version": 1,
    "scene_id": manifest.scene_id,
    "algorithm": "nearest_train_tertiles_v1",
    "holdout_algorithm": split.algorithm,
    "holdout_manifest_sha256": split.manifest_sha256,
    "image_count": len(split.validation_image_names),
    "images": {
        image_name: {
            "nearest_train_image_name": ...,
            "pose_distance": ...,
            "center_distance": ...,
            "rotation_angle_deg": ...,
            "stratum": ...,
        }
    },
}
```

Save JSON atomically with sorted keys, finite values, UTF-8, and one trailing newline.

- [ ] **Step 4: Run focused and existing holdout tests**

Run:

```bash
pytest -q tests/unit/test_pose_strata.py tests/unit/test_holdout.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/bts_nvs/evaluation/pose_strata.py tests/unit/test_pose_strata.py
git commit -m "feat: stratify validation cameras by pose"
```

### Task 3: Provenance-Checked Experiment Report

**Files:**
- Create: `src/bts_nvs/evaluation/experiment_report.py`
- Create: `tests/unit/test_experiment_report.py`

**Interfaces:**
- Consumes: existing per-image `psnr_db`/`ssim`/`lpips` validation report, Task 1 detail report, Task 2 pose-strata report, immutable identity fields, and a resource summary.
- Produces: `local_score50(metrics) -> float`, `build_experiment_report(...) -> dict[str, object]`, and `save_experiment_report(report, path) -> None`.

- [ ] **Step 1: Write tests for valid aggregation, hard-stratum reporting, set mismatches, hashes, resources, and canonical JSON**

```python
def test_build_experiment_report_aggregates_matching_images_by_stratum():
    report = build_experiment_report(
        scene_id="HCM0539",
        candidate_id="B0-reference",
        step=7000,
        config_sha256="a" * 64,
        manifest_sha256="b" * 64,
        holdout_sha256="c" * 64,
        full_frame_report=full_frame_fixture(),
        detail_report=detail_fixture(),
        pose_strata_report=pose_fixture(),
        resource_summary={
            "total_time_seconds": 100.0,
            "max_vram_mb": 5000.0,
            "peak_gaussians": 1200000,
            "final_num_gaussians": 1100000,
        },
    )
    assert report["image_count"] == 3
    assert report["strata"]["hard"]["image_count"] == 1
    assert report["overall"]["score50"] == pytest.approx(
        local_score50(report["overall"])
    )
```

Add parameterized tests for mismatched image sets/counts, unknown strata,
non-finite metrics/resources, non-positive steps, empty IDs, malformed SHA-256
values, and negative Gaussian/resource values. Verify save is byte-identical on
repeat and contains neither `NaN` nor `Infinity`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest -q tests/unit/test_experiment_report.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'bts_nvs.evaluation.experiment_report'`.

- [ ] **Step 3: Implement report validation and aggregation**

Use the fixed local diagnostic formula:

```python
score50 = 40.0 - 40.0 * lpips + 30.0 * ssim + 0.6 * psnr_db
```

Require exact image-name equality across full-frame, detail, and pose reports.
Aggregate all numeric image metrics overall and separately for `easy`,
`medium`, and `hard`. Preserve resource and provenance fields without reading
training internals. Return schema version 1 with exact scene/candidate/step
identity and canonical dictionaries.

- [ ] **Step 4: Run the complete Module 1 focused suite**

Run:

```bash
pytest -q \
  tests/unit/test_detail_metrics.py \
  tests/unit/test_pose_strata.py \
  tests/unit/test_experiment_report.py \
  tests/unit/test_metrics.py \
  tests/unit/test_evaluator.py \
  tests/unit/test_holdout.py
```

Expected: all tests pass.

- [ ] **Step 5: Run the full CPU unit suite**

Run:

```bash
pytest -q tests/unit
```

Expected: all unit tests pass. GPU integration tests are not part of Module 1.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/bts_nvs/evaluation/experiment_report.py tests/unit/test_experiment_report.py
git commit -m "feat: build scene experiment reports"
```

### Task 4: Module 1 Contract Verification

**Files:**
- Modify: `docs/superpowers/plans/2026-07-22-validation-foundation.md`

**Interfaces:**
- Consumes: all three task commits and the umbrella spec.
- Produces: checked task boxes and a verified Module 1 handoff; no production behavior.

- [ ] **Step 1: Verify scope and changed files**

Run:

```bash
git diff --name-status c78025c..HEAD
```

Expected: only this plan, three evaluation modules, and three focused unit-test files.

- [ ] **Step 2: Verify formatting and incomplete markers**

Run:

```bash
git diff --check c78025c..HEAD
rg -n "TB[D]|TO[D]O|implement[ ]later|fill[ ]in details|similar[ ]to Task" \
  docs/superpowers/plans/2026-07-22-validation-foundation.md \
  src/bts_nvs/evaluation/detail_metrics.py \
  src/bts_nvs/evaluation/pose_strata.py \
  src/bts_nvs/evaluation/experiment_report.py
```

Expected: `git diff --check` succeeds and `rg` finds no matches.

- [ ] **Step 3: Record the completed checklist and commit only the plan update**

```bash
git add docs/superpowers/plans/2026-07-22-validation-foundation.md
git commit -m "docs: complete validation foundation plan"
```

The Module 1 handoff must report exact test counts, commit IDs, unchanged user-owned files, and that no GPU test or training run was performed.
