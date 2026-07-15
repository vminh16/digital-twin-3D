# Phase 4.2 Leakage-Controlled Holdout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic pose holdout, restrict dataset sampling to explicit image subsets, and recompute sparse initialization colors using internal-train RGB observations only.

**Architecture:** Add one focused holdout module for pose distance, split invariants, hashing and JSON. Extend the existing dataset with an index-only subset view. Preserve COLMAP 2D observations in the adapter and add a separate sparse-subset builder that streams each internal-train image once. The notebook consumes these production APIs and contains no alternate camera or split implementation.

**Tech Stack:** Python 3.12, NumPy 1.26.4, Pillow, Matplotlib, pycolmap 4.1.0, pytest.

## Global Constraints

- Algorithm identifier is exactly `pose_fps_guard2_v1`; schema version is `1`.
- Split input is physical train names, C2W poses and normalization from `SceneManifest`; official test pose values and names never affect selection.
- Validation target is `max(8, floor(N / 8 + 0.5))` with half-up semantics.
- Each validation image contributes its two nearest non-validation cameras to the union guard set.
- Internal train must retain at least `max(120, ceil(0.70 * N))` images; otherwise remove the last FPS selection and recompute guard.
- Train, validation and guard are pairwise disjoint and their union equals exact manifest train names.
- Point geometry may use provided COLMAP reconstruction, but point support and RGB initialization use internal-train observations only.
- JSON is UTF-8/LF, sorted-key, `allow_nan=False`, atomic and byte-deterministic on Windows/Linux.
- Do not modify trainer, renderer, optimizer, Phase 4.1 cohort policy or Phase 4.3 cache behavior.
- Commit directly to `main`; preserve the user notebook edit and pilot archive.

---

### Task 1: Deterministic holdout contract

**Files:**
- Create: `src/bts_nvs/data/holdout.py`
- Modify: `src/bts_nvs/data/__init__.py`
- Create: `tests/unit/test_holdout.py`

**Interfaces:**
- Consumes: `SceneManifest`.
- Produces: `HoldoutSplit`, `manifest_holdout_sha256`, `build_pose_holdout`, `validate_holdout_split`, `save_holdout_split`, `load_holdout_split`.

- [ ] **Step 1: Write RED tests for pose distance and split invariants**

Create a synthetic manifest with at least 150 cameras. Assert the formula
`||Ci-Cj|| + 0.25*acos(vi·vj)/pi`, exact target/guard/train minimums,
pairwise disjointness, full train-name coverage and no official-test overlap.

- [ ] **Step 2: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_holdout.py -q
```

Expected: collection fails because `bts_nvs.data.holdout` does not exist.

- [ ] **Step 3: Implement pose FPS minimally**

Use normalized camera centers from `normalization_transform` and optical axes
`R_cw[:, 2]`. Sort camera records by exact image name before every calculation.
Tie-break seed, FPS and guard ranking by exact image name. Recompute guard after
each removed validation seed until the train minimum is met; reject fewer than
8 validation images.

- [ ] **Step 4: Add RED tests for order-independent hash and serialization**

Reorder all train-aligned manifest fields while preserving name-camera mapping;
assert identical split/hash. Save twice and assert identical bytes, LF-only,
one trailing newline and no absolute path. Assert unsupported schema, changed
manifest geometry and tampered partition fail load.

- [ ] **Step 5: Implement canonical hash/save/load**

Hash scene ID plus sorted `(image_name, C2W)` records and normalization as
little-endian float64 with length-prefixed UTF-8 names. JSON stores only schema,
scene, hash, algorithm and three name lists. Load reconstructs the dataclass,
checks schema/hash/invariants, then compares against a freshly rebuilt split.

- [ ] **Step 6: Run GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_holdout.py tests/unit/test_inventory.py -q
```

Expected: all selected tests pass and Phase 4.1 behavior is unchanged.

---

### Task 2: Dataset subset isolation

**Files:**
- Modify: `src/bts_nvs/data/dataset.py`
- Modify: `tests/unit/test_dataset.py`

**Interfaces:**
- Consumes: optional `image_names: tuple[str, ...] | None`.
- Produces: the existing `SceneDataset`/`CameraSample` APIs with subset length, order and sample isolation.

- [ ] **Step 1: Write RED subset tests**

Extend the fixture to three images. Construct `SceneDataset(...,
image_names=("c.png", "a.png"))`; assert length two, exact output order and no
sample for `b.png`. Assert duplicate, unknown and official-test names fail.

- [ ] **Step 2: Run RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_dataset.py -q
```

Expected: constructor rejects the new keyword.

- [ ] **Step 3: Implement an index-only view**

Build a manifest-name-to-index map once. `None` maps to `range(N)`; an explicit
tuple maps names in caller order after duplicate/unknown checks. `__len__` uses
the view and `__getitem__` translates the subset index before existing image,
pose, distortion, undistort and resize logic. Do not add caching.

- [ ] **Step 4: Run GREEN and regression**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_dataset.py tests/unit/test_trainer_loop.py -q
```

Expected: subset tests and existing full-dataset trainer behavior pass.

---

### Task 3: Train-only sparse support and colors

**Files:**
- Modify: `src/bts_nvs/data/colmap.py`
- Create: `src/bts_nvs/data/sparse_subset.py`
- Modify: `src/bts_nvs/data/__init__.py`
- Modify: `tests/unit/test_colmap_adapter.py`
- Create: `tests/unit/test_sparse_subset.py`

**Interfaces:**
- Extends `ColmapImageRecord` with read-only `points2d_xy: float64[M,2]` and `point3d_ids: int64[M]`, both default-empty for backward compatibility.
- Produces immutable `SparseInitialization(point_ids, points, colors)` and `build_split_sparse_initialization(manifest, scene_root, split)`.

- [ ] **Step 1: Write RED adapter tests**

Add fake pycolmap `points2D` records with `xy` and `point3D_id`. Assert aligned,
canonical, read-only arrays and reject shape mismatch/non-finite coordinates.

- [ ] **Step 2: Implement the adapter extension**

Read `image.points2D` from pinned pycolmap 4.1.0 directly; do not add version
fallbacks. Existing positional test fixtures remain valid through default-empty
fields.

- [ ] **Step 3: Write RED sparse leakage tests**

Build a synthetic valid split and tiny raw RGB images. Include: a point observed
by train+validation, a validation-only point, an out-of-frame train observation,
and multiple train observations. Assert only train-supported valid points remain,
XYZ is float64, colors are uint8 channel medians, point IDs are sorted, and
changing validation pixels cannot change output colors.

- [ ] **Step 4: Implement one-pass-per-image color aggregation**

Validate the split against the manifest, read the COLMAP model, map exact train
names to registrations, and stream each internal-train image once. Bilinearly
sample raw distorted coordinates; ignore non-finite/out-of-frame/unregistered
observations. Aggregate colors by point ID, take channel median, round nonnegative
values half-up, and retain finite points with at least one sampled train color.

- [ ] **Step 5: Run GREEN and source-data regression**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_colmap_adapter.py tests/unit/test_colmap_source.py tests/unit/test_sparse_subset.py tests/unit/test_diagnostics.py -q
```

Expected: all selected tests pass without changing the Phase 2 manifest schema.

---

### Task 4: Real-scene audit notebook and phase gate

**Files:**
- Create: `notebooks/phase4_holdout_audit.ipynb`
- Create: `tests/integration/test_phase4_holdout.py`

**Interfaces:**
- Notebook consumes `load_scene_manifest`, `build_pose_holdout` and `SceneDataset` only.
- Integration test consumes the HCM0181 manifest when present; no CUDA.

- [ ] **Step 1: Add the real-data split smoke**

Load HCM0181, build twice, assert equality, at least 120 internal-train and 8
validation images, disjoint/full coverage, and zero overlap with official test
names. Skip only when local scene/manifest is absent.

- [ ] **Step 2: Create the visual notebook**

Use production APIs to render: split summary cards, top/side camera-center plots
colored train/guard/validation, nearest-pose-distance histogram, and a labeled
contact sheet. Parameterize `SCENE_ID`, `SCENE_ROOT` and `MANIFEST_JSON` in one
configuration cell. Do not duplicate pose-distance or split logic.

- [ ] **Step 3: Execute notebook headlessly**

```powershell
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute notebooks/phase4_holdout_audit.ipynb --output phase4_holdout_audit.executed.ipynb --output-dir runs/phase4
```

Expected: execution succeeds on HCM0181 and produces all four visual sections.

- [ ] **Step 4: Run Phase 4.2 and full regression**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit tests/integration/test_phase4_inventory.py tests/integration/test_phase4_holdout.py -q
.\.venv\Scripts\python.exe -m compileall -q src
git diff --check
```

Expected: tests pass; no trainer/renderer/cache changes; user notebook and pilot archive remain unstaged.

- [ ] **Step 5: Red review and commit**

Review split math, official-test isolation, hash portability, sparse observation
domain and image subset indexing. Stage only Phase 4.2 files and commit:

```powershell
git commit -m "feat: add leakage-controlled phase 4 holdout"
```

Stop before Phase 4.3 GPU/cache work.

## Plan self-review

- All Phase 4.2 acceptance criteria map to a test or notebook execution step.
- The plan does not add cache, pinned transfer, qualification metrics or trainer changes.
- Sparse RGB uses raw distorted images and provided COLMAP 2D coordinates in the same domain.
- Split selection and hash are invariant to manifest train-record ordering.
- Official test metadata is excluded from selection and sparse colors.
- No placeholder or unresolved public interface remains.
