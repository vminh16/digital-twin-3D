# Phase 4.1 Scene Inventory & Feasibility Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic inventory and capacity preflight that blocks Phase 4 cohort locking until exactly 13 valid scenes are available and resource estimates fit the host.

**Architecture:** Add one focused data module for inventory math, cohort selection, capacity estimates, deterministic report serialization, and scene discovery. Add one thin CLI for machine resource checks and exit codes. Reuse `SceneManifest` and its existing validation; do not touch the trainer, dataset, renderer, or Phase 4.2 holdout logic.

**Tech Stack:** Python 3.12, NumPy 1.26.4, stdlib `argparse/json/itertools/shutil/tempfile/ctypes`, pytest.

## Global Constraints

- Production target is one NVIDIA L4 24 GB; scenes run sequentially.
- Expected cohort size is exactly 13 scenes.
- Every ready scene has at least 150 physical train images.
- Cohort features use train metadata and sparse geometry only; test pose values and test names never enter cohort selection.
- `test_pose_count` and test dimensions are used only for output-capacity estimates.
- Inventory JSON is deterministic: UTF-8, sorted keys, `allow_nan=False`, no timestamp, no absolute path, atomic replace.
- Keep `gsplat==1.4.0`; Phase 4.1 must not import torch or gsplat.
- Add no dependency and do not modify training behavior.
- Run TDD red → green for every task and stop for red review after Phase 4.1.

---

## File map

- Create `src/bts_nvs/data/inventory.py`: immutable contracts, inventory math, capacity heuristics, cohort selection, scene audit, deterministic JSON.
- Create `src/bts_nvs/data/run_inventory.py`: CLI, available-RAM detection, local feasibility check and stable exit codes.
- Modify `src/bts_nvs/data/__init__.py`: export only Phase 4.1 public contracts/functions.
- Create `tests/unit/test_inventory.py`: deterministic unit coverage for math, cohort, capacity, discovery and serialization.
- Create `tests/integration/test_phase4_inventory.py`: ignored-real-data smoke for HCM0181 and the currently available public-set layout.
- Reference only: `docs/phase_4_spec.md`, `src/bts_nvs/data/manifest.py`, `src/bts_nvs/training/trainer.py`.

## Locked interfaces and constants

```python
SCHEMA_VERSION = 1
EXPECTED_SCENE_COUNT = 13
MIN_TRAIN_IMAGES = 150
MAX_ESTIMATED_GAUSSIANS = 10_000_000
GAUSSIAN_GROWTH_FACTOR = 30
FULL_CHECKPOINT_BYTES_PER_GAUSSIAN = 768
COMPACT_MODEL_BYTES_PER_GAUSSIAN = 236
HOST_RAM_HEADROOM_BYTES = 4 * 1024**3
DISK_HEADROOM_BYTES = 10 * 1024**3

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

@dataclass(frozen=True)
class SceneCapacityEstimate:
    scene_id: str
    cache_bytes: int
    output_raw_bytes: int
    estimated_peak_gaussians: int
    full_checkpoint_bytes: int
    compact_model_bytes: int

@dataclass(frozen=True)
class CohortAssignment:
    calibration_scene_ids: tuple[str, ...]
    confirmation_scene_ids: tuple[str, ...]
    production_scene_ids: tuple[str, ...]

@dataclass(frozen=True, order=True)
class InventoryIssue:
    scene_id: str
    code: str
    detail: str

@dataclass(frozen=True)
class Phase4InventoryReport:
    schema_version: int
    expected_scene_count: int
    status: Literal["ready", "incomplete_cohort", "invalid"]
    scenes: tuple[SceneInventory, ...]
    capacities: tuple[SceneCapacityEstimate, ...]
    cohort: CohortAssignment | None
    issues: tuple[InventoryIssue, ...]
    required_host_ram_bytes: int
    required_artifact_disk_bytes: int
```

Capacity estimates are intentionally conservative project heuristics, not measured
guarantees. The report serializes the constants above so later phases can explain
how estimates were produced. Phase 4.5 replaces estimates with observed peaks.

---

### Task 1: Scene inventory primitives

**Files:**
- Create: `src/bts_nvs/data/inventory.py`
- Modify: `src/bts_nvs/data/__init__.py`
- Test: `tests/unit/test_inventory.py`

**Interfaces:**
- Consumes: `SceneManifest` from `bts_nvs.data.manifest`.
- Produces: `SceneInventory`, `SceneCapacityEstimate`, `build_scene_inventory(manifest)`, `estimate_scene_capacity(manifest, inventory)`.

- [ ] **Step 1: Write failing inventory math tests**

Create `tests/unit/test_inventory.py` with a minimal manifest-shaped fixture and
tests that lock p90 nearest-neighbor math, train-only distortion, canonical
dimensions, and capacity formulas:

```python
from types import SimpleNamespace

import numpy as np

from bts_nvs.cameras.distortion import CameraDistortion
from bts_nvs.cameras.intrinsics import CameraIntrinsics
from bts_nvs.data.inventory import (
    COMPACT_MODEL_BYTES_PER_GAUSSIAN,
    FULL_CHECKPOINT_BYTES_PER_GAUSSIAN,
    GAUSSIAN_GROWTH_FACTOR,
    build_scene_inventory,
    estimate_scene_capacity,
)


def manifest_stub(*, sparse_count: int = 10):
    c2w = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    c2w[:, 0, 3] = [0.0, 1.0, 3.0]
    train_k = (
        CameraIntrinsics(100, 50, 70.0, 70.0, 50.0, 25.0),
        CameraIntrinsics(120, 60, 80.0, 80.0, 60.0, 30.0),
        CameraIntrinsics(100, 50, 70.0, 70.0, 50.0, 25.0),
    )
    return SimpleNamespace(
        scene_id="scene_a",
        train_image_names=("a.JPG", "b.JPG", "c.JPG"),
        train_camera_to_world=c2w,
        train_intrinsics=train_k,
        train_distortion=(
            CameraDistortion("SIMPLE_RADIAL", (-0.1,)),
            CameraDistortion("PINHOLE", ()),
            CameraDistortion("SIMPLE_RADIAL", (0.02,)),
        ),
        test_image_names=("x.JPG", "y.JPG"),
        test_intrinsics=(
            CameraIntrinsics(80, 40, 60.0, 60.0, 40.0, 20.0),
            CameraIntrinsics(80, 40, 60.0, 60.0, 40.0, 20.0),
        ),
        sparse_points=np.zeros((sparse_count, 3), dtype=np.float64),
    )


def test_build_scene_inventory_uses_train_geometry_only():
    manifest = manifest_stub()
    inventory = build_scene_inventory(manifest)

    assert inventory.scene_id == "scene_a"
    assert inventory.train_image_count == 3
    assert inventory.test_pose_count == 2
    assert inventory.sparse_point_count == 10
    assert inventory.trajectory_nn_p90 == np.percentile(
        [1.0, 1.0, 2.0], 90, method="linear"
    )
    assert inventory.distortion_abs_max == 0.1
    assert inventory.native_widths == (100, 120, 100)
    assert inventory.native_heights == (50, 60, 50)


def test_capacity_formula_is_explicit_and_bounded():
    manifest = manifest_stub(sparse_count=10)
    inventory = build_scene_inventory(manifest)
    capacity = estimate_scene_capacity(manifest, inventory)

    peak = 10 * GAUSSIAN_GROWTH_FACTOR
    assert capacity.cache_bytes == 100 * 50 * 4 + 120 * 60 * 4 + 100 * 50 * 4
    assert capacity.output_raw_bytes == 2 * 80 * 40 * 3
    assert capacity.estimated_peak_gaussians == peak
    assert capacity.full_checkpoint_bytes == peak * FULL_CHECKPOINT_BYTES_PER_GAUSSIAN
    assert capacity.compact_model_bytes == peak * COMPACT_MODEL_BYTES_PER_GAUSSIAN
```

- [ ] **Step 2: Run the tests to verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_inventory.py -q
```

Expected: collection fails because `bts_nvs.data.inventory` does not exist.

- [ ] **Step 3: Implement immutable contracts and inventory math**

Create `src/bts_nvs/data/inventory.py` with the locked constants/contracts and
the following implementation. Validation errors use the existing
`DataContractError`.

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from .manifest import SceneManifest
from .validation import DataContractError

SCHEMA_VERSION = 1
EXPECTED_SCENE_COUNT = 13
MIN_TRAIN_IMAGES = 150
MAX_ESTIMATED_GAUSSIANS = 10_000_000
GAUSSIAN_GROWTH_FACTOR = 30
FULL_CHECKPOINT_BYTES_PER_GAUSSIAN = 768
COMPACT_MODEL_BYTES_PER_GAUSSIAN = 236
HOST_RAM_HEADROOM_BYTES = 4 * 1024**3
DISK_HEADROOM_BYTES = 10 * 1024**3


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


@dataclass(frozen=True)
class SceneCapacityEstimate:
    scene_id: str
    cache_bytes: int
    output_raw_bytes: int
    estimated_peak_gaussians: int
    full_checkpoint_bytes: int
    compact_model_bytes: int


@dataclass(frozen=True)
class CohortAssignment:
    calibration_scene_ids: tuple[str, ...]
    confirmation_scene_ids: tuple[str, ...]
    production_scene_ids: tuple[str, ...]


@dataclass(frozen=True, order=True)
class InventoryIssue:
    scene_id: str
    code: str
    detail: str


@dataclass(frozen=True)
class Phase4InventoryReport:
    schema_version: int
    expected_scene_count: int
    status: Literal["ready", "incomplete_cohort", "invalid"]
    scenes: tuple[SceneInventory, ...]
    capacities: tuple[SceneCapacityEstimate, ...]
    cohort: CohortAssignment | None
    issues: tuple[InventoryIssue, ...]
    required_host_ram_bytes: int
    required_artifact_disk_bytes: int


def build_scene_inventory(manifest: SceneManifest) -> SceneInventory:
    centers = np.asarray(manifest.train_camera_to_world[:, :3, 3], dtype=np.float64)
    if len(centers) < 2 or not np.all(np.isfinite(centers)):
        raise DataContractError("scene inventory requires at least two finite cameras")
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    if not np.all(np.isfinite(nearest)) or np.any(nearest <= 0.0):
        raise DataContractError("scene inventory requires distinct camera centers")
    if len(manifest.sparse_points) == 0:
        raise DataContractError("scene inventory requires sparse initialization points")
    distortion_values = [
        abs(value)
        for distortion in manifest.train_distortion
        for value in distortion.coefficients
    ]
    return SceneInventory(
        scene_id=manifest.scene_id,
        train_image_count=len(manifest.train_image_names),
        test_pose_count=len(manifest.test_image_names),
        sparse_point_count=len(manifest.sparse_points),
        trajectory_nn_p90=float(np.percentile(nearest, 90, method="linear")),
        distortion_abs_max=max(distortion_values, default=0.0),
        native_widths=tuple(value.width for value in manifest.train_intrinsics),
        native_heights=tuple(value.height for value in manifest.train_intrinsics),
    )


def estimate_scene_capacity(
    manifest: SceneManifest,
    inventory: SceneInventory,
) -> SceneCapacityEstimate:
    if manifest.scene_id != inventory.scene_id:
        raise DataContractError("manifest and inventory scene IDs differ")
    peak = min(
        MAX_ESTIMATED_GAUSSIANS,
        max(1, inventory.sparse_point_count) * GAUSSIAN_GROWTH_FACTOR,
    )
    cache_bytes = sum(
        intrinsics.width * intrinsics.height * 4
        for intrinsics in manifest.train_intrinsics
    )
    output_raw_bytes = sum(
        intrinsics.width * intrinsics.height * 3
        for intrinsics in manifest.test_intrinsics
    )
    return SceneCapacityEstimate(
        scene_id=inventory.scene_id,
        cache_bytes=cache_bytes,
        output_raw_bytes=output_raw_bytes,
        estimated_peak_gaussians=peak,
        full_checkpoint_bytes=peak * FULL_CHECKPOINT_BYTES_PER_GAUSSIAN,
        compact_model_bytes=peak * COMPACT_MODEL_BYTES_PER_GAUSSIAN,
    )
```

- [ ] **Step 4: Export the public Phase 4.1 types**

Append explicit imports and `__all__` entries in `src/bts_nvs/data/__init__.py`:

```python
from .inventory import (
    CohortAssignment,
    InventoryIssue,
    Phase4InventoryReport,
    SceneCapacityEstimate,
    SceneInventory,
    build_scene_inventory,
    estimate_scene_capacity,
)
```

Add the seven names to the existing `__all__`; do not reorder unrelated exports.

- [ ] **Step 5: Run focused tests to verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_inventory.py tests/unit/test_manifest.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 1**

```powershell
git add src/bts_nvs/data/inventory.py src/bts_nvs/data/__init__.py tests/unit/test_inventory.py
git commit -m "feat: add phase 4 scene inventory primitives"
```

---

### Task 2: Deterministic cohort and capacity aggregation

**Files:**
- Modify: `src/bts_nvs/data/inventory.py`
- Modify: `tests/unit/test_inventory.py`

**Interfaces:**
- Consumes: `tuple[SceneInventory, ...]`, `tuple[SceneCapacityEstimate, ...]`.
- Produces: `select_scene_cohort(inventories, expected_scene_count)`, `aggregate_capacity(capacities)`.

- [ ] **Step 1: Add failing determinism and no-test-feature tests**

Append:

```python
from dataclasses import replace

from bts_nvs.data.inventory import (
    DISK_HEADROOM_BYTES,
    HOST_RAM_HEADROOM_BYTES,
    SceneCapacityEstimate,
    SceneInventory,
    aggregate_capacity,
    select_scene_cohort,
)


def inventory(scene_id: str, x: float) -> SceneInventory:
    return SceneInventory(
        scene_id=scene_id,
        train_image_count=150 + int(x),
        test_pose_count=40 + int(x),
        sparse_point_count=1_000 + int(100 * x),
        trajectory_nn_p90=0.01 + 0.01 * x,
        distortion_abs_max=0.001 * x,
        native_widths=(100,),
        native_heights=(50,),
    )


def test_cohort_is_order_independent_and_ignores_test_count():
    scenes = tuple(inventory(f"scene_{index}", float(index)) for index in range(5))
    first = select_scene_cohort(scenes, expected_scene_count=5)
    changed_test_counts = tuple(
        replace(scene, test_pose_count=999 - index)
        for index, scene in enumerate(reversed(scenes))
    )
    second = select_scene_cohort(changed_test_counts, expected_scene_count=5)

    assert first == second
    assert len(first.calibration_scene_ids) == 3
    assert len(first.confirmation_scene_ids) == 2
    assert first.production_scene_ids == ()


def test_capacity_aggregation_models_sequential_training():
    capacities = (
        SceneCapacityEstimate("a", 100, 10, 1, 1_000, 100),
        SceneCapacityEstimate("b", 200, 20, 1, 2_000, 200),
    )
    host, disk = aggregate_capacity(capacities)

    assert host == 200 + HOST_RAM_HEADROOM_BYTES
    assert disk == (100 + 10 + 200 + 20) + 2 * 2_000 + DISK_HEADROOM_BYTES
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_inventory.py -q
```

Expected: import failures for `select_scene_cohort` and `aggregate_capacity`.

- [ ] **Step 3: Implement exact robust medoid selection**

Append to `inventory.py`:

```python
from itertools import combinations


def _cohort_features(inventories: tuple[SceneInventory, ...]) -> np.ndarray:
    values = np.asarray(
        [
            [
                np.log(scene.train_image_count),
                np.log(scene.sparse_point_count / scene.train_image_count),
                scene.trajectory_nn_p90,
                scene.distortion_abs_max,
            ]
            for scene in inventories
        ],
        dtype=np.float64,
    )
    median = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25, 75], axis=0, method="linear")
    scale = np.where(q75 > q25, q75 - q25, 1.0)
    return (values - median) / scale


def select_scene_cohort(
    inventories: tuple[SceneInventory, ...],
    *,
    expected_scene_count: int = EXPECTED_SCENE_COUNT,
) -> CohortAssignment:
    ordered = tuple(sorted(inventories, key=lambda value: value.scene_id))
    if len(ordered) != expected_scene_count:
        raise DataContractError(
            f"expected {expected_scene_count} scenes, found {len(ordered)}"
        )
    if len(ordered) < 5 or len({item.scene_id for item in ordered}) != len(ordered):
        raise DataContractError("cohort selection requires at least five unique scenes")
    features = _cohort_features(ordered)
    if not np.all(np.isfinite(features)):
        raise DataContractError("cohort features must be finite")
    pairwise = np.abs(features[:, None] - features[None, :]).sum(axis=-1)
    candidates = []
    for indices in combinations(range(len(ordered)), 3):
        cost = float(pairwise[:, indices].min(axis=1).sum())
        ids = tuple(ordered[index].scene_id for index in indices)
        candidates.append((cost, ids, indices))
    _, calibration_ids, calibration_indices = min(
        candidates, key=lambda value: (value[0], value[1])
    )
    remaining = [index for index in range(len(ordered)) if index not in calibration_indices]
    confirmation_rank = sorted(
        remaining,
        key=lambda index: (
            -float(pairwise[index, calibration_indices].min()),
            ordered[index].scene_id,
        ),
    )
    confirmation_ids = tuple(
        sorted(ordered[index].scene_id for index in confirmation_rank[:2])
    )
    selected = set(calibration_ids) | set(confirmation_ids)
    production_ids = tuple(
        item.scene_id for item in ordered if item.scene_id not in selected
    )
    return CohortAssignment(
        calibration_scene_ids=tuple(sorted(calibration_ids)),
        confirmation_scene_ids=confirmation_ids,
        production_scene_ids=production_ids,
    )


def aggregate_capacity(
    capacities: tuple[SceneCapacityEstimate, ...],
) -> tuple[int, int]:
    if not capacities:
        return HOST_RAM_HEADROOM_BYTES, DISK_HEADROOM_BYTES
    required_host = max(item.cache_bytes for item in capacities) + HOST_RAM_HEADROOM_BYTES
    retained = sum(
        item.compact_model_bytes + item.output_raw_bytes for item in capacities
    )
    atomic_active_checkpoint = 2 * max(
        item.full_checkpoint_bytes for item in capacities
    )
    return required_host, retained + atomic_active_checkpoint + DISK_HEADROOM_BYTES
```

- [ ] **Step 4: Run focused and randomized-order tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_inventory.py -q
```

Expected: all tests pass; reversing scene order and changing only test counts does
not change cohort IDs.

- [ ] **Step 5: Commit Task 2**

```powershell
git add src/bts_nvs/data/inventory.py tests/unit/test_inventory.py
git commit -m "feat: add deterministic phase 4 cohort selection"
```

---

### Task 3: Scene audit and deterministic report

**Files:**
- Modify: `src/bts_nvs/data/inventory.py`
- Modify: `tests/unit/test_inventory.py`

**Interfaces:**
- Consumes: `scenes_root`, `manifests_root`, expected scene count.
- Produces: `audit_phase4_inventory(...) -> Phase4InventoryReport`, `save_inventory_report(report, path)`.

- [ ] **Step 1: Add failing incomplete-cohort and byte-determinism tests**

Append tests that create five physical scene directories, expose one valid
manifest through monkeypatch, and require stable issue ordering:

```python
import json

from bts_nvs.data.inventory import audit_phase4_inventory, save_inventory_report


def test_audit_reports_missing_manifests_without_locking_cohort(monkeypatch, tmp_path):
    scenes_root = tmp_path / "scenes"
    manifests_root = tmp_path / "manifests"
    for index in range(5):
        (scenes_root / f"scene_{index}" / "train" / "images").mkdir(parents=True)
    (manifests_root / "scene_0").mkdir(parents=True)
    (manifests_root / "scene_0" / "manifest.json").write_text("{}")
    stub = manifest_stub(sparse_count=10)
    stub.scene_id = "scene_0"
    stub.train_image_names = tuple(f"{i}.JPG" for i in range(150))
    stub.train_intrinsics = tuple([stub.train_intrinsics[0]] * 150)
    stub.train_distortion = tuple([stub.train_distortion[0]] * 150)
    stub.train_camera_to_world = np.repeat(
        np.eye(4, dtype=np.float64)[None], 150, axis=0
    )
    stub.train_camera_to_world[:, 0, 3] = np.arange(150, dtype=np.float64)
    monkeypatch.setattr("bts_nvs.data.inventory.load_scene_manifest", lambda *_: stub)

    report = audit_phase4_inventory(scenes_root, manifests_root, expected_scene_count=13)

    physical_count = sum(
        1 for path in scenes.iterdir()
        if path.is_dir() and (path / "train" / "images").is_dir()
    )
    if physical_count != 13:
        assert report.status == "incomplete_cohort"
        assert report.cohort is None
    assert tuple(issue.scene_id for issue in report.issues if issue.code == "missing_manifest") == (
        "scene_1", "scene_2", "scene_3", "scene_4"
    )


def test_inventory_report_serialization_is_byte_deterministic(monkeypatch, tmp_path):
    scenes_root = tmp_path / "scenes"
    manifests_root = tmp_path / "manifests"
    scenes_root.mkdir()
    report = audit_phase4_inventory(scenes_root, manifests_root, expected_scene_count=13)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    save_inventory_report(report, first)
    save_inventory_report(report, second)

    assert first.read_bytes() == second.read_bytes()
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert "timestamp" not in payload
    assert payload["status"] == "incomplete_cohort"
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_inventory.py -q
```

Expected: imports fail for audit/report functions.

- [ ] **Step 3: Implement discovery, issue policy and report aggregation**

Add imports `json`, `os`, `tempfile`, `Path`, and `load_scene_manifest`. Implement:

```python
def audit_phase4_inventory(
    scenes_root: Path,
    manifests_root: Path,
    *,
    expected_scene_count: int = EXPECTED_SCENE_COUNT,
) -> Phase4InventoryReport:
    scenes_base = Path(scenes_root)
    manifests_base = Path(manifests_root)
    roots = tuple(
        sorted(
            (
                path for path in scenes_base.iterdir()
                if path.is_dir() and (path / "train" / "images").is_dir()
            ),
            key=lambda path: path.name,
        )
    )
    inventories = []
    capacities = []
    issues = []
    if len(roots) != expected_scene_count:
        issues.append(InventoryIssue(
            "*", "scene_count",
            f"expected {expected_scene_count} scenes, found {len(roots)}",
        ))
    for root in roots:
        manifest_json = manifests_base / root.name / "manifest.json"
        if not manifest_json.is_file():
            issues.append(InventoryIssue(root.name, "missing_manifest", "manifest.json is missing"))
            continue
        try:
            manifest = load_scene_manifest(manifest_json, root)
            inventory = build_scene_inventory(manifest)
            if inventory.scene_id != root.name:
                raise DataContractError("manifest scene_id does not match directory name")
            if inventory.train_image_count < MIN_TRAIN_IMAGES:
                issues.append(InventoryIssue(
                    root.name, "insufficient_train_images",
                    f"requires at least {MIN_TRAIN_IMAGES}, found {inventory.train_image_count}",
                ))
            inventories.append(inventory)
            capacities.append(estimate_scene_capacity(manifest, inventory))
        except (DataContractError, OSError, ValueError) as error:
            detail = str(error)
            for base, marker in (
                (scenes_base, "<scenes_root>"),
                (manifests_base, "<manifests_root>"),
            ):
                detail = detail.replace(str(base), marker)
                detail = detail.replace(str(base.resolve()), marker)
            issues.append(InventoryIssue(root.name, "invalid_scene", detail))
    ordered_inventories = tuple(sorted(inventories, key=lambda item: item.scene_id))
    ordered_capacities = tuple(sorted(capacities, key=lambda item: item.scene_id))
    invalid = any(issue.code not in {"scene_count", "missing_manifest"} for issue in issues)
    ready = not issues and len(ordered_inventories) == expected_scene_count
    status = "ready" if ready else ("invalid" if invalid else "incomplete_cohort")
    cohort = (
        select_scene_cohort(ordered_inventories, expected_scene_count=expected_scene_count)
        if ready else None
    )
    host, disk = aggregate_capacity(ordered_capacities)
    return Phase4InventoryReport(
        schema_version=SCHEMA_VERSION,
        expected_scene_count=expected_scene_count,
        status=status,
        scenes=ordered_inventories,
        capacities=ordered_capacities,
        cohort=cohort,
        issues=tuple(sorted(issues)),
        required_host_ram_bytes=host,
        required_artifact_disk_bytes=disk,
    )
```

- [ ] **Step 4: Implement canonical JSON and atomic replace**

Add a private recursive encoder for dataclasses/tuples, include a top-level
`capacity_assumptions` object, and save exactly one trailing newline:

```python
def _report_payload(report: Phase4InventoryReport) -> dict[str, object]:
    payload = asdict(report)
    payload["capacity_assumptions"] = {
        "compact_model_bytes_per_gaussian": COMPACT_MODEL_BYTES_PER_GAUSSIAN,
        "disk_headroom_bytes": DISK_HEADROOM_BYTES,
        "full_checkpoint_bytes_per_gaussian": FULL_CHECKPOINT_BYTES_PER_GAUSSIAN,
        "gaussian_growth_factor": GAUSSIAN_GROWTH_FACTOR,
        "host_ram_headroom_bytes": HOST_RAM_HEADROOM_BYTES,
        "max_estimated_gaussians": MAX_ESTIMATED_GAUSSIANS,
    }
    return payload


def save_inventory_report(report: Phase4InventoryReport, path: Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        _report_payload(report), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
```

- [ ] **Step 5: Run unit suite and inspect a generated JSON**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_inventory.py -q
.\.venv\Scripts\python.exe -c "from pathlib import Path; from bts_nvs.data.inventory import audit_phase4_inventory, save_inventory_report; r=audit_phase4_inventory(Path('data/phase1/public_set'), Path('runs/manifests')); save_inventory_report(r, Path('runs/phase4/inventory.plan-smoke.json')); print(r.status, len(r.scenes), len(r.issues))"
```

Expected locally: tests pass; smoke reports `incomplete_cohort`, discovers five
physical scenes, loads HCM0181 if its manifest exists, and never locks a cohort.
Delete only the generated `runs/phase4/inventory.plan-smoke.json` after inspecting
it; do not touch user archives or existing run artifacts.

- [ ] **Step 6: Commit Task 3**

```powershell
git add src/bts_nvs/data/inventory.py tests/unit/test_inventory.py
git commit -m "feat: add deterministic phase 4 inventory report"
```

---

### Task 4: Feasibility CLI and real-data gate

**Files:**
- Create: `src/bts_nvs/data/run_inventory.py`
- Create: `tests/integration/test_phase4_inventory.py`
- Modify: `tests/unit/test_inventory.py`

**Interfaces:**
- Consumes: `audit_phase4_inventory`, `save_inventory_report`.
- Produces: `main(argv: list[str] | None = None) -> int`, CLI exit codes 0/2/3.

Exit policy:

- `0`: report written and local capacity passes; incomplete cohort is allowed
  unless `--require_ready` is present.
- `2`: `--require_ready` requested but report status is not `ready`.
- `3`: host RAM or artifact disk is below the deterministic estimate.

- [ ] **Step 1: Add failing CLI capacity tests**

Append to `tests/unit/test_inventory.py`:

```python
from bts_nvs.data.run_inventory import check_local_feasibility


def test_local_feasibility_requires_reported_headroom():
    report = Phase4InventoryReport(
        schema_version=1,
        expected_scene_count=13,
        status="incomplete_cohort",
        scenes=(), capacities=(), cohort=None, issues=(),
        required_host_ram_bytes=100,
        required_artifact_disk_bytes=200,
    )
    assert check_local_feasibility(report, available_ram_bytes=100, disk_free_bytes=200) == ()
    assert check_local_feasibility(report, available_ram_bytes=99, disk_free_bytes=200) == (
        "available host RAM 99 is below required 100",
    )
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_inventory.py -q
```

Expected: `bts_nvs.data.run_inventory` import fails.

- [ ] **Step 3: Implement the thin CLI without new dependencies**

Create `src/bts_nvs/data/run_inventory.py`:

```python
from __future__ import annotations

import argparse
import ctypes
import os
import shutil
from pathlib import Path

from .inventory import audit_phase4_inventory, save_inventory_report


def available_ram_bytes() -> int:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended_virtual", ctypes.c_ulonglong),
            ]
        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.avail_phys)
    raise RuntimeError("cannot determine available host RAM")


def check_local_feasibility(
    report,
    *,
    available_ram_bytes: int,
    disk_free_bytes: int,
) -> tuple[str, ...]:
    errors = []
    if available_ram_bytes < report.required_host_ram_bytes:
        errors.append(
            f"available host RAM {available_ram_bytes} is below required "
            f"{report.required_host_ram_bytes}"
        )
    if disk_free_bytes < report.required_artifact_disk_bytes:
        errors.append(
            f"artifact disk free {disk_free_bytes} is below required "
            f"{report.required_artifact_disk_bytes}"
        )
    return tuple(errors)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Phase 4 scene inventory")
    parser.add_argument("--scenes_root", type=Path, required=True)
    parser.add_argument("--manifests_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected_scenes", type=int, default=13)
    parser.add_argument("--require_ready", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_phase4_inventory(
        args.scenes_root, args.manifests_root,
        expected_scene_count=args.expected_scenes,
    )
    save_inventory_report(report, args.output)
    capacity_errors = check_local_feasibility(
        report,
        available_ram_bytes=available_ram_bytes(),
        disk_free_bytes=shutil.disk_usage(args.output.parent).free,
    )
    print(
        f"status={report.status} scenes={len(report.scenes)}/"
        f"{report.expected_scene_count} issues={len(report.issues)}"
    )
    for error in capacity_errors:
        print(f"capacity_error: {error}")
    if capacity_errors:
        return 3
    if args.require_ready and report.status != "ready":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add real-data smoke without requiring CUDA**

Create `tests/integration/test_phase4_inventory.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from bts_nvs.data.inventory import audit_phase4_inventory, build_scene_inventory
from bts_nvs.data.manifest import load_scene_manifest


pytestmark = pytest.mark.real_data


def test_hcm0181_inventory_and_public_set_discovery():
    repo = Path(__file__).resolve().parents[2]
    scenes = repo / "data" / "phase1" / "public_set"
    manifests = repo / "runs" / "manifests"
    manifest_json = manifests / "HCM0181" / "manifest.json"
    scene_root = scenes / "HCM0181"
    if not scene_root.is_dir() or not manifest_json.is_file():
        pytest.skip("HCM0181 data/manifest is not available")

    inventory = build_scene_inventory(load_scene_manifest(manifest_json, scene_root))
    report = audit_phase4_inventory(scenes, manifests, expected_scene_count=13)

    assert inventory.train_image_count == 240
    assert inventory.test_pose_count == 60
    assert inventory.sparse_point_count > 0
    assert np.isfinite(inventory.trajectory_nn_p90)
    assert inventory.trajectory_nn_p90 > 0.0
    assert report.status == "incomplete_cohort"
    assert report.cohort is None
```

- [ ] **Step 5: Run Phase 4.1 tests and CLI smoke**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit/test_inventory.py tests/integration/test_phase4_inventory.py -q
.\.venv\Scripts\python.exe -m bts_nvs.data.run_inventory --scenes_root data/phase1/public_set --manifests_root runs/manifests --output runs/phase4/inventory.json
```

Expected locally:

- tests pass (real-data test may skip only if HCM0181 artifact is absent);
- CLI writes `runs/phase4/inventory.json`;
- status is `incomplete_cohort`, because only five physical scenes and one
  manifest are currently local;
- cohort is null;
- no CUDA initialization occurs.

- [ ] **Step 6: Run regression suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/unit -q
```

Expected: all unit tests pass. CUDA/LPIPS smoke tests are not part of Phase 4.1.

- [ ] **Step 7: Red review the completed Phase 4.1 diff**

Verify all of the following before the final commit:

```powershell
rg -n "test_image_names|test_world_to_camera|test_output_names" src/bts_nvs/data/inventory.py
git diff --check
git status --short
```

Expected:

- test values/names appear only in count/output-capacity code, never
  `_cohort_features` or `select_scene_cohort`;
- no trainer/dataset/rendering file changed;
- user archive `pilot_factor1_7000.tar.gz` remains untracked and untouched;
- diff check is clean.

- [ ] **Step 8: Commit Task 4 and stop the session**

```powershell
git add src/bts_nvs/data/run_inventory.py tests/integration/test_phase4_inventory.py tests/unit/test_inventory.py
git commit -m "feat: add phase 4 inventory feasibility gate"
```

Do not start Phase 4.2. Report actual test counts, local inventory status,
capacity estimates, commit IDs and any unresolved scene-manifest gaps.

---

## Plan self-review checklist

- Phase 4.1 only: no holdout, cache, trainer or renderer implementation.
- Every Phase 4.1 spec acceptance criterion maps to a task/test above.
- Cohort assignment is impossible before exactly 13 valid scenes.
- Test metadata cannot affect cohort features.
- Capacity JSON is deterministic; machine availability is console-only.
- Sequential one-L4 disk math keeps one doubled atomic checkpoint plus all
  retained compact models/outputs.
- No dependency addition, CUDA requirement or background process.
- Every task has a RED command, minimal GREEN implementation, verification and
  commit boundary.
