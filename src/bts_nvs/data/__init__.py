"""Scene source-data contracts and readers."""

from .manifest import (
    SceneManifest,
    SceneSourceData,
    TestPoseRecord,
    build_scene_manifest,
    load_scene_manifest,
    load_scene_source_data,
    save_scene_manifest,
    validate_scene_manifest,
)
from .validation import DataContractError
from .dataset import CameraSample, SceneDataset
from .diagnostics import SceneDiagnostics, build_scene_diagnostics
from .inventory import (
    CohortAssignment,
    InventoryIssue,
    Phase4InventoryReport,
    SceneCapacityEstimate,
    SceneInventory,
    aggregate_capacity,
    audit_phase4_inventory,
    build_scene_inventory,
    estimate_scene_capacity,
    save_phase4_inventory_report,
    select_scene_cohort,
)
from .holdout import (
    HoldoutSplit,
    build_pose_holdout,
    load_holdout_split,
    manifest_holdout_sha256,
    manifest_pose_distance_matrix,
    save_holdout_split,
    validate_holdout_split,
)
from .sparse_subset import SparseInitialization, build_split_sparse_initialization

__all__ = [
    "DataContractError",
    "CameraSample",
    "CohortAssignment",
    "InventoryIssue",
    "HoldoutSplit",
    "Phase4InventoryReport",
    "SceneDataset",
    "SceneDiagnostics",
    "SceneManifest",
    "SceneCapacityEstimate",
    "SceneInventory",
    "SceneSourceData",
    "SparseInitialization",
    "TestPoseRecord",
    "build_scene_manifest",
    "build_pose_holdout",
    "build_scene_inventory",
    "build_split_sparse_initialization",
    "build_scene_diagnostics",
    "aggregate_capacity",
    "audit_phase4_inventory",
    "load_scene_manifest",
    "load_holdout_split",
    "load_scene_source_data",
    "save_scene_manifest",
    "save_holdout_split",
    "estimate_scene_capacity",
    "save_phase4_inventory_report",
    "select_scene_cohort",
    "manifest_holdout_sha256",
    "manifest_pose_distance_matrix",
    "validate_holdout_split",
    "validate_scene_manifest",
]
