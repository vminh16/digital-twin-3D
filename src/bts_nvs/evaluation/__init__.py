"""Deterministic Phase 2 benchmark contracts."""

from .evaluator import EvaluationError, evaluate_benchmark, load_image_pairs, save_metric_report
from .metrics import LpipsBackend, MetricConfig, evaluate_image

__all__ = [
    "EvaluationError",
    "LpipsBackend",
    "MetricConfig",
    "evaluate_benchmark",
    "evaluate_image",
    "load_image_pairs",
    "save_metric_report",
]

