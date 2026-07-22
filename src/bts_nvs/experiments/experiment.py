from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType

from bts_nvs.experiments.candidates import candidate_settings


COHORT_SCENE_IDS = (
    "HCM0644",
    "HCM0674",
    "HCM0540",
    "HCM0539",
    "HCM0421",
    "chair",
    "bonsai",
)
MAX_PEAK_VRAM_MB = 23 * 1024
MAX_PAIRED_WALL_TIME_RATIO = 1.25


class ExperimentStage(str, Enum):
    REFERENCE = "reference"
    SCREEN = "screen"
    CONFIRM = "confirm"
    PRODUCTION = "production"


STAGE_HORIZONS = MappingProxyType(
    {
        ExperimentStage.REFERENCE: 7_000,
        ExperimentStage.SCREEN: 7_000,
        ExperimentStage.CONFIRM: 30_000,
        ExperimentStage.PRODUCTION: 30_000,
    }
)


@dataclass(frozen=True)
class Experiment:
    stage: ExperimentStage
    scene_id: str
    candidate_id: str
    authorized_scene_winner: str | None = None
    authorized_cohort_candidate: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, ExperimentStage):
            raise ValueError("stage must be an ExperimentStage")
        if self.scene_id not in COHORT_SCENE_IDS:
            raise ValueError("scene_id must be in the locked cohort")

        candidate_settings(self.candidate_id)
        self._validate_authorization(
            "authorized_scene_winner", self.authorized_scene_winner
        )
        self._validate_authorization(
            "authorized_cohort_candidate", self.authorized_cohort_candidate
        )

        if self.stage is ExperimentStage.REFERENCE:
            if self.candidate_id != "B0-reference":
                raise ValueError("reference requires B0-reference")
        elif self.stage is ExperimentStage.SCREEN:
            if self.candidate_id == "B0-reference":
                raise ValueError("screen requires a non-B0 candidate")
        elif self.stage is ExperimentStage.CONFIRM:
            if (
                self.candidate_id != "B0-reference"
                and self.candidate_id != self.authorized_scene_winner
            ):
                raise ValueError(
                    "confirm requires candidate_id to match authorized_scene_winner"
                )
        elif self.candidate_id != self.authorized_cohort_candidate:
            raise ValueError(
                "production requires candidate_id to match authorized_cohort_candidate"
            )

    @property
    def horizon(self) -> int:
        return STAGE_HORIZONS[self.stage]

    def run_path(self, root: str | Path) -> Path:
        path = Path(root) / self.stage.value / self.scene_id
        if self.stage is ExperimentStage.REFERENCE:
            return path
        return path / self.candidate_id

    @staticmethod
    def _validate_authorization(name: str, candidate_id: str | None) -> None:
        if candidate_id is None:
            return
        try:
            candidate_settings(candidate_id)
        except ValueError as error:
            raise ValueError(f"{name} must name a registered candidate") from error
