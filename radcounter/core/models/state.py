"""Truth/belief state boundary and independent revision counters."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from radcounter.core.models.radiation import PointSourceState, SurfaceSourceState


@dataclass
class RevisionState:
    """Independent invalidation counters for radiation calculations."""

    geometry_revision: int = 0
    material_revision: int = 0
    source_pose_revision: int = 0
    source_activity_revision: int = 0
    detector_revision: int = 0

    def copy(self) -> RevisionState:
        """Return an independent revision value."""

        return RevisionState(**vars(self))

    def bump_geometry(self) -> None:
        self.geometry_revision += 1

    def bump_material(self) -> None:
        self.material_revision += 1

    def bump_source_pose(self, *, geometry_changed: bool = False) -> None:
        self.source_pose_revision += 1
        if geometry_changed:
            self.geometry_revision += 1

    def bump_source_activity(self) -> None:
        self.source_activity_revision += 1

    def bump_detector(self) -> None:
        self.detector_revision += 1


@dataclass
class TruthState:
    """Simulator-private state; never accepted by estimator/planner APIs."""

    point_sources: dict[str, PointSourceState] = field(default_factory=dict)
    surface_sources: dict[str, SurfaceSourceState] = field(default_factory=dict)
    revision: RevisionState = field(default_factory=RevisionState)
    hidden_source_ids: set[str] = field(default_factory=set)
    detector_gain: float = 1.0
    detector_background_offset_cps: float = 0.0


@dataclass(frozen=True)
class BeliefState:
    """State visible to estimation and planning."""

    basis_ids: tuple[str, ...]
    source_strength_bq: NDArray[np.float64]
    covariance: NDArray[np.float64]
    revision: RevisionState
    remaining_resources: dict[str, float] = field(default_factory=dict)
    action_effect_parameters: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        strength = np.asarray(self.source_strength_bq, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if strength.shape != (len(self.basis_ids),):
            raise ValueError("source_strength_bq must match basis_ids")
        if covariance.shape != (len(self.basis_ids), len(self.basis_ids)):
            raise ValueError("covariance must be square over basis_ids")
        if np.any(strength < 0) or not np.all(np.isfinite(covariance)):
            raise ValueError("belief strength/covariance values are invalid")
        object.__setattr__(self, "source_strength_bq", strength)
        object.__setattr__(self, "covariance", covariance)
