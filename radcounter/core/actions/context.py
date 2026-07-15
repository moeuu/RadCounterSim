"""Simulator-private state used by deterministic action executors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from radcounter.core.actions.resources import ResourceState
from radcounter.core.models.state import TruthState

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DisposalZone:
    """Validated destination required before source deactivation."""

    zone_id: str
    center_world_m: FloatArray
    radius_m: float
    disposition: Literal["retain_in_scene", "deactivate_outside"] = "retain_in_scene"

    def __post_init__(self) -> None:
        center = np.asarray(self.center_world_m, dtype=np.float64)
        if center.shape != (3,) or self.radius_m <= 0:
            raise ValueError("disposal zone requires a 3-D center and positive radius")
        object.__setattr__(self, "center_world_m", center)

    def contains(self, position_world_m: FloatArray) -> bool:
        """Return whether a world position is within the zone."""

        return bool(
            np.linalg.norm(np.asarray(position_world_m) - self.center_world_m) <= self.radius_m
        )


@dataclass
class CountermeasureExecutionContext:
    """Truth-only mutable world state; never passed to planners."""

    truth_state: TruthState
    resources: ResourceState
    rng: np.random.Generator
    sim_time_s: float = 0.0
    object_poses_world: dict[str, FloatArray] = field(default_factory=dict)
    movable_objects: set[str] = field(default_factory=set)
    removable_objects: set[str] = field(default_factory=set)
    shield_poses_world: dict[str, FloatArray] = field(default_factory=dict)
    disposal_zones: dict[str, DisposalZone] = field(default_factory=dict)
    decon_exposure_s_by_source: dict[str, FloatArray] = field(default_factory=dict)
    decon_efficiency_by_source: dict[str, FloatArray] = field(default_factory=dict)

    def advance(self, duration_s: float) -> None:
        """Advance deterministic simulation time."""

        if duration_s < 0:
            raise ValueError("duration_s must be nonnegative")
        self.sim_time_s += duration_s


def transform_point(pose_world: FloatArray, point_world_m: FloatArray) -> FloatArray:
    """Apply a homogeneous transform to one point."""

    point = np.append(np.asarray(point_world_m, dtype=np.float64), 1.0)
    return (np.asarray(pose_world, dtype=np.float64) @ point)[:3]
