"""Countermeasure action and result contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from radcounter.core.models.state import RevisionState


class ActionType(StrEnum):
    """Supported measurement and countermeasure choices."""

    MEASURE = "measure"
    DECONTAMINATE = "decontaminate"
    PLACE_SHIELD = "place_shield"
    MOVE_SHIELD = "move_shield"
    MOVE_OBJECT = "move_object"
    REMOVE_OBJECT = "remove_object"
    REPAIR_ACTION = "repair_action"


class ActionStatus(StrEnum):
    """Terminal action states."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass(frozen=True)
class CountermeasureAction:
    """Planner-visible action request with unit-bearing costs."""

    action_id: str
    action_type: ActionType
    robot_id: str
    target_prim_path: str | None = None
    target_region: Mapping[str, Any] | None = None
    target_pose_world: NDArray[np.float64] | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    predicted_duration_s: float = 0.0
    resource_cost: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.action_id or not self.robot_id or self.predicted_duration_s < 0:
            raise ValueError("action id/robot are required and duration must be nonnegative")
        if any(value < 0 for value in self.resource_cost.values()):
            raise ValueError("resource costs must be nonnegative")
        if self.target_pose_world is not None:
            pose = np.asarray(self.target_pose_world, dtype=np.float64)
            if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
                raise ValueError("target_pose_world must be a finite 4x4 matrix")
            object.__setattr__(self, "target_pose_world", pose)


@dataclass(frozen=True)
class ActionResult:
    """Execution result with a strict public/truth detail split."""

    action_id: str
    status: ActionStatus
    started_sim_s: float
    completed_sim_s: float
    public_details: Mapping[str, Any]
    truth_details: Mapping[str, Any] | None
    before_revision: RevisionState
    after_revision: RevisionState

    def public_view(self) -> ActionResult:
        """Return the only form allowed to estimator/planner code."""

        return ActionResult(
            action_id=self.action_id,
            status=self.status,
            started_sim_s=self.started_sim_s,
            completed_sim_s=self.completed_sim_s,
            public_details=self.public_details,
            truth_details=None,
            before_revision=self.before_revision.copy(),
            after_revision=self.after_revision.copy(),
        )
