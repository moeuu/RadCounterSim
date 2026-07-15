"""Deterministic decontamination with truth-side effectiveness mismatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from radcounter.core.actions.context import CountermeasureExecutionContext
from radcounter.core.actions.robot import RobotController
from radcounter.core.models.actions import (
    ActionResult,
    ActionStatus,
    ActionType,
    CountermeasureAction,
)


@dataclass(frozen=True)
class DeconToolSpec:
    """Contact and exponential treatment model parameters."""

    footprint_sample_points_local_m: np.ndarray
    treatment_axis_local: np.ndarray
    max_contact_distance_m: float
    max_normal_angle_deg: float
    max_surface_speed_m_s: float
    rate_constant_s_inv: float

    def __post_init__(self) -> None:
        footprint = np.asarray(self.footprint_sample_points_local_m, dtype=np.float64)
        axis = np.asarray(self.treatment_axis_local, dtype=np.float64)
        if footprint.ndim != 2 or footprint.shape[1:] != (3,) or axis.shape != (3,):
            raise ValueError("decon footprint/axis dimensions are invalid")
        if (
            min(
                self.max_contact_distance_m,
                self.max_normal_angle_deg,
                self.max_surface_speed_m_s,
                self.rate_constant_s_inv,
            )
            <= 0
        ):
            raise ValueError("decon tool limits/rate must be positive")
        object.__setattr__(self, "footprint_sample_points_local_m", footprint)
        object.__setattr__(self, "treatment_axis_local", axis / np.linalg.norm(axis))


class DecontaminationExecutor:
    """Apply triangle exposure without changing geometry revisions."""

    async def execute(
        self,
        action: CountermeasureAction,
        robot: RobotController,
        context: CountermeasureExecutionContext,
    ) -> ActionResult:
        """Execute a deterministic decontamination request."""

        before = context.truth_state.revision.copy()
        started = context.sim_time_s
        if action.action_type != ActionType.DECONTAMINATE or action.target_region is None:
            return self._rejected(action, context, before, "invalid decontamination action")
        source_id = str(action.target_region.get("source_id", ""))
        source = context.truth_state.surface_sources.get(source_id)
        if source is None or not source.enabled:
            return self._rejected(action, context, before, "target surface source is unavailable")
        if not context.resources.can_afford(action.resource_cost):
            return self._rejected(action, context, before, "insufficient resources")
        if action.target_pose_world is not None:
            movement = await robot.move_end_effector(
                action.target_pose_world, max(action.predicted_duration_s, 1.0)
            )
            if not movement.succeeded:
                return self._failed(action, context, before, movement.message)
        triangle_ids = np.asarray(action.target_region.get("triangle_indices", []), dtype=np.int64)
        exposure_increment_s = np.asarray(
            action.target_region.get("exposure_s", []), dtype=np.float64
        )
        if triangle_ids.ndim != 1 or exposure_increment_s.shape != triangle_ids.shape:
            return self._rejected(action, context, before, "triangle exposure shape mismatch")
        if np.any(triangle_ids < 0) or np.any(triangle_ids >= len(source.activity_bq_per_triangle)):
            return self._rejected(action, context, before, "triangle index out of range")
        if np.any(exposure_increment_s < 0):
            return self._rejected(action, context, before, "negative treatment exposure")
        rate_constant_s_inv = float(action.parameters.get("rate_constant_s_inv", 1.0))
        if rate_constant_s_inv <= 0:
            return self._rejected(action, context, before, "invalid decon rate constant")
        transfer_mode: Literal["discard", "transfer_to_waste"] = str(
            action.parameters.get("removed_activity_mode", "transfer_to_waste")
        )  # type: ignore[assignment]
        if transfer_mode not in {"discard", "transfer_to_waste"}:
            return self._rejected(action, context, before, "invalid removed activity mode")
        waste_source_id = action.parameters.get("waste_source_id")
        if (
            transfer_mode == "transfer_to_waste"
            and context.truth_state.point_sources.get(str(waste_source_id)) is None
        ):
            return self._rejected(action, context, before, "waste source is unavailable")
        exposure = context.decon_exposure_s_by_source.setdefault(
            source_id, np.zeros_like(source.activity_bq_per_triangle)
        )
        efficiency = context.decon_efficiency_by_source.setdefault(
            source_id, np.ones_like(source.activity_bq_per_triangle)
        )
        nominal_increment = 1.0 - np.exp(-rate_constant_s_inv * exposure_increment_s)
        actual_increment = np.clip(nominal_increment * efficiency[triangle_ids], 0.0, 1.0)
        before_activity = source.activity_bq_per_triangle[triangle_ids].copy()
        removed_activity = before_activity * actual_increment
        source.activity_bq_per_triangle[triangle_ids] -= removed_activity
        exposure[triangle_ids] += exposure_increment_s
        if transfer_mode == "transfer_to_waste":
            waste = context.truth_state.point_sources.get(str(waste_source_id))
            assert waste is not None
            waste.activity_bq += float(removed_activity.sum())
        context.resources.consume(action.resource_cost)
        context.truth_state.revision.bump_source_activity()
        context.advance(action.predicted_duration_s)
        return ActionResult(
            action.action_id,
            ActionStatus.COMPLETED,
            started,
            context.sim_time_s,
            {
                "target_source_id": source_id,
                "treated_triangle_count": len(triangle_ids),
                "removed_activity_mode": transfer_mode,
            },
            {
                "actual_removed_activity_bq": float(removed_activity.sum()),
                "actual_removal_fraction": actual_increment.tolist(),
            },
            before,
            context.truth_state.revision.copy(),
        )

    @staticmethod
    def _rejected(
        action: CountermeasureAction,
        context: CountermeasureExecutionContext,
        before: object,
        message: str,
    ) -> ActionResult:
        return ActionResult(
            action.action_id,
            ActionStatus.REJECTED,
            context.sim_time_s,
            context.sim_time_s,
            {"message": message},
            None,
            before,  # type: ignore[arg-type]
            context.truth_state.revision.copy(),
        )

    @staticmethod
    def _failed(
        action: CountermeasureAction,
        context: CountermeasureExecutionContext,
        before: object,
        message: str,
    ) -> ActionResult:
        return ActionResult(
            action.action_id,
            ActionStatus.FAILED,
            context.sim_time_s,
            context.sim_time_s,
            {"message": message},
            None,
            before,  # type: ignore[arg-type]
            context.truth_state.revision.copy(),
        )
