"""Deterministic shield placement with truth-only pose error."""

from __future__ import annotations

import numpy as np

from radcounter.core.actions.context import CountermeasureExecutionContext
from radcounter.core.actions.robot import RobotController
from radcounter.core.models.actions import (
    ActionResult,
    ActionStatus,
    ActionType,
    CountermeasureAction,
)


def _pose_error_transform(
    rng: np.random.Generator, translation_std_m: float, rotation_std_deg: float
) -> np.ndarray:
    translation = rng.normal(0.0, translation_std_m, size=3)
    angles = np.deg2rad(rng.normal(0.0, rotation_std_deg, size=3))
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)
    rotation_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rotation_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rotation_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    result = np.eye(4)
    result[:3, :3] = rotation_z @ rotation_y @ rotation_x
    result[:3, 3] = translation
    return result


class ShieldPlacementExecutor:
    """Execute PLACE_SHIELD or MOVE_SHIELD through a shared lifecycle."""

    async def execute(
        self,
        action: CountermeasureAction,
        robot: RobotController,
        context: CountermeasureExecutionContext,
    ) -> ActionResult:
        before = context.truth_state.revision.copy()
        started = context.sim_time_s
        if (
            action.action_type not in {ActionType.PLACE_SHIELD, ActionType.MOVE_SHIELD}
            or action.target_prim_path is None
            or action.target_pose_world is None
        ):
            return self._terminal(
                action, context, before, ActionStatus.REJECTED, "invalid shield action"
            )
        if not context.resources.can_afford(action.resource_cost):
            return self._terminal(
                action, context, before, ActionStatus.REJECTED, "insufficient resources"
            )
        shield_path = action.target_prim_path
        current_pose = context.shield_poses_world.get(shield_path, np.eye(4))
        lifecycle: list[str] = ["PLAN"]
        for state, operation in (
            ("NAVIGATE_TO_SHIELD", robot.navigate_to(current_pose, 30.0)),
            ("GRASP", robot.grasp(shield_path)),
            ("NAVIGATE_TO_TARGET", robot.navigate_to(action.target_pose_world, 30.0)),
            ("RELEASE", robot.release()),
        ):
            result = await operation
            lifecycle.append(state)
            if not result.succeeded:
                return self._terminal(action, context, before, ActionStatus.FAILED, result.message)
        translation_std_m = float(action.parameters.get("translation_error_std_m", 0.0))
        rotation_std_deg = float(action.parameters.get("rotation_error_std_deg", 0.0))
        if min(translation_std_m, rotation_std_deg) < 0:
            return self._terminal(
                action, context, before, ActionStatus.REJECTED, "negative pose error"
            )
        actual_pose = action.target_pose_world @ _pose_error_transform(
            context.rng, translation_std_m, rotation_std_deg
        )
        context.shield_poses_world[shield_path] = actual_pose
        context.resources.consume(action.resource_cost)
        context.truth_state.revision.bump_geometry()
        context.advance(action.predicted_duration_s)
        lifecycle.extend(["WAIT_SETTLE", "COMMIT_RADIATION_SCENE", "COMPLETE"])
        return ActionResult(
            action.action_id,
            ActionStatus.COMPLETED,
            started,
            context.sim_time_s,
            {"shield_prim_path": shield_path, "lifecycle": lifecycle},
            {"actual_pose_world": actual_pose.tolist()},
            before,
            context.truth_state.revision.copy(),
        )

    @staticmethod
    def _terminal(
        action: CountermeasureAction,
        context: CountermeasureExecutionContext,
        before: object,
        status: ActionStatus,
        message: str,
    ) -> ActionResult:
        return ActionResult(
            action.action_id,
            status,
            context.sim_time_s,
            context.sim_time_s,
            {"message": message},
            None,
            before,  # type: ignore[arg-type]
            context.truth_state.revision.copy(),
        )
