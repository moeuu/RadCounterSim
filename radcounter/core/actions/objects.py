"""Move and disposal-validated removal of scene objects."""

from __future__ import annotations

import numpy as np

from radcounter.core.actions.context import CountermeasureExecutionContext, transform_point
from radcounter.core.actions.robot import RobotController
from radcounter.core.models.actions import (
    ActionResult,
    ActionStatus,
    ActionType,
    CountermeasureAction,
)


async def _move_robot_sequence(
    target_prim_path: str,
    old_pose: np.ndarray,
    new_pose: np.ndarray,
    robot: RobotController,
) -> tuple[bool, str]:
    for operation in (
        robot.navigate_to(old_pose, 30.0),
        robot.grasp(target_prim_path),
        robot.navigate_to(new_pose, 30.0),
        robot.release(),
    ):
        result = await operation
        if not result.succeeded:
            return False, result.message
    return True, ""


def _update_attached_point_sources(
    target_prim_path: str,
    old_pose: np.ndarray,
    new_pose: np.ndarray,
    context: CountermeasureExecutionContext,
) -> bool:
    delta = new_pose @ np.linalg.inv(old_pose)
    moved_source = False
    for source in context.truth_state.point_sources.values():
        if source.attached_prim_path == target_prim_path:
            source.position_world_m = transform_point(delta, source.position_world_m)
            moved_source = True
    return moved_source


class MoveObjectExecutor:
    """Move a validated object and all attached point sources."""

    async def execute(
        self,
        action: CountermeasureAction,
        robot: RobotController,
        context: CountermeasureExecutionContext,
    ) -> ActionResult:
        before = context.truth_state.revision.copy()
        started = context.sim_time_s
        target = action.target_prim_path
        if (
            action.action_type != ActionType.MOVE_OBJECT
            or target is None
            or action.target_pose_world is None
            or target not in context.movable_objects
            or target not in context.object_poses_world
        ):
            return self._terminal(
                action, context, before, ActionStatus.REJECTED, "object is not movable"
            )
        if not context.resources.can_afford(action.resource_cost):
            return self._terminal(
                action, context, before, ActionStatus.REJECTED, "insufficient resources"
            )
        old_pose = context.object_poses_world[target]
        success, message = await _move_robot_sequence(
            target, old_pose, action.target_pose_world, robot
        )
        if not success:
            return self._terminal(action, context, before, ActionStatus.FAILED, message)
        moved_source = _update_attached_point_sources(
            target, old_pose, action.target_pose_world, context
        )
        context.object_poses_world[target] = action.target_pose_world.copy()
        context.resources.consume(action.resource_cost)
        context.truth_state.revision.bump_source_pose(geometry_changed=True)
        context.advance(action.predicted_duration_s)
        return ActionResult(
            action.action_id,
            ActionStatus.COMPLETED,
            started,
            context.sim_time_s,
            {"target_prim_path": target, "attached_source_pose_updated": moved_source},
            {"actual_pose_world": action.target_pose_world.tolist()},
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


class RemoveObjectExecutor:
    """Remove an object only after its actual pose reaches a disposal zone."""

    async def execute(
        self,
        action: CountermeasureAction,
        robot: RobotController,
        context: CountermeasureExecutionContext,
    ) -> ActionResult:
        before = context.truth_state.revision.copy()
        started = context.sim_time_s
        target = action.target_prim_path
        zone_id = str(action.parameters.get("disposal_zone_id", ""))
        zone = context.disposal_zones.get(zone_id)
        if (
            action.action_type != ActionType.REMOVE_OBJECT
            or target is None
            or action.target_pose_world is None
            or target not in context.removable_objects
            or target not in context.object_poses_world
            or zone is None
        ):
            return self._terminal(action, context, before, ActionStatus.REJECTED, "invalid removal")
        if not zone.contains(action.target_pose_world[:3, 3]):
            return self._terminal(
                action,
                context,
                before,
                ActionStatus.REJECTED,
                "target pose is outside disposal zone",
            )
        if not context.resources.can_afford(action.resource_cost):
            return self._terminal(
                action, context, before, ActionStatus.REJECTED, "insufficient resources"
            )
        old_pose = context.object_poses_world[target]
        success, message = await _move_robot_sequence(
            target, old_pose, action.target_pose_world, robot
        )
        if not success:
            return self._terminal(action, context, before, ActionStatus.FAILED, message)
        _update_attached_point_sources(target, old_pose, action.target_pose_world, context)
        context.object_poses_world[target] = action.target_pose_world.copy()
        deactivated_sources: list[str] = []
        if zone.disposition == "deactivate_outside":
            for source in context.truth_state.point_sources.values():
                if source.attached_prim_path == target and source.enabled:
                    source.enabled = False
                    deactivated_sources.append(source.source_id)
            if deactivated_sources:
                context.truth_state.revision.bump_source_activity()
        context.resources.consume(action.resource_cost)
        context.truth_state.revision.bump_source_pose(geometry_changed=True)
        context.advance(action.predicted_duration_s)
        return ActionResult(
            action.action_id,
            ActionStatus.COMPLETED,
            started,
            context.sim_time_s,
            {
                "target_prim_path": target,
                "disposal_zone_id": zone_id,
                "disposition": zone.disposition,
            },
            {"deactivated_source_ids": deactivated_sources},
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
