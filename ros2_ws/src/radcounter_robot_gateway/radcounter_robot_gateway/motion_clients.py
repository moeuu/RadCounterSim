"""Typed asynchronous clients for Nav2, MoveIt 2, and a ROS 2 gripper."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, OrientationConstraint, PositionConstraint
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectoryPoint


@dataclass(frozen=True, slots=True)
class MotionOutcome:
    success: bool
    message: str
    status: int


class Nav2Client:
    def __init__(self, node, action_name: str = "/navigate_to_pose") -> None:
        self._client = ActionClient(node, NavigateToPose, action_name)

    async def navigate(self, pose: PoseStamped, behavior_tree: str = "") -> MotionOutcome:
        if not self._client.wait_for_server(timeout_sec=2.0):
            return MotionOutcome(False, "Nav2 action server unavailable", GoalStatus.STATUS_UNKNOWN)
        goal = NavigateToPose.Goal()
        goal.pose = pose
        goal.behavior_tree = behavior_tree
        handle = await self._client.send_goal_async(goal)
        if not handle.accepted:
            return MotionOutcome(False, "Nav2 rejected goal", GoalStatus.STATUS_ABORTED)
        wrapped = await handle.get_result_async()
        result = wrapped.result
        success = wrapped.status == GoalStatus.STATUS_SUCCEEDED and int(result.error_code) == 0
        return MotionOutcome(
            success,
            result.error_msg or ("navigation complete" if success else "navigation failed"),
            wrapped.status,
        )

    async def cancel_all(self) -> None:
        await self._client.cancel_all_goals_async()


class MoveIt2Client:
    def __init__(
        self,
        node,
        action_name: str = "/move_action",
        group_name: str = "manipulator",
        end_effector_link: str = "tool0",
    ) -> None:
        self._client = ActionClient(node, MoveGroup, action_name)
        self.group_name = group_name
        self.end_effector_link = end_effector_link

    @staticmethod
    def _constraints(pose: PoseStamped, link_name: str) -> Constraints:
        constraints = Constraints()
        constraints.name = "radcounter_pose_goal"
        position = PositionConstraint()
        position.header = pose.header
        position.link_name = link_name
        position.weight = 1.0
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [0.01]
        position.constraint_region.primitives = [sphere]
        position.constraint_region.primitive_poses = [pose.pose]
        orientation = OrientationConstraint()
        orientation.header = pose.header
        orientation.link_name = link_name
        orientation.orientation = pose.pose.orientation
        orientation.absolute_x_axis_tolerance = 0.04
        orientation.absolute_y_axis_tolerance = 0.04
        orientation.absolute_z_axis_tolerance = 0.04
        orientation.weight = 1.0
        constraints.position_constraints = [position]
        constraints.orientation_constraints = [orientation]
        return constraints

    async def move_to_pose(self, pose: PoseStamped) -> MotionOutcome:
        if not self._client.wait_for_server(timeout_sec=2.0):
            return MotionOutcome(
                False, "MoveIt 2 action server unavailable", GoalStatus.STATUS_UNKNOWN
            )
        goal = MoveGroup.Goal()
        goal.request.group_name = self.group_name
        goal.request.num_planning_attempts = 4
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.35
        goal.request.max_acceleration_scaling_factor = 0.25
        goal.request.goal_constraints = [self._constraints(pose, self.end_effector_link)]
        goal.planning_options.plan_only = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2
        handle = await self._client.send_goal_async(goal)
        if not handle.accepted:
            return MotionOutcome(False, "MoveIt 2 rejected goal", GoalStatus.STATUS_ABORTED)
        wrapped = await handle.get_result_async()
        error_code = int(wrapped.result.error_code.val)
        success = wrapped.status == GoalStatus.STATUS_SUCCEEDED and error_code == 1
        return MotionOutcome(success, f"MoveIt error code {error_code}", wrapped.status)

    async def cancel_all(self) -> None:
        await self._client.cancel_all_goals_async()


class JointTrajectoryClient:
    """Execute an explicit joint-space trajectory through ros2_control."""

    def __init__(
        self,
        node,
        action_name: str = "/joint_trajectory_controller/follow_joint_trajectory",
    ) -> None:
        self._client = ActionClient(node, FollowJointTrajectory, action_name)

    async def execute(
        self,
        joint_names: Sequence[str],
        positions: Sequence[float],
        duration_s: float,
        *,
        velocities: Sequence[float] = (),
        accelerations: Sequence[float] = (),
    ) -> MotionOutcome:
        names = list(joint_names)
        position_values = [float(value) for value in positions]
        if not names or len(names) != len(position_values):
            return MotionOutcome(
                False,
                "joint names and positions must be nonempty and equal length",
                GoalStatus.STATUS_UNKNOWN,
            )
        if duration_s <= 0:
            return MotionOutcome(
                False, "trajectory duration must be positive", GoalStatus.STATUS_UNKNOWN
            )
        if velocities and len(velocities) != len(names):
            return MotionOutcome(
                False, "velocity vector length mismatch", GoalStatus.STATUS_UNKNOWN
            )
        if accelerations and len(accelerations) != len(names):
            return MotionOutcome(
                False, "acceleration vector length mismatch", GoalStatus.STATUS_UNKNOWN
            )
        if not self._client.wait_for_server(timeout_sec=2.0):
            return MotionOutcome(
                False,
                "joint trajectory action server unavailable",
                GoalStatus.STATUS_UNKNOWN,
            )
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = names
        point = JointTrajectoryPoint()
        point.positions = position_values
        point.velocities = [float(value) for value in velocities]
        point.accelerations = [float(value) for value in accelerations]
        seconds = int(duration_s)
        nanoseconds = int(round((duration_s - seconds) * 1.0e9))
        if nanoseconds == 1_000_000_000:
            seconds += 1
            nanoseconds = 0
        point.time_from_start.sec = seconds
        point.time_from_start.nanosec = nanoseconds
        goal.trajectory.points = [point]
        handle = await self._client.send_goal_async(goal)
        if not handle.accepted:
            return MotionOutcome(False, "joint trajectory rejected goal", GoalStatus.STATUS_ABORTED)
        wrapped = await handle.get_result_async()
        result = wrapped.result
        success = (
            wrapped.status == GoalStatus.STATUS_SUCCEEDED
            and int(result.error_code) == FollowJointTrajectory.Result.SUCCESSFUL
        )
        message = result.error_string or (
            "joint trajectory complete" if success else "joint trajectory failed"
        )
        return MotionOutcome(success, message, wrapped.status)

    async def cancel_all(self) -> None:
        await self._client.cancel_all_goals_async()


class GripperClient:
    def __init__(self, node, action_name: str = "/gripper_controller/gripper_cmd") -> None:
        self._client = ActionClient(node, GripperCommand, action_name)

    async def command(self, position: float, max_effort: float) -> MotionOutcome:
        if not self._client.wait_for_server(timeout_sec=2.0):
            return MotionOutcome(
                False, "gripper action server unavailable", GoalStatus.STATUS_UNKNOWN
            )
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = max_effort
        handle = await self._client.send_goal_async(goal)
        if not handle.accepted:
            return MotionOutcome(False, "gripper rejected goal", GoalStatus.STATUS_ABORTED)
        wrapped = await handle.get_result_async()
        success = wrapped.status == GoalStatus.STATUS_SUCCEEDED and bool(
            wrapped.result.reached_goal
        )
        return MotionOutcome(
            success, "gripper goal reached" if success else "gripper failed", wrapped.status
        )

    async def cancel_all(self) -> None:
        await self._client.cancel_all_goals_async()
