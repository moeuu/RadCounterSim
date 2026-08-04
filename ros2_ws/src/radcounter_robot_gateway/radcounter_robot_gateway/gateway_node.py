"""Countermeasure task action server orchestrating ROS motion and Isaac physics."""

from __future__ import annotations

import rclpy
from radcounter_msgs.action import ExecuteRobotTask
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node

from .isaac_command import IsaacCommandClient
from .motion_clients import (
    GripperClient,
    JointTrajectoryClient,
    MotionOutcome,
    MoveIt2Client,
    Nav2Client,
)


class RobotGateway(Node):
    def __init__(self) -> None:
        super().__init__("radcounter_robot_gateway")
        self.declare_parameter("navigate_action", "/navigate_to_pose")
        self.declare_parameter("move_group_action", "/move_action")
        self.declare_parameter("gripper_action", "/gripper_controller/gripper_cmd")
        self.declare_parameter(
            "joint_trajectory_action",
            "/joint_trajectory_controller/follow_joint_trajectory",
        )
        self.declare_parameter("move_group", "manipulator")
        self.declare_parameter("end_effector_link", "tool0")
        self.declare_parameter("isaac_command_topic", "/radcounter/isaac/command")
        self.declare_parameter("isaac_result_topic", "/radcounter/isaac/result")
        self._nav = Nav2Client(self, self.get_parameter("navigate_action").value)
        self._moveit = MoveIt2Client(
            self,
            self.get_parameter("move_group_action").value,
            self.get_parameter("move_group").value,
            self.get_parameter("end_effector_link").value,
        )
        self._gripper = GripperClient(self, self.get_parameter("gripper_action").value)
        self._trajectory = JointTrajectoryClient(
            self, self.get_parameter("joint_trajectory_action").value
        )
        self._isaac = IsaacCommandClient(
            self,
            self.get_parameter("isaac_command_topic").value,
            self.get_parameter("isaac_result_topic").value,
        )
        self._active_task_ids: set[str] = set()
        self._server = ActionServer(
            self,
            ExecuteRobotTask,
            "/radcounter/execute_robot_task",
            execute_callback=self._execute,
            goal_callback=self._goal,
            cancel_callback=self._cancel,
        )

    def _goal(self, request: ExecuteRobotTask.Goal) -> GoalResponse:
        if not request.task_id or request.task_id in self._active_task_ids:
            return GoalResponse.REJECT
        if request.action_type not in {
            "shield_place",
            "object_move",
            "object_remove",
            "decontaminate",
        }:
            return GoalResponse.REJECT
        if request.use_joint_trajectory and (
            not request.joint_names
            or len(request.pickup_joint_positions) != len(request.joint_names)
            or len(request.placement_joint_positions) != len(request.joint_names)
            or request.trajectory_duration_s <= 0
        ):
            return GoalResponse.REJECT
        self._active_task_ids.add(request.task_id)
        return GoalResponse.ACCEPT

    def _cancel(self, goal_handle) -> CancelResponse:
        self._isaac.cancel(goal_handle.request.task_id)
        return CancelResponse.ACCEPT

    @staticmethod
    def _feedback(goal_handle, phase: str, progress: float) -> None:
        feedback = ExecuteRobotTask.Feedback()
        feedback.phase = phase
        feedback.progress = float(progress)
        goal_handle.publish_feedback(feedback)

    @staticmethod
    def _failure(message: str, error_code: int = 1) -> ExecuteRobotTask.Result:
        result = ExecuteRobotTask.Result()
        result.success = False
        result.error_code = error_code
        result.message = message
        return result

    @staticmethod
    def _success(message: str) -> ExecuteRobotTask.Result:
        result = ExecuteRobotTask.Result()
        result.success = True
        result.error_code = 0
        result.message = message
        return result

    async def _motion(
        self, goal_handle, phase: str, progress: float, awaitable
    ) -> MotionOutcome | None:
        self._feedback(goal_handle, phase, progress)
        outcome = await awaitable
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            return None
        return outcome

    async def _arm_motion(self, request, *, pickup: bool) -> MotionOutcome:
        if request.use_joint_trajectory:
            positions = (
                request.pickup_joint_positions if pickup else request.placement_joint_positions
            )
            return await self._trajectory.execute(
                request.joint_names,
                positions,
                request.trajectory_duration_s,
            )
        pose = request.pickup_tool_pose if pickup else request.placement_tool_pose
        return await self._moveit.move_to_pose(pose)

    async def _execute(self, goal_handle) -> ExecuteRobotTask.Result:
        request = goal_handle.request
        task_id = request.task_id
        try:
            navigation = await self._motion(
                goal_handle,
                "navigate_pickup",
                0.08,
                self._nav.navigate(request.pickup_base_pose),
            )
            if navigation is None:
                return self._failure("task canceled", 2)
            if not navigation.success:
                goal_handle.abort()
                return self._failure(navigation.message, 10)
            arm = await self._motion(
                goal_handle,
                "move_to_pickup",
                0.22,
                self._arm_motion(request, pickup=True),
            )
            if arm is None:
                return self._failure("task canceled", 2)
            if not arm.success:
                goal_handle.abort()
                return self._failure(arm.message, 11)

            if request.action_type == "decontaminate":
                self._feedback(goal_handle, "physical_decontamination", 0.55)
                physical = await self._isaac.execute(
                    task_id,
                    "decontaminate",
                    surface_path=request.surface_path,
                    duration_s=request.treatment_duration_s,
                )
                if not bool(physical.get("success", False)):
                    goal_handle.abort()
                    return self._failure(str(physical.get("message", "decontamination failed")), 20)
                goal_handle.succeed()
                self._feedback(goal_handle, "complete", 1.0)
                return self._success("decontamination completed and activity map committed")

            gripper = await self._motion(
                goal_handle, "close_gripper", 0.34, self._gripper.command(0.0, 80.0)
            )
            if gripper is None:
                return self._failure("task canceled", 2)
            if not gripper.success:
                goal_handle.abort()
                return self._failure(gripper.message, 12)
            grasp = await self._isaac.execute(task_id, "grasp", object_path=request.object_path)
            if not bool(grasp.get("success", False)):
                goal_handle.abort()
                return self._failure(str(grasp.get("message", "physical grasp failed")), 21)
            navigation = await self._motion(
                goal_handle,
                "navigate_placement",
                0.52,
                self._nav.navigate(request.placement_base_pose),
            )
            if navigation is None:
                return self._failure("task canceled", 2)
            if not navigation.success:
                goal_handle.abort()
                return self._failure(navigation.message, 13)
            arm = await self._motion(
                goal_handle,
                "move_to_placement",
                0.68,
                self._arm_motion(request, pickup=False),
            )
            if arm is None:
                return self._failure("task canceled", 2)
            if not arm.success:
                goal_handle.abort()
                return self._failure(arm.message, 14)
            release = await self._isaac.execute(task_id, "release", object_path=request.object_path)
            if not bool(release.get("success", False)):
                goal_handle.abort()
                return self._failure(str(release.get("message", "physical release failed")), 22)
            opened = await self._motion(
                goal_handle, "open_gripper", 0.82, self._gripper.command(0.08, 30.0)
            )
            if opened is None:
                return self._failure("task canceled", 2)
            if request.action_type == "object_remove":
                removed = await self._isaac.execute(
                    task_id,
                    "remove",
                    object_path=request.object_path,
                    disposal_zone_path=request.disposal_zone_path,
                )
                if not bool(removed.get("success", False)):
                    goal_handle.abort()
                    return self._failure(
                        str(removed.get("message", "removal validation failed")), 23
                    )
            synchronized = await self._isaac.execute(task_id, "synchronize_radiation")
            if not bool(synchronized.get("success", False)):
                goal_handle.abort()
                return self._failure(
                    str(synchronized.get("message", "radiation synchronization failed")), 24
                )
            goal_handle.succeed()
            self._feedback(goal_handle, "complete", 1.0)
            return self._success("countermeasure executed and radiation scene synchronized")
        finally:
            self._active_task_ids.discard(task_id)

    def destroy_node(self):
        self._server.destroy()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotGateway()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
