"""Live DDS action roundtrip gate for Nav2, MoveIt, gripper, and Isaac commands."""

from __future__ import annotations

import json
import threading
import time

import rclpy
from control_msgs.action import FollowJointTrajectory, GripperCommand
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MoveItErrorCodes
from nav2_msgs.action import NavigateToPose
from radcounter_msgs.action import ExecuteRobotTask
from radcounter_robot_gateway.gateway_node import RobotGateway
from rclpy.action import ActionClient, ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String


def _wait(future, timeout_s: float = 10.0):
    deadline = time.monotonic() + timeout_s
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not future.done():
        raise TimeoutError("ROS future did not complete")
    return future.result()


class FakeMotionAndIsaac(Node):
    def __init__(self) -> None:
        super().__init__("radcounter_fake_motion_and_isaac")
        self.commands: list[dict[str, object]] = []
        self.moveit_calls = 0
        self.trajectory_calls = 0
        self.nav_server = ActionServer(self, NavigateToPose, "/navigate_to_pose", self._navigate)
        self.moveit_server = ActionServer(self, MoveGroup, "/move_action", self._move)
        self.gripper_server = ActionServer(
            self,
            GripperCommand,
            "/gripper_controller/gripper_cmd",
            self._gripper,
        )
        self.trajectory_server = ActionServer(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
            self._trajectory,
        )
        self.result_publisher = self.create_publisher(String, "/radcounter/isaac/result", 10)
        self.command_subscription = self.create_subscription(
            String,
            "/radcounter/isaac/command",
            self._isaac_command,
            10,
        )

    @staticmethod
    def _navigate(goal_handle):
        goal_handle.succeed()
        result = NavigateToPose.Result()
        result.error_code = 0
        result.error_msg = ""
        return result

    def _move(self, goal_handle):
        self.moveit_calls += 1
        goal_handle.succeed()
        result = MoveGroup.Result()
        result.error_code.val = MoveItErrorCodes.SUCCESS
        return result

    def _trajectory(self, goal_handle):
        self.trajectory_calls += 1
        assert goal_handle.request.trajectory.joint_names == ["joint_a", "joint_b"]
        assert len(goal_handle.request.trajectory.points) == 1
        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        result.error_string = ""
        return result

    @staticmethod
    def _gripper(goal_handle):
        goal_handle.succeed()
        result = GripperCommand.Result()
        result.position = goal_handle.request.command.position
        result.effort = goal_handle.request.command.max_effort
        result.reached_goal = True
        return result

    def _isaac_command(self, message: String) -> None:
        command = json.loads(message.data)
        self.commands.append(command)
        result = String()
        result.data = json.dumps(
            {
                "task_id": command["task_id"],
                "success": True,
                "message": f"{command['command']} complete",
            }
        )
        self.result_publisher.publish(result)


def main() -> int:
    rclpy.init()
    fake = FakeMotionAndIsaac()
    gateway = RobotGateway()
    client_node = Node("radcounter_gateway_gate_client")
    executor = MultiThreadedExecutor(num_threads=6)
    for node in (fake, gateway, client_node):
        executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    phases: list[str] = []
    try:
        client = ActionClient(client_node, ExecuteRobotTask, "/radcounter/execute_robot_task")
        assert client.wait_for_server(timeout_sec=3.0)
        goal = ExecuteRobotTask.Goal()
        goal.task_id = "ros-gate-shield-1"
        goal.action_type = "shield_place"
        goal.object_path = "/World/LeadShield"
        for pose in (
            goal.pickup_base_pose,
            goal.pickup_tool_pose,
            goal.placement_base_pose,
            goal.placement_tool_pose,
        ):
            pose.header.frame_id = "map"
            pose.pose.orientation.w = 1.0
        goal.pickup_base_pose.pose.position.x = -3.0
        goal.pickup_tool_pose.pose.position.z = 0.9
        goal.placement_base_pose.pose.position.x = 0.6
        goal.placement_tool_pose.pose.position.x = 1.2
        goal.placement_tool_pose.pose.position.z = 0.9
        handle = _wait(
            client.send_goal_async(
                goal,
                feedback_callback=lambda message: phases.append(message.feedback.phase),
            )
        )
        assert handle.accepted
        wrapped = _wait(handle.get_result_async())
        assert wrapped.result.success, wrapped.result.message
        commands = [str(command["command"]) for command in fake.commands]
        assert commands == ["grasp", "release", "synchronize_radiation"], commands
        assert phases[0] == "navigate_pickup"
        assert phases[-1] == "complete"

        trajectory_phases: list[str] = []
        trajectory_goal = ExecuteRobotTask.Goal()
        trajectory_goal.task_id = "ros-gate-shield-joints-2"
        trajectory_goal.action_type = "shield_place"
        trajectory_goal.object_path = "/World/LeadShield"
        trajectory_goal.use_joint_trajectory = True
        trajectory_goal.joint_names = ["joint_a", "joint_b"]
        trajectory_goal.pickup_joint_positions = [0.2, -0.4]
        trajectory_goal.placement_joint_positions = [0.7, 0.1]
        trajectory_goal.trajectory_duration_s = 0.25
        for pose in (
            trajectory_goal.pickup_base_pose,
            trajectory_goal.pickup_tool_pose,
            trajectory_goal.placement_base_pose,
            trajectory_goal.placement_tool_pose,
        ):
            pose.header.frame_id = "map"
            pose.pose.orientation.w = 1.0
        second_handle = _wait(
            client.send_goal_async(
                trajectory_goal,
                feedback_callback=lambda message: trajectory_phases.append(message.feedback.phase),
            )
        )
        assert second_handle.accepted
        second_result = _wait(second_handle.get_result_async())
        assert second_result.result.success, second_result.result.message
        assert fake.moveit_calls == 2
        assert fake.trajectory_calls == 2
        assert [str(command["command"]) for command in fake.commands] == [
            "grasp",
            "release",
            "synchronize_radiation",
            "grasp",
            "release",
            "synchronize_radiation",
        ]
        print(
            json.dumps(
                {
                    "success": True,
                    "commands": [str(item["command"]) for item in fake.commands],
                    "moveit_calls": fake.moveit_calls,
                    "trajectory_calls": fake.trajectory_calls,
                    "phases": phases,
                    "trajectory_phases": trajectory_phases,
                }
            ),
            flush=True,
        )
        return 0
    finally:
        executor.shutdown(timeout_sec=2.0)
        spin_thread.join(timeout=2.0)
        for node in (client_node, gateway, fake):
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
