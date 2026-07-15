"""Optional ROS 2 clients used by an Isaac-hosted robot controller."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any


class Ros2RuntimeUnavailable(RuntimeError):
    """Raised when ROS 2 Jazzy generated interfaces are unavailable."""


def require_ros2_runtime() -> dict[str, Any]:
    """Import ROS 2 and generated interfaces only when this adapter is selected."""

    try:
        rclpy = import_module("rclpy")
        action_client = import_module("rclpy.action").ActionClient
        actions = import_module("radcounter_msgs.action")
        services = import_module("radcounter_msgs.srv")
    except ModuleNotFoundError as error:
        raise Ros2RuntimeUnavailable(
            "ROS 2 Jazzy/radcounter_msgs are not installed; core remains usable without ROS 2"
        ) from error
    return {
        "rclpy": rclpy,
        "ActionClient": action_client,
        "MeasureRadiation": actions.MeasureRadiation,
        "ExecuteCountermeasure": actions.ExecuteCountermeasure,
        "GetDoseMap": services.GetDoseMap,
        "EvaluateCountermeasure": services.EvaluateCountermeasure,
        "ResetEpisode": services.ResetEpisode,
    }


@dataclass(frozen=True)
class Pose3D:
    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class DoseMapResult:
    success: bool
    points_m: tuple[tuple[float, float, float], ...]
    dose_rate_sv_h: tuple[float, ...]
    scene_revision: int
    message: str


class Ros2RobotController:
    """Call RadCounterSim actions/services through an externally spun ROS node."""

    def __init__(self, node: Any, *, namespace: str, timeout_s: float) -> None:
        modules = require_ros2_runtime()
        normalized_namespace = namespace.strip("/")
        if not normalized_namespace or timeout_s <= 0.0:
            raise ValueError("ROS namespace and positive timeout are required")
        self.node = node
        self.namespace = normalized_namespace
        self.timeout_s = timeout_s
        action_client = modules["ActionClient"]
        self._measure_type = modules["MeasureRadiation"]
        self._countermeasure_type = modules["ExecuteCountermeasure"]
        self._dose_map_type = modules["GetDoseMap"]
        self._evaluation_type = modules["EvaluateCountermeasure"]
        self._reset_type = modules["ResetEpisode"]
        self._measure_client = action_client(
            node,
            self._measure_type,
            self._endpoint("measure_radiation"),
        )
        self._countermeasure_client = action_client(
            node,
            self._countermeasure_type,
            self._endpoint("execute_countermeasure"),
        )
        self._dose_map_client = node.create_client(
            self._dose_map_type,
            self._endpoint("get_dose_map"),
        )
        self._evaluation_client = node.create_client(
            self._evaluation_type,
            self._endpoint("evaluate_countermeasure"),
        )
        self._reset_client = node.create_client(
            self._reset_type,
            self._endpoint("reset_episode"),
        )
        self._active_goals: set[Any] = set()

    def _endpoint(self, name: str) -> str:
        return f"/{self.namespace}/{name}"

    async def _await(self, future: Any) -> Any:
        try:
            return await asyncio.wait_for(future, timeout=self.timeout_s)
        except TimeoutError as error:
            raise TimeoutError(f"ROS operation exceeded {self.timeout_s:.3f} s") from error

    def _wait_for_action(self, client: Any, endpoint: str) -> None:
        if not client.wait_for_server(timeout_sec=self.timeout_s):
            raise TimeoutError(f"ROS action server unavailable: {endpoint}")

    def _wait_for_service(self, client: Any, endpoint: str) -> None:
        if not client.wait_for_service(timeout_sec=self.timeout_s):
            raise TimeoutError(f"ROS service unavailable: {endpoint}")

    async def measure_radiation(
        self,
        *,
        detector_id: str,
        duration_s: float,
        target_pose: Pose3D,
        feedback: Callable[[float, str], None] | None = None,
    ) -> Any:
        """Request one radiation measurement and return its generated ROS message."""

        if not detector_id or duration_s <= 0.0:
            raise ValueError("detector_id and positive duration_s are required")
        endpoint = self._endpoint("measure_radiation")
        self._wait_for_action(self._measure_client, endpoint)
        goal = self._measure_type.Goal()
        goal.detector_id = detector_id
        goal.duration_s = duration_s
        self._set_pose(goal.target_pose, target_pose)

        def feedback_callback(message: Any) -> None:
            if feedback is not None:
                feedback(float(message.feedback.progress), str(message.feedback.state))

        goal_handle = await self._await(
            self._measure_client.send_goal_async(
                goal,
                feedback_callback=feedback_callback,
            )
        )
        if not goal_handle.accepted:
            raise RuntimeError("measurement goal was rejected")
        self._active_goals.add(goal_handle)
        try:
            wrapped_result = await self._await(goal_handle.get_result_async())
        finally:
            self._active_goals.discard(goal_handle)
        if not wrapped_result.result.success:
            raise RuntimeError(wrapped_result.result.message)
        return wrapped_result.result.measurement

    async def execute_countermeasure(
        self,
        *,
        action_id: str,
        action_type: str,
        robot_id: str,
        target_prim_path: str,
        target_pose: Pose3D,
        parameters: Mapping[str, object] | None = None,
        feedback: Callable[[Any], None] | None = None,
    ) -> Any:
        """Execute a countermeasure and return its final status message."""

        if not action_id or not action_type or not robot_id:
            raise ValueError("action_id, action_type, and robot_id are required")
        endpoint = self._endpoint("execute_countermeasure")
        self._wait_for_action(self._countermeasure_client, endpoint)
        goal = self._countermeasure_type.Goal()
        goal.action_id = action_id
        goal.action_type = action_type
        goal.robot_id = robot_id
        goal.target_prim_path = target_prim_path
        self._set_pose(goal.target_pose, target_pose)
        goal.parameters_json = json.dumps(
            {} if parameters is None else dict(parameters),
            sort_keys=True,
        )

        def feedback_callback(message: Any) -> None:
            if feedback is not None:
                feedback(message.feedback.status)

        goal_handle = await self._await(
            self._countermeasure_client.send_goal_async(
                goal,
                feedback_callback=feedback_callback,
            )
        )
        if not goal_handle.accepted:
            raise RuntimeError("countermeasure goal was rejected")
        self._active_goals.add(goal_handle)
        try:
            wrapped_result = await self._await(goal_handle.get_result_async())
        finally:
            self._active_goals.discard(goal_handle)
        if not wrapped_result.result.success:
            raise RuntimeError(wrapped_result.result.message)
        return wrapped_result.result.final_status

    async def get_dose_map(
        self,
        *,
        frame_id: str,
        minimum_m: tuple[float, float, float],
        maximum_m: tuple[float, float, float],
        resolution_m: float,
    ) -> DoseMapResult:
        endpoint = self._endpoint("get_dose_map")
        self._wait_for_service(self._dose_map_client, endpoint)
        request = self._dose_map_type.Request()
        request.frame_id = frame_id
        self._set_point(request.minimum, minimum_m)
        self._set_point(request.maximum, maximum_m)
        request.resolution_m = resolution_m
        response = await self._await(self._dose_map_client.call_async(request))
        return DoseMapResult(
            bool(response.success),
            tuple((point.x, point.y, point.z) for point in response.points),
            tuple(response.dose_rate_sv_h),
            int(response.scene_revision),
            str(response.message),
        )

    async def evaluate_countermeasure(
        self,
        *,
        action_type: str,
        target_prim_path: str,
        target_pose: Pose3D,
        parameters: Mapping[str, object] | None = None,
    ) -> Any:
        endpoint = self._endpoint("evaluate_countermeasure")
        self._wait_for_service(self._evaluation_client, endpoint)
        request = self._evaluation_type.Request()
        request.action_type = action_type
        request.target_prim_path = target_prim_path
        self._set_pose(request.target_pose, target_pose)
        request.parameters_json = json.dumps(
            {} if parameters is None else dict(parameters),
            sort_keys=True,
        )
        return await self._await(self._evaluation_client.call_async(request))

    async def reset_episode(self, *, scenario_id: str, seed: int) -> str:
        endpoint = self._endpoint("reset_episode")
        self._wait_for_service(self._reset_client, endpoint)
        request = self._reset_type.Request()
        request.scenario_id = scenario_id
        request.seed = seed
        response = await self._await(self._reset_client.call_async(request))
        if not response.success:
            raise RuntimeError(response.message)
        return str(response.run_id)

    async def cancel_all(self) -> None:
        handles = tuple(self._active_goals)
        if handles:
            await asyncio.gather(
                *(self._await(handle.cancel_goal_async()) for handle in handles),
                return_exceptions=True,
            )

    def destroy(self) -> None:
        """Destroy clients after all active goals have been canceled."""

        self._measure_client.destroy()
        self._countermeasure_client.destroy()
        self.node.destroy_client(self._dose_map_client)
        self.node.destroy_client(self._evaluation_client)
        self.node.destroy_client(self._reset_client)

    @staticmethod
    def _set_point(message: Any, values: tuple[float, float, float]) -> None:
        message.x, message.y, message.z = values

    @staticmethod
    def _set_pose(message: Any, pose: Pose3D) -> None:
        message.position.x, message.position.y, message.position.z = pose.position_m
        (
            message.orientation.x,
            message.orientation.y,
            message.orientation.z,
            message.orientation.w,
        ) = pose.orientation_xyzw
