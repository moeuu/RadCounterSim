"""ROS 2 actions, services, and topics for RadCounterSim."""

from __future__ import annotations

import asyncio
import json
import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Protocol

import rclpy
from geometry_msgs.msg import Point, Pose
from radcounter_msgs.action import ExecuteCountermeasure, MeasureRadiation
from radcounter_msgs.msg import (
    CountermeasureStatus,
    RadiationMeasurement,
    SourceEstimate,
)
from radcounter_msgs.srv import EvaluateCountermeasure, GetDoseMap, ResetEpisode
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


@dataclass(frozen=True)
class PoseData:
    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class MeasurementData:
    measurement_id: str
    detector_id: str
    pose: PoseData
    duration_s: float
    energy_bin_edges_kev: tuple[float, ...]
    counts_per_bin: tuple[int, ...]
    dose_rate_sv_h: float
    covariance_flat: tuple[float, ...]
    scene_revision: int


@dataclass(frozen=True)
class CountermeasureData:
    action_id: str
    action_type: str
    state: str
    progress: float
    message: str
    success: bool


@dataclass(frozen=True)
class DoseMapData:
    points_m: tuple[tuple[float, float, float], ...]
    dose_rate_sv_h: tuple[float, ...]
    scene_revision: int


@dataclass(frozen=True)
class EvaluationData:
    feasible: bool
    expected_task_path_dose_sv: float
    expected_peak_dose_rate_sv_h: float
    expected_information_gain: float
    objective_score: float
    message: str


class HostBridge(Protocol):
    """Host-neutral contract implemented by analytic or Isaac adapters."""

    def measure(
        self,
        detector_id: str,
        duration_s: float,
        target_pose: PoseData,
        feedback: Callable[[float, str], None],
    ) -> MeasurementData: ...

    def execute_countermeasure(
        self,
        action_id: str,
        action_type: str,
        target_prim_path: str,
        target_pose: PoseData,
        parameters: dict[str, object],
        feedback: Callable[[CountermeasureData], None],
    ) -> CountermeasureData: ...

    def dose_map(
        self,
        minimum_m: tuple[float, float, float],
        maximum_m: tuple[float, float, float],
        resolution_m: float,
    ) -> DoseMapData: ...

    def evaluate_countermeasure(
        self,
        action_type: str,
        target_pose: PoseData,
        parameters: dict[str, object],
    ) -> EvaluationData: ...

    def reset(self, scenario_id: str, seed: int) -> tuple[str, str]: ...

    def source_estimate(self) -> tuple[tuple[float, float, float], float, float]: ...


class AnalyticHostBridge:
    """Deterministic ROS round-trip host requiring neither Isaac nor a GPU."""

    def __init__(self, seed: int = 0) -> None:
        self.reset("analytic", seed)

    def _poisson(self, expectation: float) -> int:
        if expectation <= 0.0:
            return 0
        if expectation >= 30.0:
            return max(
                0,
                int(round(self._rng.gauss(expectation, math.sqrt(expectation)))),
            )
        threshold = math.exp(-expectation)
        product_value = 1.0
        count = 0
        while product_value > threshold:
            count += 1
            product_value *= self._rng.random()
        return count - 1

    def _dose_rate(self, position_m: tuple[float, float, float]) -> float:
        squared_distance = sum(
            (value - source_value) ** 2
            for value, source_value in zip(position_m, self._source_position_m, strict=True)
        )
        response_cps = (
            self._activity_bq
            * self._shield_transmission
            / (4.0 * math.pi * max(squared_distance, 0.0625))
        )
        return response_cps * 1.0e-9

    def measure(
        self,
        detector_id: str,
        duration_s: float,
        target_pose: PoseData,
        feedback: Callable[[float, str], None],
    ) -> MeasurementData:
        if duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        feedback(0.25, "positioning")
        dose_rate = self._dose_rate(target_pose.position_m)
        total_rate_cps = dose_rate * 1.0e9 + 2.0
        feedback(0.65, "counting")
        counts = (
            self._poisson(total_rate_cps * 0.8 * duration_s),
            self._poisson(total_rate_cps * 0.2 * duration_s),
        )
        feedback(1.0, "completed")
        self._measurement_index += 1
        return MeasurementData(
            measurement_id=f"measurement-{self._measurement_index:06d}",
            detector_id=detector_id,
            pose=target_pose,
            duration_s=duration_s,
            energy_bin_edges_kev=(0.0, 800.0, 1600.0),
            counts_per_bin=counts,
            dose_rate_sv_h=dose_rate,
            covariance_flat=(float(max(counts[0], 1)), 0.0, 0.0, float(max(counts[1], 1))),
            scene_revision=self._scene_revision,
        )

    def execute_countermeasure(
        self,
        action_id: str,
        action_type: str,
        target_prim_path: str,
        target_pose: PoseData,
        parameters: dict[str, object],
        feedback: Callable[[CountermeasureData], None],
    ) -> CountermeasureData:
        del target_prim_path
        feedback(CountermeasureData(action_id, action_type, "executing", 0.5, "executing", True))
        if action_type == "decontamination":
            removal = float(parameters.get("removal_fraction", 0.5))
            if not 0.0 <= removal <= 1.0:
                raise ValueError("removal_fraction must be in [0, 1]")
            self._activity_bq *= 1.0 - removal
        elif action_type == "shield_placement":
            transmission = float(parameters.get("transmission", 0.25))
            if not 0.0 <= transmission <= 1.0:
                raise ValueError("transmission must be in [0, 1]")
            self._shield_transmission = transmission
        elif action_type == "object_move":
            self._source_position_m = target_pose.position_m
        else:
            raise ValueError(f"unsupported action_type: {action_type}")
        self._scene_revision += 1
        result = CountermeasureData(
            action_id,
            action_type,
            "succeeded",
            1.0,
            "countermeasure completed",
            True,
        )
        feedback(result)
        return result

    def dose_map(
        self,
        minimum_m: tuple[float, float, float],
        maximum_m: tuple[float, float, float],
        resolution_m: float,
    ) -> DoseMapData:
        if resolution_m <= 0.0:
            raise ValueError("resolution_m must be positive")
        axes: list[list[float]] = []
        for minimum, maximum in zip(minimum_m, maximum_m, strict=True):
            if maximum < minimum:
                raise ValueError("dose-map maximum must not be below minimum")
            count = int(math.floor((maximum - minimum) / resolution_m)) + 1
            axes.append([minimum + index * resolution_m for index in range(count)])
        point_count = math.prod(len(axis) for axis in axes)
        if point_count > 250_000:
            raise ValueError("dose map exceeds the 250000-point safety limit")
        points = tuple(product(*axes))
        rates = tuple(self._dose_rate(point) for point in points)
        return DoseMapData(points, rates, self._scene_revision)

    def evaluate_countermeasure(
        self,
        action_type: str,
        target_pose: PoseData,
        parameters: dict[str, object],
    ) -> EvaluationData:
        current_dose = self._dose_rate(target_pose.position_m)
        factors = {
            "decontamination": 0.5,
            "shield_placement": 0.25,
            "object_move": 0.7,
        }
        if action_type not in factors:
            return EvaluationData(
                False, 0.0, current_dose, 0.0, math.inf, "unsupported action type"
            )
        factor = float(parameters.get("predicted_factor", factors[action_type]))
        feasible = 0.0 <= factor <= 1.0
        peak = current_dose * factor
        information_gain = max(0.0, 1.0 - factor)
        objective = peak + 0.1 * current_dose * (1.0 - information_gain)
        return EvaluationData(
            feasible,
            current_dose * 60.0 / 3600.0,
            peak,
            information_gain,
            objective,
            "feasible" if feasible else "predicted_factor must be in [0, 1]",
        )

    def reset(self, scenario_id: str, seed: int) -> tuple[str, str]:
        self._rng = random.Random(seed)
        self._source_position_m = (0.0, 0.0, 0.8)
        self._activity_bq = 120_000.0
        self._shield_transmission = 1.0
        self._scene_revision = 1
        self._measurement_index = 0
        run_id = f"{scenario_id or 'analytic'}-seed-{seed}"
        return run_id, "episode reset"

    def source_estimate(self) -> tuple[tuple[float, float, float], float, float]:
        return self._source_position_m, self._activity_bq, 0.08 * self._activity_bq


class RadCounterBridgeNode(Node):
    """Expose a HostBridge through the public RadCounterSim ROS contract."""

    def __init__(self, host: HostBridge | None = None) -> None:
        super().__init__("bridge")
        self._host = AnalyticHostBridge() if host is None else host
        self._callback_group = ReentrantCallbackGroup()
        measurement_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        status_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._measurement_publisher = self.create_publisher(
            RadiationMeasurement,
            "measurements",
            measurement_qos,
        )
        self._estimate_publisher = self.create_publisher(
            SourceEstimate,
            "source_estimates",
            measurement_qos,
        )
        self._status_publisher = self.create_publisher(
            CountermeasureStatus,
            "countermeasure_status",
            status_qos,
        )
        self._dose_map_service = self.create_service(
            GetDoseMap,
            "get_dose_map",
            self._get_dose_map,
            callback_group=self._callback_group,
        )
        self._evaluation_service = self.create_service(
            EvaluateCountermeasure,
            "evaluate_countermeasure",
            self._evaluate_countermeasure,
            callback_group=self._callback_group,
        )
        self._reset_service = self.create_service(
            ResetEpisode,
            "reset_episode",
            self._reset_episode,
            callback_group=self._callback_group,
        )
        self._measure_action = ActionServer(
            self,
            MeasureRadiation,
            "measure_radiation",
            execute_callback=self._measure,
            goal_callback=self._measure_goal,
            cancel_callback=self._cancel,
            callback_group=self._callback_group,
        )
        self._countermeasure_action = ActionServer(
            self,
            ExecuteCountermeasure,
            "execute_countermeasure",
            execute_callback=self._execute_countermeasure,
            goal_callback=self._countermeasure_goal,
            cancel_callback=self._cancel,
            callback_group=self._callback_group,
        )

    def _measure_goal(self, request: MeasureRadiation.Goal) -> GoalResponse:
        return GoalResponse.ACCEPT if request.duration_s > 0.0 else GoalResponse.REJECT

    def _countermeasure_goal(
        self,
        request: ExecuteCountermeasure.Goal,
    ) -> GoalResponse:
        return (
            GoalResponse.ACCEPT
            if request.action_id and request.action_type
            else GoalResponse.REJECT
        )

    def _cancel(self, _goal_handle: object) -> CancelResponse:
        return CancelResponse.ACCEPT

    async def _measure(self, goal_handle: object) -> MeasureRadiation.Result:
        await asyncio.sleep(0)
        result = MeasureRadiation.Result()
        request = goal_handle.request
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.success = False
            result.message = "measurement canceled"
            return result

        def feedback(progress: float, state: str) -> None:
            message = MeasureRadiation.Feedback()
            message.progress = float(progress)
            message.state = state
            goal_handle.publish_feedback(message)

        try:
            measurement = self._host.measure(
                request.detector_id,
                request.duration_s,
                self._pose_data(request.target_pose),
                feedback,
            )
            ros_measurement = self._measurement_message(measurement)
            self._measurement_publisher.publish(ros_measurement)
            result.success = True
            result.measurement = ros_measurement
            result.message = "measurement completed"
            goal_handle.succeed()
        except (RuntimeError, ValueError) as error:
            result.success = False
            result.message = str(error)
            goal_handle.abort()
        return result

    async def _execute_countermeasure(
        self,
        goal_handle: object,
    ) -> ExecuteCountermeasure.Result:
        await asyncio.sleep(0)
        result = ExecuteCountermeasure.Result()
        request = goal_handle.request
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result.success = False
            result.message = "countermeasure canceled"
            return result

        def feedback(status: CountermeasureData) -> None:
            ros_status = self._status_message(status)
            self._status_publisher.publish(ros_status)
            message = ExecuteCountermeasure.Feedback()
            message.status = ros_status
            goal_handle.publish_feedback(message)

        try:
            parameters = self._parameters(request.parameters_json)
            status = self._host.execute_countermeasure(
                request.action_id,
                request.action_type,
                request.target_prim_path,
                self._pose_data(request.target_pose),
                parameters,
                feedback,
            )
            result.success = status.success
            result.final_status = self._status_message(status)
            result.message = status.message
            self._publish_source_estimate()
            goal_handle.succeed()
        except (json.JSONDecodeError, RuntimeError, ValueError) as error:
            result.success = False
            result.message = str(error)
            goal_handle.abort()
        return result

    def _get_dose_map(
        self,
        request: GetDoseMap.Request,
        response: GetDoseMap.Response,
    ) -> GetDoseMap.Response:
        try:
            dose_map = self._host.dose_map(
                (request.minimum.x, request.minimum.y, request.minimum.z),
                (request.maximum.x, request.maximum.y, request.maximum.z),
                request.resolution_m,
            )
            response.points = [
                Point(x=point[0], y=point[1], z=point[2]) for point in dose_map.points_m
            ]
            response.dose_rate_sv_h = list(dose_map.dose_rate_sv_h)
            response.scene_revision = dose_map.scene_revision
            response.success = True
            response.message = "dose map generated"
        except (RuntimeError, ValueError) as error:
            response.success = False
            response.message = str(error)
        return response

    def _evaluate_countermeasure(
        self,
        request: EvaluateCountermeasure.Request,
        response: EvaluateCountermeasure.Response,
    ) -> EvaluateCountermeasure.Response:
        try:
            evaluation = self._host.evaluate_countermeasure(
                request.action_type,
                self._pose_data(request.target_pose),
                self._parameters(request.parameters_json),
            )
            response.feasible = evaluation.feasible
            response.expected_task_path_dose_sv = evaluation.expected_task_path_dose_sv
            response.expected_peak_dose_rate_sv_h = evaluation.expected_peak_dose_rate_sv_h
            response.expected_information_gain = evaluation.expected_information_gain
            response.objective_score = evaluation.objective_score
            response.message = evaluation.message
        except (json.JSONDecodeError, RuntimeError, ValueError) as error:
            response.feasible = False
            response.message = str(error)
        return response

    def _reset_episode(
        self,
        request: ResetEpisode.Request,
        response: ResetEpisode.Response,
    ) -> ResetEpisode.Response:
        try:
            run_id, message = self._host.reset(request.scenario_id, request.seed)
            response.success = True
            response.run_id = run_id
            response.message = message
            self._publish_source_estimate()
        except (RuntimeError, ValueError) as error:
            response.success = False
            response.message = str(error)
        return response

    def _measurement_message(self, data: MeasurementData) -> RadiationMeasurement:
        message = RadiationMeasurement()
        self._set_header(message, "world")
        message.measurement_id = data.measurement_id
        message.detector_id = data.detector_id
        self._set_pose(message.detector_pose, data.pose)
        message.duration_s = data.duration_s
        message.energy_bin_edges_kev = list(data.energy_bin_edges_kev)
        message.counts_per_bin = list(data.counts_per_bin)
        message.dose_rate_sv_h = data.dose_rate_sv_h
        message.covariance_flat = list(data.covariance_flat)
        message.scene_revision = data.scene_revision
        return message

    def _status_message(self, data: CountermeasureData) -> CountermeasureStatus:
        message = CountermeasureStatus()
        self._set_header(message, "world")
        message.action_id = data.action_id
        message.action_type = data.action_type
        message.state = data.state
        message.progress = data.progress
        message.message = data.message
        return message

    def _publish_source_estimate(self) -> None:
        position, activity_bq, activity_std_bq = self._host.source_estimate()
        message = SourceEstimate()
        self._set_header(message, "world")
        message.estimate_id = "current-source-estimate"
        message.positions = [Point(x=position[0], y=position[1], z=position[2])]
        message.activity_bq = [activity_bq]
        message.position_covariance_flat = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
        message.activity_std_bq = [activity_std_bq]
        self._estimate_publisher.publish(message)

    def _set_header(self, message: object, frame_id: str) -> None:
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = frame_id

    @staticmethod
    def _pose_data(message: Pose) -> PoseData:
        return PoseData(
            (message.position.x, message.position.y, message.position.z),
            (
                message.orientation.x,
                message.orientation.y,
                message.orientation.z,
                message.orientation.w,
            ),
        )

    @staticmethod
    def _set_pose(message: Pose, data: PoseData) -> None:
        message.position.x, message.position.y, message.position.z = data.position_m
        (
            message.orientation.x,
            message.orientation.y,
            message.orientation.z,
            message.orientation.w,
        ) = data.orientation_xyzw

    @staticmethod
    def _parameters(encoded: str) -> dict[str, object]:
        value = json.loads(encoded or "{}")
        if not isinstance(value, dict):
            raise ValueError("parameters_json must encode an object")
        return value


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RadCounterBridgeNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
