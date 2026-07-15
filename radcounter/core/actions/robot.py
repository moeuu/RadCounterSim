"""Robot controller protocol and deterministic implementation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class ExecutionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True)
class RobotExecutionResult:
    """One controller operation outcome."""

    status: ExecutionStatus
    message: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status == ExecutionStatus.SUCCEEDED


class RobotController(Protocol):
    """Shared deterministic/physics/ROS robot contract."""

    async def navigate_to(
        self, pose_world: NDArray[np.float64], timeout_s: float
    ) -> RobotExecutionResult: ...

    async def move_end_effector(
        self, pose_world: NDArray[np.float64], timeout_s: float
    ) -> RobotExecutionResult: ...

    async def execute_joint_trajectory(self, trajectory: object) -> RobotExecutionResult: ...

    async def grasp(self, target_prim_path: str) -> RobotExecutionResult: ...

    async def release(self) -> RobotExecutionResult: ...

    async def stop(self) -> None: ...


@dataclass
class DeterministicRobotController:
    """High-level controller for CI, sweeps, and closed-loop research."""

    robot_id: str
    reachability: Callable[[NDArray[np.float64]], bool] = lambda _pose: True
    graspable_prims: set[str] | None = None
    operation_log: list[tuple[str, str]] = field(default_factory=list)
    stopped: bool = False

    @staticmethod
    def _pose(pose_world: NDArray[np.float64]) -> NDArray[np.float64]:
        pose = np.asarray(pose_world, dtype=np.float64)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise ValueError("robot pose must be a finite 4x4 matrix")
        return pose

    async def navigate_to(
        self, pose_world: NDArray[np.float64], timeout_s: float
    ) -> RobotExecutionResult:
        pose = self._pose(pose_world)
        self.operation_log.append(("navigate", f"{pose[:3, 3].tolist()}"))
        if self.stopped or timeout_s <= 0 or not self.reachability(pose):
            return RobotExecutionResult(ExecutionStatus.FAILED, "navigation infeasible")
        return RobotExecutionResult(ExecutionStatus.SUCCEEDED)

    async def move_end_effector(
        self, pose_world: NDArray[np.float64], timeout_s: float
    ) -> RobotExecutionResult:
        pose = self._pose(pose_world)
        self.operation_log.append(("move_end_effector", f"{pose[:3, 3].tolist()}"))
        if self.stopped or timeout_s <= 0 or not self.reachability(pose):
            return RobotExecutionResult(ExecutionStatus.FAILED, "end-effector pose infeasible")
        return RobotExecutionResult(ExecutionStatus.SUCCEEDED)

    async def execute_joint_trajectory(self, trajectory: object) -> RobotExecutionResult:
        self.operation_log.append(("joint_trajectory", type(trajectory).__name__))
        status = ExecutionStatus.FAILED if self.stopped else ExecutionStatus.SUCCEEDED
        return RobotExecutionResult(status)

    async def grasp(self, target_prim_path: str) -> RobotExecutionResult:
        self.operation_log.append(("grasp", target_prim_path))
        if self.stopped or (
            self.graspable_prims is not None and target_prim_path not in self.graspable_prims
        ):
            return RobotExecutionResult(ExecutionStatus.FAILED, "target is not graspable")
        return RobotExecutionResult(ExecutionStatus.SUCCEEDED)

    async def release(self) -> RobotExecutionResult:
        self.operation_log.append(("release", ""))
        status = ExecutionStatus.FAILED if self.stopped else ExecutionStatus.SUCCEEDED
        return RobotExecutionResult(status)

    async def stop(self) -> None:
        self.stopped = True
        self.operation_log.append(("stop", ""))
