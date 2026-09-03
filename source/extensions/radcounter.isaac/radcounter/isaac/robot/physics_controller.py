"""PhysX-backed robot actions with no transform-teleport fallback."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from radcounter.core.experiments import EvidenceClass

from .disposal import apply_disposal_state, disposal_configuration

FloatArray = NDArray[np.float64]


class RobotExecutionState(StrEnum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    MOVING_END_EFFECTOR = "moving_end_effector"
    GRASPING = "grasping"
    CARRYING = "carrying"
    RELEASING = "releasing"
    SETTLING = "settling"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PhysicsControllerConfig:
    linear_gain: float = 1.8
    angular_gain: float = 2.5
    maximum_linear_speed_m_s: float = 0.8
    maximum_angular_speed_rad_s: float = 1.5
    navigation_tolerance_m: float = 0.06
    yaw_tolerance_rad: float = 0.18
    grasp_distance_m: float = 0.18
    settle_linear_speed_m_s: float = 0.02
    settle_angular_speed_rad_s: float = 0.03
    maximum_navigation_steps: int = 2400
    maximum_settle_steps: int = 360


@dataclass(frozen=True, slots=True)
class PhysicsActionReport:
    state: RobotExecutionState
    success: bool
    steps: int
    message: str
    object_path: str | None = None
    evidence_class: EvidenceClass = EvidenceClass.PHYSICAL_ROBOT_EXECUTION


class PhysicsStepper(Protocol):
    def step(self, *, render: bool = False) -> None: ...


class InverseKinematicsSolver(Protocol):
    def solve(
        self, position_m: FloatArray, orientation_xyzw: FloatArray | None
    ) -> tuple[FloatArray, bool]: ...


class IsaacLulaIkSolver:
    """Concrete Isaac Motion Generation adapter for articulated robots."""

    def __init__(
        self,
        articulation: Any,
        robot_description_path: str,
        urdf_path: str,
        end_effector_frame_name: str,
    ) -> None:
        from isaacsim.robot_motion.motion_generation import (
            ArticulationKinematicsSolver,
            LulaKinematicsSolver,
        )

        lula = LulaKinematicsSolver(robot_description_path, urdf_path)
        self._solver = ArticulationKinematicsSolver(articulation, lula, end_effector_frame_name)

    def solve(
        self, position_m: FloatArray, orientation_xyzw: FloatArray | None
    ) -> tuple[FloatArray, bool]:
        orientation_wxyz = None
        if orientation_xyzw is not None:
            x, y, z, w = np.asarray(orientation_xyzw, dtype=np.float64)
            orientation_wxyz = np.asarray((w, x, y, z), dtype=np.float64)
        action, success = self._solver.compute_inverse_kinematics(
            np.asarray(position_m, dtype=np.float64), orientation_wxyz
        )
        positions = getattr(action, "joint_positions", None)
        if not success or positions is None:
            return np.empty(0, dtype=np.float64), False
        return np.asarray(positions, dtype=np.float64), True


class IsaacPhysicsRobotController:
    """Drive a rigid mobile base and articulation through PhysX APIs."""

    def __init__(
        self,
        stage: Any,
        robot_body_path: str,
        end_effector_path: str,
        stepper: PhysicsStepper,
        *,
        articulation_path: str | None = None,
        ik_solver: InverseKinematicsSolver | None = None,
        config: PhysicsControllerConfig | None = None,
    ) -> None:
        from isaacsim.core.prims import RigidPrim

        self.stage = stage
        self.robot_body_path = robot_body_path
        self.end_effector_path = end_effector_path
        self.stepper = stepper
        self.config = config or PhysicsControllerConfig()
        self.ik_solver = ik_solver
        self.state = RobotExecutionState.IDLE
        self._base = RigidPrim(robot_body_path, name="radcounter_countermeasure_base")
        self._base.initialize()
        self._articulation = None
        if articulation_path is not None:
            from isaacsim.core.prims import Articulation

            self._articulation = Articulation(
                articulation_path, name="radcounter_countermeasure_arm"
            )
            self._articulation.initialize()
        self._grasp_joint_path = f"{robot_body_path}/RadCounterGraspJoint"
        self._grasped_object_path: str | None = None
        self._grasp_collision_state: dict[str, bool] = {}
        self._grasp_disable_gravity = False
        self.trace: list[RobotExecutionState] = [self.state]

    def _transition(self, state: RobotExecutionState) -> None:
        self.state = state
        self.trace.append(state)

    @staticmethod
    def _numpy(value: Any) -> FloatArray:
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=np.float64)

    def _base_pose(self) -> tuple[FloatArray, FloatArray]:
        positions, orientations = self._base.get_world_poses()
        return self._numpy(positions)[0], self._numpy(orientations)[0]

    @staticmethod
    def _yaw_from_wxyz(quaternion: FloatArray) -> float:
        w, x, y, z = quaternion
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _wrap_angle(value: float) -> float:
        return math.atan2(math.sin(value), math.cos(value))

    def _set_twist(self, linear_xyz: Sequence[float], angular_xyz: Sequence[float]) -> None:
        self._base.set_linear_velocities(np.asarray([linear_xyz], dtype=np.float32))
        self._base.set_angular_velocities(np.asarray([angular_xyz], dtype=np.float32))

    def stop(self) -> None:
        self._set_twist((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def navigate_to(
        self, target_position_m: Sequence[float], target_yaw_rad: float | None = None
    ) -> PhysicsActionReport:
        target = np.asarray(target_position_m, dtype=np.float64)
        if target.shape not in {(2,), (3,)}:
            raise ValueError("target_position_m must contain x/y or x/y/z")
        self._transition(RobotExecutionState.NAVIGATING)
        for step in range(1, self.config.maximum_navigation_steps + 1):
            position, orientation = self._base_pose()
            error = target[:2] - position[:2]
            distance = float(np.linalg.norm(error))
            yaw = self._yaw_from_wxyz(orientation)
            desired_yaw = (
                math.atan2(error[1], error[0])
                if distance > self.config.navigation_tolerance_m
                else target_yaw_rad
            )
            yaw_error = 0.0 if desired_yaw is None else self._wrap_angle(desired_yaw - yaw)
            if distance <= self.config.navigation_tolerance_m and (
                target_yaw_rad is None
                or abs(self._wrap_angle(target_yaw_rad - yaw)) <= self.config.yaw_tolerance_rad
            ):
                self.stop()
                self._transition(RobotExecutionState.COMPLETE)
                return PhysicsActionReport(self.state, True, step, "navigation target reached")
            speed = (
                0.0
                if distance <= self.config.navigation_tolerance_m
                else min(self.config.maximum_linear_speed_m_s, self.config.linear_gain * distance)
            )
            if abs(yaw_error) > math.pi / 3.0:
                speed *= 0.15
            linear = speed * np.asarray([math.cos(yaw), math.sin(yaw), 0.0])
            angular_z = float(
                np.clip(
                    self.config.angular_gain * yaw_error,
                    -self.config.maximum_angular_speed_rad_s,
                    self.config.maximum_angular_speed_rad_s,
                )
            )
            self._set_twist(linear, (0.0, 0.0, angular_z))
            self.stepper.step(render=False)
        self.stop()
        position, orientation = self._base_pose()
        final_distance = float(np.linalg.norm(target[:2] - position[:2]))
        final_yaw = self._yaw_from_wxyz(orientation)
        final_yaw_error = (
            0.0 if target_yaw_rad is None else self._wrap_angle(target_yaw_rad - final_yaw)
        )
        self._transition(RobotExecutionState.FAILED)
        return PhysicsActionReport(
            self.state,
            False,
            self.config.maximum_navigation_steps,
            (
                "navigation timeout: "
                f"position_m={position.tolist()}, target_m={target.tolist()}, "
                f"distance_error_m={final_distance:.6f}, "
                f"yaw_error_rad={final_yaw_error:.6f}"
            ),
        )

    def navigate_route(
        self,
        waypoints_m: Sequence[Sequence[float]],
        target_yaw_rad: float | None = None,
    ) -> PhysicsActionReport:
        waypoints = tuple(waypoints_m)
        if not waypoints:
            return PhysicsActionReport(
                RobotExecutionState.FAILED, False, 0, "navigation route is empty"
            )
        total_steps = 0
        for index, waypoint in enumerate(waypoints):
            final_waypoint = index == len(waypoints) - 1
            report = self.navigate_to(
                waypoint,
                target_yaw_rad if final_waypoint else None,
            )
            total_steps += report.steps
            if not report.success:
                return PhysicsActionReport(
                    report.state,
                    False,
                    total_steps,
                    f"route waypoint {index + 1}/{len(waypoints)} failed: {report.message}",
                )
        return PhysicsActionReport(
            RobotExecutionState.COMPLETE,
            True,
            total_steps,
            f"navigation route completed ({len(waypoints)} waypoints)",
        )

    def move_end_effector(
        self,
        target_position_m: Sequence[float],
        orientation_xyzw: Sequence[float] | None = None,
        *,
        maximum_steps: int = 600,
        tolerance_m: float = 0.015,
    ) -> PhysicsActionReport:
        if self._articulation is None or self.ik_solver is None:
            raise RuntimeError(
                "physics end-effector motion requires an articulation and an IK solver"
            )
        from pxr import Gf, UsdGeom

        target = np.asarray(target_position_m, dtype=np.float64)
        orientation = (
            None if orientation_xyzw is None else np.asarray(orientation_xyzw, dtype=np.float64)
        )
        self._transition(RobotExecutionState.MOVING_END_EFFECTOR)
        for step in range(1, maximum_steps + 1):
            joints, success = self.ik_solver.solve(target, orientation)
            if not success:
                self._transition(RobotExecutionState.FAILED)
                return PhysicsActionReport(
                    self.state, False, step, "IK did not find a collision-free solution"
                )
            self._articulation.set_joint_position_targets(np.asarray([joints], dtype=np.float32))
            self.stepper.step(render=False)
            end_effector = self.stage.GetPrimAtPath(self.end_effector_path)
            world = UsdGeom.XformCache().GetLocalToWorldTransform(end_effector)
            actual = np.asarray(world.Transform(Gf.Vec3d()), dtype=np.float64)
            if np.linalg.norm(actual - target) <= tolerance_m:
                self._transition(RobotExecutionState.COMPLETE)
                return PhysicsActionReport(self.state, True, step, "end effector reached target")
        self._transition(RobotExecutionState.FAILED)
        return PhysicsActionReport(self.state, False, maximum_steps, "end-effector motion timeout")

    def check_reachability(
        self,
        target_position_m: Sequence[float],
        orientation_xyzw: Sequence[float] | None = None,
    ) -> bool:
        """Run the configured IK solver without commanding the articulation."""

        if self._articulation is None or self.ik_solver is None:
            raise NotImplementedError("no articulation/IK solver is configured")
        target = np.asarray(target_position_m, dtype=np.float64)
        orientation = (
            None if orientation_xyzw is None else np.asarray(orientation_xyzw, dtype=np.float64)
        )
        joints, success = self.ik_solver.solve(target, orientation)
        return bool(success and joints.size > 0 and np.all(np.isfinite(joints)))

    def _distance_between(self, first_path: str, second_path: str) -> float:
        from pxr import Gf, UsdGeom

        cache = UsdGeom.XformCache()
        first = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(first_path)).Transform(
            Gf.Vec3d()
        )
        second = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(second_path)).Transform(
            Gf.Vec3d()
        )
        return float(np.linalg.norm(np.asarray(first) - np.asarray(second)))

    def grasp(self, object_path: str) -> PhysicsActionReport:
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

        target = self.stage.GetPrimAtPath(object_path)
        if not target or not target.IsValid():
            return PhysicsActionReport(
                RobotExecutionState.FAILED, False, 0, "grasp target does not exist", object_path
            )
        movable = target.GetAttribute("rad:manipulation:movable")
        if movable and movable.HasAuthoredValueOpinion() and not bool(movable.Get()):
            return PhysicsActionReport(
                RobotExecutionState.FAILED, False, 0, "grasp target is not movable", object_path
            )
        frame_name = ""
        frame_attribute = target.GetAttribute("rad:manipulation:graspFrame")
        if frame_attribute and frame_attribute.HasAuthoredValueOpinion():
            frame_name = str(frame_attribute.Get() or "")
        grasp_path = f"{object_path.rstrip('/')}/{frame_name}" if frame_name else object_path
        grasp_prim = self.stage.GetPrimAtPath(grasp_path)
        if not grasp_prim or not grasp_prim.IsValid():
            return PhysicsActionReport(
                RobotExecutionState.FAILED,
                False,
                0,
                f"grasp frame does not exist: {grasp_path}",
                object_path,
            )
        distance = self._distance_between(self.end_effector_path, grasp_path)
        if distance > self.config.grasp_distance_m:
            return PhysicsActionReport(
                RobotExecutionState.FAILED,
                False,
                0,
                f"grasp target is {distance:.3f} m away",
                object_path,
            )
        self._transition(RobotExecutionState.GRASPING)
        cache = UsdGeom.XformCache()
        body_world = cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(self.robot_body_path))
        target_world = cache.GetLocalToWorldTransform(target)
        grasp_world = cache.GetLocalToWorldTransform(grasp_prim)
        anchor_world = grasp_world.Transform(Gf.Vec3d())
        local_anchor = body_world.GetInverse().Transform(anchor_world)
        target_anchor = target_world.GetInverse().Transform(anchor_world)

        self._grasp_collision_state.clear()
        for collision_prim in Usd.PrimRange(target):
            collision = collision_prim.GetAttribute("physics:collisionEnabled")
            if collision and collision.HasAuthoredValueOpinion():
                path = str(collision_prim.GetPath())
                self._grasp_collision_state[path] = bool(collision.Get())
                collision.Set(False)
        rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(target)
        gravity = rigid_api.GetDisableGravityAttr()
        self._grasp_disable_gravity = (
            bool(gravity.Get()) if gravity.HasAuthoredValueOpinion() else False
        )
        rigid_api.CreateDisableGravityAttr().Set(True)

        body_rotation = body_world.ExtractRotationQuat()
        target_rotation = target_world.ExtractRotationQuat()
        local_rotation = body_rotation.GetInverse() * target_rotation
        imaginary = local_rotation.GetImaginary()

        joint = UsdPhysics.FixedJoint.Define(self.stage, self._grasp_joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(self.robot_body_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(object_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_anchor))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*target_anchor))
        joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(
                float(local_rotation.GetReal()),
                Gf.Vec3f(float(imaginary[0]), float(imaginary[1]), float(imaginary[2])),
            )
        )
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        self._grasped_object_path = object_path
        self.stepper.step(render=False)
        self._transition(RobotExecutionState.CARRYING)
        return PhysicsActionReport(self.state, True, 1, "fixed-joint grasp engaged", object_path)

    def release(self) -> PhysicsActionReport:
        from pxr import PhysxSchema

        if self._grasped_object_path is None:
            return PhysicsActionReport(RobotExecutionState.FAILED, False, 0, "no object is grasped")
        object_path = self._grasped_object_path
        self._transition(RobotExecutionState.RELEASING)
        self.stage.RemovePrim(self._grasp_joint_path)
        target = self.stage.GetPrimAtPath(object_path)
        if target and target.IsValid():
            PhysxSchema.PhysxRigidBodyAPI.Apply(target).CreateDisableGravityAttr().Set(
                self._grasp_disable_gravity
            )
        for path, enabled in self._grasp_collision_state.items():
            collision_prim = self.stage.GetPrimAtPath(path)
            if collision_prim and collision_prim.IsValid():
                collision_prim.GetAttribute("physics:collisionEnabled").Set(enabled)
        self._grasp_collision_state.clear()
        self._grasped_object_path = None
        self.stepper.step(render=False)
        self._transition(RobotExecutionState.SETTLING)
        return self.wait_until_settled(object_path)

    def wait_until_settled(self, object_path: str) -> PhysicsActionReport:
        from isaacsim.core.prims import RigidPrim

        rigid = RigidPrim(object_path, name="radcounter_settle_target")
        rigid.initialize()
        for step in range(1, self.config.maximum_settle_steps + 1):
            linear = self._numpy(rigid.get_linear_velocities())[0]
            angular = self._numpy(rigid.get_angular_velocities())[0]
            if (
                np.linalg.norm(linear) <= self.config.settle_linear_speed_m_s
                and np.linalg.norm(angular) <= self.config.settle_angular_speed_rad_s
            ):
                self._transition(RobotExecutionState.COMPLETE)
                return PhysicsActionReport(
                    self.state, True, step, "released object settled", object_path
                )
            self.stepper.step(render=False)
        self._transition(RobotExecutionState.FAILED)
        return PhysicsActionReport(
            self.state, False, self.config.maximum_settle_steps, "settle timeout", object_path
        )

    def remove_to_disposal_zone(
        self, object_path: str, disposal_zone_path: str
    ) -> PhysicsActionReport:
        from pxr import Gf, UsdGeom

        target = self.stage.GetPrimAtPath(object_path)
        zone = self.stage.GetPrimAtPath(disposal_zone_path)
        try:
            disposition = disposal_configuration(self.stage, disposal_zone_path)
        except ValueError as error:
            return PhysicsActionReport(
                RobotExecutionState.FAILED, False, 0, str(error), object_path
            )
        removable = target.GetAttribute("rad:manipulation:removable")
        if not removable or not bool(removable.Get()):
            return PhysicsActionReport(
                RobotExecutionState.FAILED, False, 0, "object is not removable", object_path
            )
        cache = UsdGeom.XformCache()
        object_position = np.asarray(cache.GetLocalToWorldTransform(target).Transform(Gf.Vec3d()))
        zone_position = np.asarray(cache.GetLocalToWorldTransform(zone).Transform(Gf.Vec3d()))
        extent = UsdGeom.Boundable(zone).ComputeExtentFromPlugins(UsdGeom.Boundable(zone), 0.0)
        radius = (
            1.0
            if not extent
            else float(np.linalg.norm(np.asarray(extent[1]) - np.asarray(extent[0])) / 2.0)
        )
        if np.linalg.norm(object_position - zone_position) > radius:
            return PhysicsActionReport(
                RobotExecutionState.FAILED, False, 0, "object is outside disposal zone", object_path
            )
        try:
            state_change = apply_disposal_state(self.stage, target, disposition)
        except ValueError as error:
            return PhysicsActionReport(
                RobotExecutionState.FAILED, False, 0, str(error), object_path
            )
        self._transition(RobotExecutionState.COMPLETE)
        return PhysicsActionReport(
            self.state,
            True,
            0,
            (
                "object secured in explicit shielding; source remains present"
                if state_change.source_present
                else "object transferred outside the evaluation domain; source disabled"
            ),
            object_path,
        )

    def execute_pick_and_place(
        self,
        object_path: str,
        pickup_base_position_m: Sequence[float],
        placement_base_position_m: Sequence[float],
        pickup_tool_position_m: Sequence[float] | None = None,
        placement_tool_position_m: Sequence[float] | None = None,
        tool_orientation_xyzw: Sequence[float] | None = None,
        before_release: Callable[[], None] | None = None,
        after_release: Callable[[], None] | None = None,
        pickup_base_yaw_rad: float | None = None,
        placement_base_yaw_rad: float | None = None,
        pickup_base_route_m: Sequence[Sequence[float]] | None = None,
        placement_base_route_m: Sequence[Sequence[float]] | None = None,
        target_root_position_m: Sequence[float] | None = None,
        placement_settle_tolerance_m: float = 0.15,
    ) -> PhysicsActionReport:
        del target_root_position_m, placement_settle_tolerance_m
        pickup_route = (
            (pickup_base_position_m,) if pickup_base_route_m is None else pickup_base_route_m
        )
        pickup = self.navigate_route(pickup_route, pickup_base_yaw_rad)
        if not pickup.success:
            return PhysicsActionReport(
                pickup.state,
                False,
                pickup.steps,
                f"pickup navigation failed: {pickup.message}",
                object_path,
            )
        if pickup_tool_position_m is not None:
            tool = self.move_end_effector(pickup_tool_position_m, tool_orientation_xyzw)
            if not tool.success:
                return PhysicsActionReport(
                    tool.state,
                    False,
                    tool.steps,
                    f"pickup tool motion failed: {tool.message}",
                    object_path,
                )
        grasp = self.grasp(object_path)
        if not grasp.success:
            return PhysicsActionReport(
                grasp.state,
                False,
                grasp.steps,
                f"grasp failed: {grasp.message}",
                object_path,
            )
        placement_route = (
            (placement_base_position_m,)
            if placement_base_route_m is None
            else placement_base_route_m
        )
        placement = self.navigate_route(placement_route, placement_base_yaw_rad)
        if not placement.success:
            return PhysicsActionReport(
                placement.state,
                False,
                placement.steps,
                f"placement navigation failed: {placement.message}",
                object_path,
            )
        if placement_tool_position_m is not None:
            tool = self.move_end_effector(placement_tool_position_m, tool_orientation_xyzw)
            if not tool.success:
                return PhysicsActionReport(
                    tool.state,
                    False,
                    tool.steps,
                    f"placement tool motion failed: {tool.message}",
                    object_path,
                )
        if before_release is not None:
            before_release()
        released = self.release()
        if released.success and after_release is not None:
            after_release()
        return released
