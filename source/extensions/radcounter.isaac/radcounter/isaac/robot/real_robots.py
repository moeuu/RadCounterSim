"""Real Isaac Sim robot assets and articulated countermeasure motions.

This module deliberately keeps Isaac imports inside runtime methods so the
extension package remains importable by ordinary unit tests.  Task motion is
performed through articulation targets and PhysX constraints; no operation
uses USD transform teleportation after physics starts.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FrameCallback = Callable[[float, int], None]


@dataclass(frozen=True, slots=True)
class RealRobotAssetConfig:
    countermeasure_asset: str = (
        "/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"
    )
    measurement_asset: str = "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
    countermeasure_root: str = "/World/CountermeasureRobot"
    measurement_root: str = "/World/MeasurementRobot"
    measurement_articulation: str = "/World/MeasurementRobot/chassis_link"
    detector_path: str = "/World/MeasurementRobot/chassis_link/Detector"
    panda_base_path: str = "/World/CountermeasureRobot/panda_link0"
    panda_hand_path: str = "/World/CountermeasureRobot/panda_hand"
    decon_tool_path: str = (
        "/World/CountermeasureRobot/panda_hand/RadCounterDeconTool/ContactPad"
    )
    shield_path: str = "/World/LeadShield"
    shield_grasp_path: str = "/World/LeadShield/ShieldGraspFrame"
    shield_grasp_offset_m: tuple[float, float, float] = (-0.12, 0.0, 0.59)
    decon_surface_path: str = "/World/DeconWorkSurface"


@dataclass(frozen=True, slots=True)
class HandMotionResult:
    success: bool
    steps: int
    target_position_m: tuple[float, float, float]
    final_position_m: tuple[float, float, float]
    position_error_m: float


@dataclass(frozen=True, slots=True)
class DecontaminationMotionReport:
    success: bool
    accepted_contacts: int
    rejected_contacts: int
    removed_activity_bq: float
    activity_before_bq: float
    activity_after_bq: float
    treated_triangle_indices: tuple[int, ...]
    waypoint_errors_m: tuple[float, ...]
    tool_path_length_m: float


@dataclass(frozen=True, slots=True)
class ShieldMotionReport:
    success: bool
    failed_phase: str | None
    grasp_distance_m: float
    finger_aperture_open_m: float
    finger_aperture_closed_m: float
    lift_height_m: float
    placement_error_m: float
    initial_position_m: tuple[float, float, float]
    final_position_m: tuple[float, float, float]
    base_waypoint_errors: tuple[float, ...]
    placement_hand_steps: tuple[int, ...]
    placement_hand_errors_m: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class MeasurementMotionReport:
    success: bool
    steps: int
    displacement_m: float
    initial_position_m: tuple[float, float, float]
    final_position_m: tuple[float, float, float]


def enable_real_robot_extensions() -> None:
    """Enable the two optional Isaac extensions used by the real controllers."""

    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.robot_motion.motion_generation")
    enable_extension("isaacsim.robot.wheeled_robots")


def _custom_attribute(prim: Any, name: str, value_type: Any, value: object) -> None:
    attribute = prim.GetAttribute(name)
    if not attribute:
        attribute = prim.CreateAttribute(name, value_type, custom=True)
    attribute.Set(value)


def _world_pose(stage: Any, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"required prim does not exist: {prim_path}")
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    position = np.asarray(matrix.Transform(Gf.Vec3d()), dtype=np.float64)
    quaternion = matrix.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    orientation = np.asarray(
        (
            quaternion.GetReal(),
            imaginary[0],
            imaginary[1],
            imaginary[2],
        ),
        dtype=np.float64,
    )
    return position, orientation


def _cube(
    stage: Any,
    path: str,
    *,
    translate: Sequence[float],
    half_scale: Sequence[float],
    color: Sequence[float],
    collision: bool = True,
) -> Any:
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(2.0)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, translate)))
    xform.AddScaleOp().Set(Gf.Vec3d(*map(float, half_scale)))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*map(float, color))])
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
    return cube.GetPrim()


def add_real_robot_references(
    stage: Any,
    *,
    assets_root: str | None = None,
    config: RealRobotAssetConfig | None = None,
) -> dict[str, str]:
    """Replace placeholder boxes with NVIDIA's official robot USD references."""

    from isaacsim.storage.native import get_assets_root_path
    from pxr import UsdGeom

    cfg = config or RealRobotAssetConfig()
    root = assets_root or get_assets_root_path()
    if not root:
        raise RuntimeError("Isaac Sim assets root is unavailable")
    for prim_path in (cfg.countermeasure_root, cfg.measurement_root):
        if stage.GetPrimAtPath(prim_path).IsValid():
            stage.RemovePrim(prim_path)
    countermeasure = UsdGeom.Xform.Define(stage, cfg.countermeasure_root).GetPrim()
    measurement = UsdGeom.Xform.Define(stage, cfg.measurement_root).GetPrim()
    countermeasure.GetReferences().AddReference(root + cfg.countermeasure_asset)
    measurement.GetReferences().AddReference(root + cfg.measurement_asset)
    return {
        "assets_root": root,
        "countermeasure_usd": root + cfg.countermeasure_asset,
        "measurement_usd": root + cfg.measurement_asset,
    }


def author_real_robot_task_scene(
    stage: Any,
    activity_map_path: str | Path,
    *,
    config: RealRobotAssetConfig | None = None,
) -> None:
    """Attach detector/tool hardware and author physically plausible task props."""

    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    cfg = config or RealRobotAssetConfig()
    required = (
        cfg.countermeasure_root,
        cfg.measurement_articulation,
        cfg.panda_hand_path,
    )
    missing = [path for path in required if not stage.GetPrimAtPath(path).IsValid()]
    if missing:
        raise RuntimeError(f"real robot references are not composed: {missing}")

    # The legacy vertical slice authors both a floor cube and a second collision
    # mesh 15 mm above it.  Keep that mesh as a radiation source, but remove the
    # duplicate contact plane so wheeled robots are not trapped between surfaces.
    legacy_surface = stage.GetPrimAtPath("/World/ContaminatedFloor")
    legacy_collision = legacy_surface.GetAttribute("physics:collisionEnabled")
    if legacy_collision:
        legacy_collision.Set(False)

    countermeasure = stage.GetPrimAtPath(cfg.countermeasure_root)
    measurement = stage.GetPrimAtPath(cfg.measurement_root)
    _custom_attribute(countermeasure, "rad:role", Sdf.ValueTypeNames.String, "countermeasure_robot")
    _custom_attribute(measurement, "rad:role", Sdf.ValueTypeNames.String, "measurement_robot")

    # The Ridgeback asset represents its holonomic base with x/y/yaw joints.
    # Increase their physical drives before the timeline starts so the base can
    # overcome contact friction on the concrete validation floor.
    base_drives = (
        (
            cfg.countermeasure_root + "/world/dummy_base_prismatic_x_joint",
            "linear",
            8.0e4,
            1.2e4,
            2.5e5,
        ),
        (
            cfg.countermeasure_root + "/dummy_base_x/dummy_base_prismatic_y_joint",
            "linear",
            8.0e4,
            1.2e4,
            2.5e5,
        ),
        (
            cfg.countermeasure_root + "/dummy_base_y/dummy_base_revolute_z_joint",
            "angular",
            5.0e4,
            8.0e3,
            1.0e5,
        ),
    )
    for joint_path, drive_name, stiffness, damping, maximum_force in base_drives:
        joint = stage.GetPrimAtPath(joint_path)
        if not joint or not joint.IsValid():
            raise RuntimeError(f"Ridgeback base joint is missing: {joint_path}")
        drive = UsdPhysics.DriveAPI.Apply(joint, drive_name)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(stiffness)
        drive.CreateDampingAttr(damping)
        drive.CreateMaxForceAttr(maximum_force)

    detector = UsdGeom.Xform.Define(stage, cfg.detector_path)
    detector.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.72))
    detector_prim = detector.GetPrim()
    _custom_attribute(detector_prim, "rad:role", Sdf.ValueTypeNames.String, "detector")
    _custom_attribute(
        detector_prim,
        "rad:detector:id",
        Sdf.ValueTypeNames.String,
        "gamma-counter",
    )
    detector_body = UsdGeom.Cylinder.Define(stage, cfg.detector_path + "/GammaDetector")
    detector_body.CreateAxisAttr("Z")
    detector_body.CreateRadiusAttr(0.11)
    detector_body.CreateHeightAttr(0.28)
    detector_body.CreateDisplayColorAttr([Gf.Vec3f(0.95, 0.68, 0.08)])

    tool_root_path = cfg.decon_tool_path.rsplit("/", 1)[0]
    UsdGeom.Xform.Define(stage, tool_root_path)
    contact = UsdGeom.Xform.Define(stage, cfg.decon_tool_path)
    contact.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.22))
    contact_prim = contact.GetPrim()
    _custom_attribute(contact_prim, "rad:role", Sdf.ValueTypeNames.String, "decon_tool")
    _custom_attribute(
        contact_prim,
        "rad:decon:toolAxis",
        Sdf.ValueTypeNames.Double3,
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    _cube(
        stage,
        tool_root_path + "/ToolShaft",
        translate=(0.0, 0.0, 0.11),
        half_scale=(0.025, 0.025, 0.11),
        color=(0.08, 0.24, 0.34),
        collision=False,
    )
    _cube(
        stage,
        cfg.decon_tool_path + "/Pad",
        translate=(0.0, 0.0, 0.0),
        half_scale=(0.095, 0.075, 0.012),
        color=(0.05, 0.78, 0.78),
        collision=False,
    )

    for path in ("/World/DeconWorkbench", cfg.decon_surface_path, cfg.shield_path):
        if stage.GetPrimAtPath(path).IsValid():
            stage.RemovePrim(path)
    workbench = UsdGeom.Xform.Define(stage, "/World/DeconWorkbench")
    _cube(
        stage,
        "/World/DeconWorkbench/Top",
        translate=(0.68, 0.0, 0.60),
        half_scale=(0.23, 0.30, 0.04),
        color=(0.16, 0.18, 0.19),
    )
    for index, (x, y) in enumerate(
        ((0.50, -0.23), (0.86, -0.23), (0.50, 0.23), (0.86, 0.23))
    ):
        _cube(
            stage,
            f"/World/DeconWorkbench/Leg{index}",
            translate=(x, y, 0.29),
            half_scale=(0.035, 0.035, 0.29),
            color=(0.10, 0.11, 0.12),
        )
    workbench.GetPrim().SetDisplayName("Contaminated work surface")

    x_values = (0.48, 0.68, 0.88)
    y_values = (-0.24, 0.0, 0.24)
    points = [Gf.Vec3f(x, y, 0.651) for y in y_values for x in x_values]
    indices: list[int] = []
    for row in range(2):
        for column in range(2):
            lower_left = row * 3 + column
            lower_right = lower_left + 1
            upper_left = lower_left + 3
            upper_right = upper_left + 1
            indices.extend(
                (lower_left, lower_right, upper_right, lower_left, upper_right, upper_left)
            )
    surface = UsdGeom.Mesh.Define(stage, cfg.decon_surface_path)
    surface.CreatePointsAttr(points)
    surface.CreateFaceVertexCountsAttr([3] * 8)
    surface.CreateFaceVertexIndicesAttr(indices)
    surface.CreateSubdivisionSchemeAttr("none")
    surface.CreateDisplayColorAttr([Gf.Vec3f(0.78, 0.18, 0.08)])
    UsdPhysics.CollisionAPI.Apply(surface.GetPrim()).CreateCollisionEnabledAttr(True)
    surface_prim = surface.GetPrim()
    activity_path = Path(activity_map_path).resolve()
    digest = hashlib.sha256(activity_path.read_bytes()).hexdigest()
    for name, value_type, value in (
        ("rad:role", Sdf.ValueTypeNames.String, "contaminated_surface"),
        ("rad:source:type", Sdf.ValueTypeNames.String, "surface"),
        ("rad:source:isotopeId", Sdf.ValueTypeNames.String, "Cs-137"),
        ("rad:source:activityMapUri", Sdf.ValueTypeNames.String, str(activity_path)),
        ("rad:source:activityMapSha256", Sdf.ValueTypeNames.String, digest),
        ("rad:source:hiddenFromEstimator", Sdf.ValueTypeNames.Bool, False),
        ("rad:source:enabled", Sdf.ValueTypeNames.Bool, True),
        ("rad:decon:enabled", Sdf.ValueTypeNames.Bool, True),
        ("rad:decon:activityMapUri", Sdf.ValueTypeNames.String, str(activity_path)),
        ("rad:decon:activityMapSha256", Sdf.ValueTypeNames.String, digest),
    ):
        _custom_attribute(surface_prim, name, value_type, value)

    shield = UsdGeom.Xform.Define(stage, cfg.shield_path)
    shield.AddTranslateOp().Set(Gf.Vec3d(0.78, -0.62, 0.0))
    shield_prim = shield.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(shield_prim).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(shield_prim).CreateMassAttr(2.2)
    for name, value_type, value in (
        ("rad:role", Sdf.ValueTypeNames.String, "shield"),
        ("rad:material:id", Sdf.ValueTypeNames.String, "lead"),
        ("rad:material:mode", Sdf.ValueTypeNames.String, "solid"),
        ("rad:shield:movable", Sdf.ValueTypeNames.Bool, True),
        ("rad:shield:resourceUnits", Sdf.ValueTypeNames.Int, 1),
        ("rad:manipulation:movable", Sdf.ValueTypeNames.Bool, True),
        ("rad:manipulation:removable", Sdf.ValueTypeNames.Bool, False),
        ("rad:manipulation:graspFrame", Sdf.ValueTypeNames.String, "ShieldGraspFrame"),
    ):
        _custom_attribute(shield_prim, name, value_type, value)
    _cube(
        stage,
        cfg.shield_path + "/Base",
        translate=(0.0, 0.0, 0.035),
        half_scale=(0.14, 0.20, 0.035),
        color=(0.11, 0.12, 0.13),
    )
    plate = _cube(
        stage,
        cfg.shield_path + "/Plate",
        translate=(0.0, 0.0, 0.55),
        half_scale=(0.025, 0.22, 0.35),
        color=(0.20, 0.22, 0.24),
    )
    _custom_attribute(plate, "rad:material:id", Sdf.ValueTypeNames.String, "lead")
    _custom_attribute(plate, "rad:material:mode", Sdf.ValueTypeNames.String, "solid")
    _cube(
        stage,
        cfg.shield_path + "/Handle",
        translate=cfg.shield_grasp_offset_m,
        half_scale=(0.10, 0.015, 0.018),
        color=(0.92, 0.56, 0.08),
    )
    grasp = UsdGeom.Xform.Define(stage, cfg.shield_grasp_path)
    grasp.AddTranslateOp().Set(Gf.Vec3d(*cfg.shield_grasp_offset_m))
    UsdGeom.Xform.Define(stage, "/World/DeconCleanTrace")


class RidgebackFrankaController:
    """Control every Ridgeback+Franka DOF through one Isaac articulation."""

    base_joint_names = (
        "dummy_base_prismatic_x_joint",
        "dummy_base_prismatic_y_joint",
        "dummy_base_revolute_z_joint",
    )
    arm_joint_names = tuple(f"panda_joint{index}" for index in range(1, 8))
    finger_joint_names = ("panda_finger_joint1", "panda_finger_joint2")
    downward_orientation_wxyz = np.asarray((0.0, 1.0, 0.0, 0.0), dtype=np.float64)

    def __init__(
        self,
        stage: Any,
        stepper: Any,
        *,
        config: RealRobotAssetConfig | None = None,
        articulation: Any | None = None,
    ) -> None:
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.robot_motion.motion_generation import (
            ArticulationKinematicsSolver,
            LulaKinematicsSolver,
        )
        from isaacsim.robot_motion.motion_generation.interface_config_loader import (
            load_supported_lula_kinematics_solver_config,
        )

        self.stage = stage
        self.stepper = stepper
        self.config = config or RealRobotAssetConfig()
        self.robot = articulation or SingleArticulation(
            self.config.countermeasure_root, name="radcounter_ridgeback_franka"
        )
        if articulation is None:
            self.robot.initialize()
        self.controller = self.robot.get_articulation_controller()
        self.controller.switch_control_mode(mode="position")
        self.dof_names = tuple(self.robot.dof_names)
        expected = set(self.base_joint_names + self.arm_joint_names + self.finger_joint_names)
        missing = sorted(expected.difference(self.dof_names))
        if missing:
            raise RuntimeError(f"Ridgeback+Franka DOFs are missing: {missing}")
        self.indices = {name: self.robot.get_dof_index(name) for name in expected}
        lula_config = load_supported_lula_kinematics_solver_config("Franka")
        self._lula = LulaKinematicsSolver(**lula_config)
        self._solver = ArticulationKinematicsSolver(
            self.robot,
            self._lula,
            "right_gripper",
        )
        arm = self._arm_positions()
        self._arm_min = arm.copy()
        self._arm_max = arm.copy()
        self._grasp_joint_path = (
            self.config.panda_hand_path + "/RadCounterShieldGraspJoint"
        )
        self._handle_collision_enabled = True
        self.last_base_target = np.zeros(3, dtype=np.float64)
        self.last_base_positions = self._positions()[
            [self.indices[name] for name in self.base_joint_names]
        ].copy()

    @staticmethod
    def _smoothstep(value: float) -> float:
        return value * value * (3.0 - 2.0 * value)

    def _positions(self) -> np.ndarray:
        return np.asarray(self.robot.get_joint_positions(), dtype=np.float64)

    def _arm_positions(self) -> np.ndarray:
        positions = self._positions()
        return positions[[self.indices[name] for name in self.arm_joint_names]]

    def _observe_arm(self) -> None:
        arm = self._arm_positions()
        self._arm_min = np.minimum(self._arm_min, arm)
        self._arm_max = np.maximum(self._arm_max, arm)

    @property
    def arm_joint_excursion_rad(self) -> float:
        return float(np.max(self._arm_max - self._arm_min))

    def _step(self, callback: FrameCallback | None, frame: int) -> None:
        self.stepper.step(render=True)
        self._observe_arm()
        if callback is not None:
            callback(1.0 / 60.0, frame)

    def _sync_kinematics_base(self) -> None:
        position, orientation = _world_pose(self.stage, self.config.panda_base_path)
        self._lula.set_robot_base_pose(position, orientation)

    def end_effector_position(self) -> np.ndarray:
        self._sync_kinematics_base()
        position, _ = self._solver.compute_end_effector_pose(position_only=True)
        return np.asarray(position, dtype=np.float64)

    def move_base(
        self,
        target_xy_yaw: Sequence[float],
        *,
        interpolation_steps: int = 90,
        settle_steps: int = 180,
        tolerance: float = 0.025,
    ) -> bool:
        from isaacsim.core.utils.types import ArticulationAction

        target = np.asarray(target_xy_yaw, dtype=np.float64)
        if target.shape != (3,):
            raise ValueError("target_xy_yaw must contain x, y, and yaw")
        indices = np.asarray([self.indices[name] for name in self.base_joint_names], dtype=np.int32)
        self.last_base_target = target.copy()
        start = self._positions()[indices]
        for frame in range(1, interpolation_steps + 1):
            fraction = self._smoothstep(frame / interpolation_steps)
            command = start + fraction * (target - start)
            self.controller.apply_action(
                ArticulationAction(joint_positions=command, joint_indices=indices)
            )
            self._step(None, frame)
        for frame in range(1, settle_steps + 1):
            self.controller.apply_action(
                ArticulationAction(joint_positions=target, joint_indices=indices)
            )
            self._step(None, interpolation_steps + frame)
            if np.linalg.norm(self._positions()[indices] - target) <= tolerance:
                self.last_base_positions = self._positions()[indices].copy()
                return True
        self.last_base_positions = self._positions()[indices].copy()
        return False

    def move_hand(
        self,
        target_position_m: Sequence[float],
        orientation_wxyz: Sequence[float] | None = None,
        *,
        interpolation_steps: int = 75,
        settle_steps: int = 180,
        tolerance_m: float = 0.012,
        callback: FrameCallback | None = None,
    ) -> HandMotionResult:
        from isaacsim.core.utils.types import ArticulationAction

        target = np.asarray(target_position_m, dtype=np.float64)
        orientation = np.asarray(
            self.downward_orientation_wxyz
            if orientation_wxyz is None
            else orientation_wxyz,
            dtype=np.float64,
        )
        self._sync_kinematics_base()
        action, success = self._solver.compute_inverse_kinematics(
            target,
            orientation,
            position_tolerance=tolerance_m / 2.0,
            orientation_tolerance=0.08,
        )
        if not success or action.joint_positions is None or action.joint_indices is None:
            actual = self.end_effector_position()
            return HandMotionResult(
                False,
                0,
                tuple(map(float, target)),
                tuple(map(float, actual)),
                float(np.linalg.norm(actual - target)),
            )
        indices = np.asarray(action.joint_indices, dtype=np.int32)
        target_joints = np.asarray(action.joint_positions, dtype=np.float64)
        start = self._positions()[indices]
        frame_number = 0
        for frame in range(1, interpolation_steps + 1):
            frame_number += 1
            fraction = self._smoothstep(frame / interpolation_steps)
            command = start + fraction * (target_joints - start)
            self.controller.apply_action(
                ArticulationAction(joint_positions=command, joint_indices=indices)
            )
            self._step(callback, frame_number)
        actual = self.end_effector_position()
        for _ in range(settle_steps):
            if np.linalg.norm(actual - target) <= tolerance_m:
                break
            frame_number += 1
            self.controller.apply_action(action)
            self._step(callback, frame_number)
            actual = self.end_effector_position()
        error = float(np.linalg.norm(actual - target))
        return HandMotionResult(
            error <= tolerance_m,
            frame_number,
            tuple(map(float, target)),
            tuple(map(float, actual)),
            error,
        )

    def hold(self, frames: int, callback: FrameCallback | None = None) -> None:
        for frame in range(1, frames + 1):
            self._step(callback, frame)

    def set_gripper(self, finger_position_m: float, *, steps: int = 45) -> float:
        from isaacsim.core.utils.types import ArticulationAction

        target = np.asarray((finger_position_m, finger_position_m), dtype=np.float64)
        indices = np.asarray(
            [self.indices[name] for name in self.finger_joint_names], dtype=np.int32
        )
        start = self._positions()[indices]
        for frame in range(1, steps + 1):
            fraction = self._smoothstep(frame / steps)
            command = start + fraction * (target - start)
            self.controller.apply_action(
                ArticulationAction(joint_positions=command, joint_indices=indices)
            )
            self._step(None, frame)
        return float(np.sum(self._positions()[indices]))

    def set_decon_tool_visible(self, visible: bool) -> None:
        from pxr import UsdGeom

        root_path = self.config.decon_tool_path.rsplit("/", 1)[0]
        imageable = UsdGeom.Imageable(self.stage.GetPrimAtPath(root_path))
        if visible:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    def _clean_trace_patch(self, index: int) -> None:
        tool_position, _ = _world_pose(self.stage, self.config.decon_tool_path)
        _cube(
            self.stage,
            f"/World/DeconCleanTrace/Patch{index:02d}",
            translate=(tool_position[0], tool_position[1], 0.655),
            half_scale=(0.09, 0.065, 0.002),
            color=(0.10, 0.72, 0.58),
            collision=False,
        )

    def execute_decontamination(
        self,
        decontaminator: Any,
        hand_waypoints_m: Sequence[Sequence[float]],
        *,
        dwell_frames: int = 32,
    ) -> DecontaminationMotionReport:
        if not hand_waypoints_m:
            raise ValueError("at least one decontamination waypoint is required")
        self.set_decon_tool_visible(True)
        activity_before = float(np.sum(decontaminator.activity_bq))
        first = np.asarray(hand_waypoints_m[0], dtype=np.float64)
        approach = first + np.asarray((0.0, 0.0, 0.16))
        approach_result = self.move_hand(approach)
        if not approach_result.success:
            return DecontaminationMotionReport(
                False,
                0,
                0,
                0.0,
                activity_before,
                activity_before,
                (),
                (approach_result.position_error_m,),
                0.0,
            )
        accepted = 0
        rejected = 0
        removed = 0.0
        triangles: set[int] = set()
        errors: list[float] = []
        simulation_step = 0
        tool_positions: list[np.ndarray] = []

        def treatment_tick(dt_s: float, _: int) -> None:
            nonlocal accepted, rejected, removed, simulation_step
            simulation_step += 1
            result = decontaminator.tick(dt_s, simulation_step)
            accepted += result.accepted_contacts
            rejected += result.rejected_contacts
            removed += result.removed_activity_bq
            triangles.update(result.treated_triangle_indices)
            position, _ = _world_pose(self.stage, self.config.decon_tool_path)
            tool_positions.append(position)

        for index, waypoint in enumerate(hand_waypoints_m):
            contact_before = accepted
            motion = self.move_hand(waypoint, callback=treatment_tick)
            errors.append(motion.position_error_m)
            if not motion.success:
                decontaminator.flush()
                return DecontaminationMotionReport(
                    False,
                    accepted,
                    rejected,
                    removed,
                    activity_before,
                    float(np.sum(decontaminator.activity_bq)),
                    tuple(sorted(triangles)),
                    tuple(errors),
                    0.0,
                )
            self.hold(dwell_frames, treatment_tick)
            if accepted > contact_before:
                self._clean_trace_patch(index)
        retreat = np.asarray(hand_waypoints_m[-1], dtype=np.float64) + np.asarray(
            (0.0, 0.0, 0.18)
        )
        self.move_hand(retreat)
        decontaminator.flush()
        path_length = 0.0
        for first_position, second_position in zip(
            tool_positions, tool_positions[1:], strict=False
        ):
            path_length += float(np.linalg.norm(second_position - first_position))
        activity_after = float(np.sum(decontaminator.activity_bq))
        return DecontaminationMotionReport(
            accepted > 0 and removed > 0.0 and activity_after < activity_before,
            accepted,
            rejected,
            removed,
            activity_before,
            activity_after,
            tuple(sorted(triangles)),
            tuple(errors),
            path_length,
        )

    def _attach_shield(self) -> float:
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics

        hand_position = self.end_effector_position()
        grasp_position, _ = _world_pose(self.stage, self.config.shield_grasp_path)
        distance = float(np.linalg.norm(hand_position - grasp_position))
        if distance > 0.055:
            raise RuntimeError(f"shield grasp frame is {distance:.3f} m from the gripper")
        hand = self.stage.GetPrimAtPath(self.config.panda_hand_path)
        shield = self.stage.GetPrimAtPath(self.config.shield_path)
        cache = UsdGeom.XformCache()
        hand_world = cache.GetLocalToWorldTransform(hand)
        shield_world = cache.GetLocalToWorldTransform(shield)
        grasp_world = cache.GetLocalToWorldTransform(
            self.stage.GetPrimAtPath(self.config.shield_grasp_path)
        )
        anchor_world = grasp_world.Transform(Gf.Vec3d())
        hand_anchor = hand_world.GetInverse().Transform(anchor_world)
        shield_anchor = shield_world.GetInverse().Transform(anchor_world)
        hand_rotation = hand_world.ExtractRotationQuat()
        shield_rotation = shield_world.ExtractRotationQuat()
        local_rotation = hand_rotation.GetInverse() * shield_rotation
        imaginary = local_rotation.GetImaginary()
        joint = UsdPhysics.FixedJoint.Define(self.stage, self._grasp_joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(self.config.panda_hand_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(self.config.shield_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*hand_anchor))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*shield_anchor))
        joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(
                float(local_rotation.GetReal()),
                Gf.Vec3f(*map(float, imaginary)),
            )
        )
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        handle = self.stage.GetPrimAtPath(self.config.shield_path + "/Handle")
        collision = handle.GetAttribute("physics:collisionEnabled")
        self._handle_collision_enabled = bool(collision.Get())
        collision.Set(False)
        self._step(None, 1)
        return distance

    def _release_shield(self) -> None:
        self.stage.RemovePrim(self._grasp_joint_path)
        handle = self.stage.GetPrimAtPath(self.config.shield_path + "/Handle")
        handle.GetAttribute("physics:collisionEnabled").Set(self._handle_collision_enabled)
        self._step(None, 1)

    def execute_shield_pick_and_place(
        self,
        *,
        pickup_base_xy_yaw: Sequence[float],
        placement_base_route_xy_yaw: Sequence[Sequence[float]],
        destination_root_position_m: Sequence[float],
    ) -> ShieldMotionReport:
        self.set_decon_tool_visible(False)
        initial, _ = _world_pose(self.stage, self.config.shield_path)
        if not self.move_base(pickup_base_xy_yaw):
            return ShieldMotionReport(
                False, "pickup_base", math.inf, 0.0, 0.0, 0.0, math.inf,
                tuple(map(float, initial)), tuple(map(float, initial)),
                (float(np.linalg.norm(self.last_base_positions - self.last_base_target)),),
                (),
                (),
            )
        open_aperture = self.set_gripper(0.035)
        grasp_position, _ = _world_pose(self.stage, self.config.shield_grasp_path)
        approach = grasp_position + np.asarray((0.0, 0.0, 0.16))
        if not self.move_hand(approach).success or not self.move_hand(grasp_position).success:
            return ShieldMotionReport(
                False, "pickup_hand", math.inf, open_aperture, 0.0, 0.0, math.inf,
                tuple(map(float, initial)), tuple(map(float, initial)), ()
                , (), ()
            )
        closed_aperture = self.set_gripper(0.004)
        grasp_distance = self._attach_shield()
        lift_target = grasp_position + np.asarray((0.0, 0.0, 0.24))
        if not self.move_hand(lift_target).success:
            self._release_shield()
            return ShieldMotionReport(
                False, "lift", grasp_distance, open_aperture, closed_aperture, 0.0, math.inf,
                tuple(map(float, initial)), tuple(map(float, initial)), ()
                , (), ()
            )
        lifted, _ = _world_pose(self.stage, self.config.shield_path)
        base_errors: list[float] = []
        for waypoint in placement_base_route_xy_yaw:
            if not self.move_base(waypoint):
                base_error = float(
                    np.linalg.norm(self.last_base_positions - self.last_base_target)
                )
                base_errors.append(base_error)
                self._release_shield()
                final, _ = _world_pose(self.stage, self.config.shield_path)
                return ShieldMotionReport(
                    False,
                    f"placement_base_{len(base_errors)}",
                    grasp_distance,
                    open_aperture,
                    closed_aperture,
                    float(lifted[2] - initial[2]),
                    math.inf,
                    tuple(map(float, initial)),
                    tuple(map(float, final)),
                    tuple(base_errors),
                    (),
                    (),
                )
            base_errors.append(
                float(np.linalg.norm(self.last_base_positions - self.last_base_target))
            )
        destination_root = np.asarray(destination_root_position_m, dtype=np.float64)
        release_root = destination_root + np.asarray((0.0, 0.0, 0.06))
        destination_grasp = release_root + np.asarray(
            self.config.shield_grasp_offset_m, dtype=np.float64
        )
        preplace = destination_grasp + np.asarray((0.0, 0.0, 0.12))
        loaded_tolerance_m = 0.025
        placement_motions = [
            self.move_hand(preplace, tolerance_m=loaded_tolerance_m)
        ]
        if placement_motions[0].success:
            for height_offset in (0.08, 0.04, 0.0):
                placement_motions.append(
                    self.move_hand(
                        destination_grasp + np.asarray((0.0, 0.0, height_offset)),
                        tolerance_m=loaded_tolerance_m,
                    )
                )
                if not placement_motions[-1].success:
                    break
        if not all(motion.success for motion in placement_motions):
            self._release_shield()
            final, _ = _world_pose(self.stage, self.config.shield_path)
            return ShieldMotionReport(
                False,
                "placement_hand",
                grasp_distance,
                open_aperture,
                closed_aperture,
                float(lifted[2] - initial[2]),
                math.inf,
                tuple(map(float, initial)),
                tuple(map(float, final)),
                tuple(base_errors),
                tuple(motion.steps for motion in placement_motions),
                tuple(motion.position_error_m for motion in placement_motions),
            )
        self.set_gripper(0.035)
        self._release_shield()
        self.hold(90)
        final, _ = _world_pose(self.stage, self.config.shield_path)
        placement_error = float(np.linalg.norm(final - destination_root))
        self.move_hand(preplace)
        return ShieldMotionReport(
            placement_error <= 0.12,
            None if placement_error <= 0.12 else "placement_settle",
            grasp_distance,
            open_aperture,
            closed_aperture,
            float(lifted[2] - initial[2]),
            placement_error,
            tuple(map(float, initial)),
            tuple(map(float, final)),
            tuple(base_errors),
            tuple(motion.steps for motion in placement_motions),
            tuple(motion.position_error_m for motion in placement_motions),
        )


class NovaCarterController:
    """Differential-wheel controller for the official Nova Carter articulation."""

    wheel_names = ("joint_wheel_left", "joint_wheel_right")

    def __init__(
        self,
        stage: Any,
        stepper: Any,
        *,
        config: RealRobotAssetConfig | None = None,
        wheel_radius_m: float = 0.14,
        wheel_base_m: float = 0.413,
        articulation: Any | None = None,
    ) -> None:
        from isaacsim.robot.wheeled_robots.robots import WheeledRobot

        self.stage = stage
        self.stepper = stepper
        self.config = config or RealRobotAssetConfig()
        self.wheel_radius_m = wheel_radius_m
        self.wheel_base_m = wheel_base_m
        self.robot = articulation or WheeledRobot(
            prim_path=self.config.measurement_articulation,
            name="radcounter_nova_carter",
            wheel_dof_names=list(self.wheel_names),
        )
        if articulation is None:
            self.robot.initialize()
        self.robot.get_articulation_controller().switch_control_mode(mode="velocity")
        missing = [name for name in self.wheel_names if name not in self.robot.dof_names]
        if missing:
            raise RuntimeError(f"Nova Carter wheel DOFs are missing: {missing}")

    @staticmethod
    def _yaw(orientation_wxyz: np.ndarray) -> float:
        w, x, y, z = orientation_wxyz
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _wrap(value: float) -> float:
        return math.atan2(math.sin(value), math.cos(value))

    def set_initial_pose(
        self,
        position_m: Sequence[float],
        orientation_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    ) -> None:
        self.robot.set_world_pose(
            position=np.asarray(position_m, dtype=np.float64),
            orientation=np.asarray(orientation_wxyz, dtype=np.float64),
        )
        self.robot.set_linear_velocity(np.zeros(3, dtype=np.float64))
        self.robot.set_angular_velocity(np.zeros(3, dtype=np.float64))
        self.stop()
        for _ in range(20):
            self.stepper.step(render=True)

    def _command(self, linear_m_s: float, angular_rad_s: float) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        left = (
            linear_m_s - angular_rad_s * self.wheel_base_m / 2.0
        ) / self.wheel_radius_m
        right = (
            linear_m_s + angular_rad_s * self.wheel_base_m / 2.0
        ) / self.wheel_radius_m
        self.robot.apply_wheel_actions(
            ArticulationAction(joint_velocities=np.asarray((left, right), dtype=np.float64))
        )

    def stop(self) -> None:
        self._command(0.0, 0.0)

    def navigate_to(
        self,
        target_xy_m: Sequence[float],
        *,
        tolerance_m: float = 0.12,
        maximum_steps: int = 1200,
    ) -> MeasurementMotionReport:
        target = np.asarray(target_xy_m, dtype=np.float64)
        initial, _ = _world_pose(self.stage, self.config.measurement_articulation)
        success = False
        steps = 0
        for step in range(1, maximum_steps + 1):
            steps = step
            position, orientation = _world_pose(
                self.stage, self.config.measurement_articulation
            )
            error = target[:2] - position[:2]
            distance = float(np.linalg.norm(error))
            if distance <= tolerance_m:
                success = True
                break
            desired_yaw = math.atan2(error[1], error[0])
            yaw_error = self._wrap(desired_yaw - self._yaw(orientation))
            angular = float(np.clip(2.2 * yaw_error, -1.2, 1.2))
            linear = min(0.55, 0.9 * distance)
            if abs(yaw_error) > 0.45:
                linear = 0.0
            self._command(linear, angular)
            self.stepper.step(render=True)
        self.stop()
        for _ in range(20):
            self.stepper.step(render=True)
        final, _ = _world_pose(self.stage, self.config.measurement_articulation)
        displacement = float(np.linalg.norm(final[:2] - initial[:2]))
        return MeasurementMotionReport(
            success,
            steps,
            displacement,
            tuple(map(float, initial)),
            tuple(map(float, final)),
        )


__all__ = [
    "DecontaminationMotionReport",
    "HandMotionResult",
    "MeasurementMotionReport",
    "NovaCarterController",
    "RealRobotAssetConfig",
    "RidgebackFrankaController",
    "ShieldMotionReport",
    "add_real_robot_references",
    "author_real_robot_task_scene",
    "enable_real_robot_extensions",
]
