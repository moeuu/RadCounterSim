"""Simulator-independent descriptors for arbitrary articulated robots."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .sensors import SensorMountConfig


class RobotFormat(StrEnum):
    AUTO = "auto"
    USD = "usd"
    URDF = "urdf"
    XACRO = "xacro"
    MJCF = "mjcf"


class RobotType(StrEnum):
    DEFAULT = "default"
    END_EFFECTOR = "end_effector"
    MANIPULATOR = "manipulator"
    HUMANOID = "humanoid"
    WHEELED = "wheeled"
    HOLONOMIC = "holonomic"
    QUADRUPED = "quadruped"
    MOBILE_MANIPULATOR = "mobile_manipulator"
    AERIAL = "aerial"
    FLOATING = "floating"


class JointControlMode(StrEnum):
    POSITION = "position"
    VELOCITY = "velocity"
    EFFORT = "effort"


class RobotGeometryFidelity(StrEnum):
    """Provenance level of geometry supplied for a real robot."""

    MANUFACTURER_ASSET = "manufacturer_asset"
    LICENSED_CAD = "licensed_cad"
    REFERENCE_PROCEDURAL = "reference_procedural"
    USER_SUPPLIED = "user_supplied"


class RobotReferenceConfig(BaseModel):
    """Traceable real-machine reference for a simulator robot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    manufacturer: str
    model: str
    source_urls: tuple[str, ...]
    geometry_fidelity: RobotGeometryFidelity
    dimensions_m: tuple[float, float, float] | None = None
    mass_kg: float | None = None

    def model_post_init(self, __context: Any) -> None:
        if not self.source_urls:
            raise ValueError("A real-robot reference requires at least one source URL")
        if any(not url.startswith(("https://", "http://")) for url in self.source_urls):
            raise ValueError("Robot reference sources must be HTTP(S) URLs")
        if self.dimensions_m is not None and any(value <= 0 for value in self.dimensions_m):
            raise ValueError("Robot reference dimensions must be positive")
        if self.mass_kg is not None and self.mass_kg <= 0:
            raise ValueError("Robot reference mass must be positive")


class JointGroupConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    joint_names: tuple[str, ...] = ()
    joint_patterns: tuple[str, ...] = ()
    mode: JointControlMode = JointControlMode.POSITION
    stiffness: float | None = None
    damping: float | None = None
    maximum_effort: float | None = None
    maximum_velocity: float | None = None


class BaseControllerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = "direct"
    joint_names: tuple[str, ...] = ()
    wheel_radius_m: float | None = None
    track_width_m: float | None = None
    wheelbase_m: float | None = None
    twist_to_joint_velocity: tuple[tuple[float, float, float], ...] = ()
    maximum_linear_speed_m_s: float | None = None
    maximum_vertical_speed_m_s: float | None = None
    maximum_angular_speed_rad_s: float | None = None
    maximum_roll_pitch_rate_rad_s: float | None = None
    spatial_to_joint_velocity: tuple[tuple[float, float, float, float, float, float], ...] = ()


class GripperConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = "gripper"
    joint_names: tuple[str, ...]
    open_positions: tuple[float, ...]
    closed_positions: tuple[float, ...]

    def model_post_init(self, __context: Any) -> None:
        lengths = {
            len(self.joint_names),
            len(self.open_positions),
            len(self.closed_positions),
        }
        if len(lengths) != 1:
            raise ValueError("Gripper names/open/closed arrays must have equal lengths")


class LulaKinematicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    robot_description_path: str
    urdf_path: str
    end_effector_frame: str


class RobotImportConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    id: str
    uri: str
    format: RobotFormat = RobotFormat.AUTO
    prim_path: str | None = None
    robot_type: RobotType = RobotType.DEFAULT
    fixed_base: bool | None = None
    merge_fixed_joints: bool = False
    merge_meshes: bool = False
    collision_from_visuals: bool = False
    collision_type: str = "Convex Hull"
    allow_self_collision: bool = False
    package_paths: tuple[str, ...] = ()
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    joint_groups: tuple[JointGroupConfig, ...] = ()
    base_controller: BaseControllerConfig | None = None
    grippers: tuple[GripperConfig, ...] = ()
    lula_kinematics: LulaKinematicsConfig | None = None
    reference: RobotReferenceConfig | None = None
    sensor_mounts: tuple[SensorMountConfig, ...] = Field(default=(), alias="sensors")
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        sensor_ids = [sensor.id for sensor in self.sensor_mounts]
        if len(sensor_ids) != len(set(sensor_ids)):
            raise ValueError(f"Robot {self.id!r} has duplicate sensor IDs")

    @property
    def resolved_format(self) -> RobotFormat:
        if self.format is not RobotFormat.AUTO:
            return self.format
        suffix = Path(self.uri.split("?", 1)[0]).suffix.lower()
        formats = {
            ".usd": RobotFormat.USD,
            ".usda": RobotFormat.USD,
            ".usdc": RobotFormat.USD,
            ".urdf": RobotFormat.URDF,
            ".xacro": RobotFormat.XACRO,
            ".xml": RobotFormat.MJCF,
            ".mjcf": RobotFormat.MJCF,
        }
        if suffix not in formats:
            raise ValueError(f"Cannot infer robot format from {self.uri!r}")
        return formats[suffix]


class RobotFleetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    robots: tuple[RobotImportConfig, ...]

    def model_post_init(self, __context: Any) -> None:
        ids = [robot.id for robot in self.robots]
        if len(ids) != len(set(ids)):
            raise ValueError("Robot IDs must be unique within a fleet")


def load_robot_fleet(path: str | Path) -> RobotFleetConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    return RobotFleetConfig.model_validate(payload)
