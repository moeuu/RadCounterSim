"""Generic joint, base, and gripper command translation."""

from __future__ import annotations

import fnmatch
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .models import BaseControllerConfig, GripperConfig, JointGroupConfig


@dataclass(frozen=True)
class TwistCommand:
    linear_x_m_s: float = 0.0
    linear_y_m_s: float = 0.0
    angular_z_rad_s: float = 0.0


@dataclass(frozen=True)
class SpatialVelocityCommand:
    """Six-degree-of-freedom body velocity for aerial and floating robots."""

    linear_x_m_s: float = 0.0
    linear_y_m_s: float = 0.0
    linear_z_m_s: float = 0.0
    angular_x_rad_s: float = 0.0
    angular_y_rad_s: float = 0.0
    angular_z_rad_s: float = 0.0


@dataclass(frozen=True)
class JointCommand:
    names: tuple[str, ...]
    values: tuple[float, ...]
    mode: str


@dataclass(frozen=True)
class RobotCapabilities:
    dof_names: tuple[str, ...]
    groups: Mapping[str, tuple[str, ...]]
    has_base_controller: bool
    grippers: tuple[str, ...]
    has_inverse_kinematics: bool
    unresolved_groups: tuple[str, ...]
    supports_spatial_velocity: bool = False


class RobotControllerPlugin(Protocol):
    def command_joint(self, command: JointCommand) -> None: ...
    def command_twist(self, command: TwistCommand) -> None: ...
    def command_spatial_velocity(self, command: SpatialVelocityCommand) -> None: ...
    def command_gripper(self, name: str, fraction_closed: float) -> None: ...


class ControllerPluginRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., RobotControllerPlugin]] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., RobotControllerPlugin],
    ) -> None:
        if name in self._factories:
            raise ValueError(f"Controller plugin {name!r} is already registered")
        self._factories[name] = factory

    def create(self, name: str, **kwargs: object) -> RobotControllerPlugin:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise KeyError(f"Unknown controller plugin: {name}") from exc
        return factory(**kwargs)


def resolve_joint_groups(
    dof_names: Sequence[str],
    groups: Sequence[JointGroupConfig],
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    available = tuple(dof_names)
    resolved: dict[str, tuple[str, ...]] = {}
    unresolved = []
    for group in groups:
        requested = list(group.joint_names)
        for pattern in group.joint_patterns:
            requested.extend(name for name in available if fnmatch.fnmatchcase(name, pattern))
        unique = tuple(dict.fromkeys(name for name in requested if name in available))
        resolved[group.name] = unique
        if not unique:
            unresolved.append(group.name)
    return resolved, tuple(unresolved)


def twist_to_joint_velocity(
    command: TwistCommand,
    config: BaseControllerConfig,
) -> np.ndarray:
    twist = np.asarray(
        [command.linear_x_m_s, command.linear_y_m_s, command.angular_z_rad_s],
        dtype=np.float64,
    )
    if config.maximum_linear_speed_m_s is not None:
        twist[:2] = np.clip(
            twist[:2],
            -config.maximum_linear_speed_m_s,
            config.maximum_linear_speed_m_s,
        )
    if config.maximum_angular_speed_rad_s is not None:
        twist[2] = np.clip(
            twist[2],
            -config.maximum_angular_speed_rad_s,
            config.maximum_angular_speed_rad_s,
        )

    if config.twist_to_joint_velocity:
        matrix = np.asarray(config.twist_to_joint_velocity, dtype=np.float64)
        if matrix.shape != (len(config.joint_names), 3):
            raise ValueError("twist_to_joint_velocity must have shape (joint count, 3)")
        return matrix @ twist

    controller_type = config.type.lower()
    if controller_type == "differential":
        if config.wheel_radius_m is None or config.track_width_m is None:
            raise ValueError("Differential base requires wheel_radius_m and track_width_m")
        if len(config.joint_names) != 2:
            raise ValueError("Differential base requires left and right wheel joints")
        left = (twist[0] - 0.5 * config.track_width_m * twist[2]) / config.wheel_radius_m
        right = (twist[0] + 0.5 * config.track_width_m * twist[2]) / config.wheel_radius_m
        return np.asarray([left, right])

    if controller_type == "ackermann":
        if len(config.joint_names) != 2:
            raise ValueError("Ackermann base expects drive and steering joints")
        if config.wheel_radius_m is None or config.wheelbase_m is None:
            raise ValueError("Ackermann base requires wheel_radius_m and wheelbase_m")
        drive = twist[0] / config.wheel_radius_m
        steering = np.arctan2(config.wheelbase_m * twist[2], max(abs(twist[0]), 1e-9))
        return np.asarray([drive, steering])

    if controller_type in {"direct", "floating"}:
        return twist
    raise ValueError(f"Base type {config.type!r} needs a twist_to_joint_velocity matrix")


def spatial_to_joint_velocity(
    command: SpatialVelocityCommand,
    config: BaseControllerConfig,
) -> np.ndarray:
    """Map a 6-DoF command to a floating base or custom flight controller."""

    velocity = np.asarray(
        [
            command.linear_x_m_s,
            command.linear_y_m_s,
            command.linear_z_m_s,
            command.angular_x_rad_s,
            command.angular_y_rad_s,
            command.angular_z_rad_s,
        ],
        dtype=np.float64,
    )
    if config.maximum_linear_speed_m_s is not None:
        velocity[:2] = np.clip(
            velocity[:2],
            -config.maximum_linear_speed_m_s,
            config.maximum_linear_speed_m_s,
        )
    if config.maximum_vertical_speed_m_s is not None:
        velocity[2] = np.clip(
            velocity[2],
            -config.maximum_vertical_speed_m_s,
            config.maximum_vertical_speed_m_s,
        )
    if config.maximum_roll_pitch_rate_rad_s is not None:
        velocity[3:5] = np.clip(
            velocity[3:5],
            -config.maximum_roll_pitch_rate_rad_s,
            config.maximum_roll_pitch_rate_rad_s,
        )
    if config.maximum_angular_speed_rad_s is not None:
        velocity[5] = np.clip(
            velocity[5],
            -config.maximum_angular_speed_rad_s,
            config.maximum_angular_speed_rad_s,
        )
    if config.spatial_to_joint_velocity:
        matrix = np.asarray(config.spatial_to_joint_velocity, dtype=np.float64)
        if matrix.shape != (len(config.joint_names), 6):
            raise ValueError("spatial_to_joint_velocity must have shape (joint count, 6)")
        return matrix @ velocity
    if config.type.lower() in {"aerial", "floating", "direct"} and not config.joint_names:
        return velocity
    raise ValueError(f"Base type {config.type!r} needs a spatial_to_joint_velocity matrix")


def interpolate_gripper(config: GripperConfig, fraction_closed: float) -> np.ndarray:
    fraction = float(np.clip(fraction_closed, 0.0, 1.0))
    opened = np.asarray(config.open_positions, dtype=np.float64)
    closed = np.asarray(config.closed_positions, dtype=np.float64)
    return opened + fraction * (closed - opened)
