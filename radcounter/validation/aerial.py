"""Small deterministic six-degree-of-freedom multirotor validation model."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RotorSpec:
    position_body_m: tuple[float, float, float]
    spin_direction: int
    thrust_coefficient: float
    torque_coefficient: float
    maximum_speed_rad_s: float
    time_constant_s: float

    def __post_init__(self) -> None:
        if self.spin_direction not in (-1, 1):
            raise ValueError("rotor spin_direction must be -1 or 1")
        if min(
            self.thrust_coefficient,
            self.torque_coefficient,
            self.maximum_speed_rad_s,
            self.time_constant_s,
        ) <= 0.0:
            raise ValueError("rotor coefficients, speed, and time constant must be positive")


@dataclass(frozen=True)
class MultirotorSpec:
    mass_kg: float
    inertia_kg_m2: tuple[float, float, float]
    linear_drag_n_per_m_s: tuple[float, float, float]
    angular_drag_nm_per_rad_s: tuple[float, float, float]
    rotors: tuple[RotorSpec, ...]
    gravity_m_s2: float = 9.80665

    def __post_init__(self) -> None:
        if self.mass_kg <= 0.0 or len(self.rotors) < 3:
            raise ValueError("a multirotor requires positive mass and at least three rotors")
        if min(self.inertia_kg_m2) <= 0.0:
            raise ValueError("principal moments of inertia must be positive")

    @classmethod
    def quad_x(cls) -> MultirotorSpec:
        arm = 0.23 / math.sqrt(2.0)
        positions = (
            (arm, arm, 0.0),
            (-arm, arm, 0.0),
            (-arm, -arm, 0.0),
            (arm, -arm, 0.0),
        )
        spins = (1, -1, 1, -1)
        rotors = tuple(
            RotorSpec(
                position_body_m=position,
                spin_direction=spin,
                thrust_coefficient=5.0e-6,
                torque_coefficient=7.0e-8,
                maximum_speed_rad_s=950.0,
                time_constant_s=0.04,
            )
            for position, spin in zip(positions, spins, strict=True)
        )
        return cls(
            mass_kg=1.5,
            inertia_kg_m2=(0.029, 0.029, 0.055),
            linear_drag_n_per_m_s=(0.08, 0.08, 0.12),
            angular_drag_nm_per_rad_s=(0.012, 0.012, 0.018),
            rotors=rotors,
        )


@dataclass
class MultirotorState:
    position_world_m: np.ndarray
    velocity_world_m_s: np.ndarray
    quaternion_body_to_world: np.ndarray
    angular_velocity_body_rad_s: np.ndarray
    rotor_speed_rad_s: np.ndarray

    @classmethod
    def at_rest(cls, rotor_count: int) -> MultirotorState:
        return cls(
            position_world_m=np.zeros(3, dtype=np.float64),
            velocity_world_m_s=np.zeros(3, dtype=np.float64),
            quaternion_body_to_world=np.asarray((1.0, 0.0, 0.0, 0.0)),
            angular_velocity_body_rad_s=np.zeros(3, dtype=np.float64),
            rotor_speed_rad_s=np.zeros(rotor_count, dtype=np.float64),
        )


def _quaternion_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


class MultirotorDynamics:
    """Motor lag, rotor thrust/drag torque, rigid-body motion, and air drag."""

    def __init__(self, spec: MultirotorSpec) -> None:
        self.spec = spec
        self._inertia = np.asarray(spec.inertia_kg_m2, dtype=np.float64)

    def hover_commands(self) -> np.ndarray:
        per_rotor_thrust = self.spec.mass_kg * self.spec.gravity_m_s2 / len(self.spec.rotors)
        return np.asarray(
            [
                per_rotor_thrust
                / (rotor.thrust_coefficient * rotor.maximum_speed_rad_s**2)
                for rotor in self.spec.rotors
            ],
            dtype=np.float64,
        )

    def step(
        self,
        state: MultirotorState,
        normalized_commands: Sequence[float],
        dt_s: float,
    ) -> MultirotorState:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        commands = np.clip(np.asarray(normalized_commands, dtype=np.float64), 0.0, 1.0)
        if commands.shape != (len(self.spec.rotors),):
            raise ValueError("one normalized command is required per rotor")

        targets = np.asarray(
            [
                math.sqrt(command) * rotor.maximum_speed_rad_s
                for command, rotor in zip(commands, self.spec.rotors, strict=True)
            ]
        )
        time_constants = np.asarray([rotor.time_constant_s for rotor in self.spec.rotors])
        motor_fraction = 1.0 - np.exp(-dt_s / time_constants)
        rotor_speeds = state.rotor_speed_rad_s + motor_fraction * (
            targets - state.rotor_speed_rad_s
        )

        force_body = np.zeros(3, dtype=np.float64)
        torque_body = np.zeros(3, dtype=np.float64)
        for rotor, speed in zip(self.spec.rotors, rotor_speeds, strict=True):
            thrust = rotor.thrust_coefficient * speed * speed
            rotor_force = np.asarray((0.0, 0.0, thrust))
            force_body += rotor_force
            torque_body += np.cross(np.asarray(rotor.position_body_m), rotor_force)
            torque_body[2] += rotor.spin_direction * rotor.torque_coefficient * speed * speed

        rotation = _rotation_matrix(state.quaternion_body_to_world)
        drag = np.asarray(self.spec.linear_drag_n_per_m_s) * state.velocity_world_m_s
        force_world = rotation @ force_body - drag
        force_world[2] -= self.spec.mass_kg * self.spec.gravity_m_s2
        acceleration = force_world / self.spec.mass_kg
        velocity = state.velocity_world_m_s + acceleration * dt_s
        position = state.position_world_m + velocity * dt_s

        angular_drag = (
            np.asarray(self.spec.angular_drag_nm_per_rad_s)
            * state.angular_velocity_body_rad_s
        )
        gyroscopic = np.cross(
            state.angular_velocity_body_rad_s,
            self._inertia * state.angular_velocity_body_rad_s,
        )
        angular_acceleration = (torque_body - angular_drag - gyroscopic) / self._inertia
        angular_velocity = state.angular_velocity_body_rad_s + angular_acceleration * dt_s
        quaternion_rate = 0.5 * _quaternion_product(
            state.quaternion_body_to_world,
            np.asarray((0.0, *angular_velocity)),
        )
        quaternion = state.quaternion_body_to_world + quaternion_rate * dt_s
        quaternion /= np.linalg.norm(quaternion)
        return MultirotorState(
            position_world_m=position,
            velocity_world_m_s=velocity,
            quaternion_body_to_world=quaternion,
            angular_velocity_body_rad_s=angular_velocity,
            rotor_speed_rad_s=rotor_speeds,
        )


@dataclass(frozen=True)
class RotorDynamicsGateResult:
    passed: bool
    duration_s: float
    integration_step_s: float
    hover_error_m: float
    maximum_roll_deg: float
    maximum_motor_speed_rad_s: float
    finite_state: bool

    def as_dict(self) -> dict[str, bool | float]:
        return asdict(self)


def run_rotor_dynamics_gate(
    *,
    duration_s: float = 8.0,
    integration_step_s: float = 0.002,
) -> RotorDynamicsGateResult:
    """Exercise hover and a differential-thrust roll pulse without a flight controller."""

    spec = MultirotorSpec.quad_x()
    dynamics = MultirotorDynamics(spec)
    state = MultirotorState.at_rest(len(spec.rotors))
    hover = dynamics.hover_commands()
    state.rotor_speed_rad_s = np.asarray(
        [
            math.sqrt(command) * rotor.maximum_speed_rad_s
            for command, rotor in zip(hover, spec.rotors, strict=True)
        ]
    )
    maximum_roll = 0.0
    hover_error = 0.0
    finite = True
    steps = int(duration_s / integration_step_s)
    for index in range(steps):
        time_s = index * integration_step_s
        command = hover.copy()
        if 3.0 <= time_s < 3.35:
            command[:2] += 0.07
            command[2:] -= 0.07
        elif 3.35 <= time_s < 3.70:
            command[:2] -= 0.07
            command[2:] += 0.07
        state = dynamics.step(state, command, integration_step_s)
        quaternion = state.quaternion_body_to_world
        roll = math.atan2(
            2.0 * (quaternion[0] * quaternion[1] + quaternion[2] * quaternion[3]),
            1.0 - 2.0 * (quaternion[1] ** 2 + quaternion[2] ** 2),
        )
        maximum_roll = max(maximum_roll, abs(math.degrees(roll)))
        if 1.0 <= time_s < 3.0:
            hover_error = max(hover_error, abs(float(state.position_world_m[2])))
        finite = finite and all(
            bool(np.all(np.isfinite(value)))
            for value in (
                state.position_world_m,
                state.velocity_world_m_s,
                state.quaternion_body_to_world,
                state.angular_velocity_body_rad_s,
                state.rotor_speed_rad_s,
            )
        )
    maximum_speed = float(np.max(state.rotor_speed_rad_s))
    passed = (
        finite
        and hover_error < 0.20
        and maximum_roll > 4.0
        and maximum_speed < max(rotor.maximum_speed_rad_s for rotor in spec.rotors)
    )
    return RotorDynamicsGateResult(
        passed=passed,
        duration_s=duration_s,
        integration_step_s=integration_step_s,
        hover_error_m=hover_error,
        maximum_roll_deg=maximum_roll,
        maximum_motor_speed_rad_s=maximum_speed,
        finite_state=finite,
    )
