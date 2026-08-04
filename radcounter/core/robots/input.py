"""Simulator-independent robot input commands and safe source arbitration."""

from __future__ import annotations

import itertools
import shlex
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .control import JointCommand, TwistCommand


class ControlSource(StrEnum):
    EMERGENCY = "emergency"
    KEYBOARD = "keyboard"
    GAMEPAD = "gamepad"
    GUI = "gui"
    COMMAND = "command"
    AUTO = "auto"


SOURCE_PRIORITY: Mapping[ControlSource, int] = {
    ControlSource.EMERGENCY: 1_000,
    ControlSource.KEYBOARD: 400,
    ControlSource.GAMEPAD: 400,
    ControlSource.GUI: 350,
    ControlSource.COMMAND: 300,
    ControlSource.AUTO: 100,
}


@dataclass(frozen=True)
class GripperCommand:
    name: str
    fraction_closed: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.fraction_closed <= 1.0:
            raise ValueError("fraction_closed must be in [0, 1]")


@dataclass(frozen=True)
class StopCommand:
    reason: str = "stop requested"


RobotInputCommand = TwistCommand | JointCommand | GripperCommand | StopCommand


@dataclass(frozen=True)
class CommandEnvelope:
    robot_id: str
    source: ControlSource
    command: RobotInputCommand
    issued_at_s: float
    ttl_s: float
    priority: int
    sequence: int

    def is_alive(self, now_s: float) -> bool:
        return now_s <= self.issued_at_s + self.ttl_s


class RobotCommandMux:
    """Chooses one live command per robot without mixing control sources."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._sequence = itertools.count()
        self._commands: dict[tuple[str, ControlSource], CommandEnvelope] = {}
        self._enabled: dict[tuple[str, ControlSource], bool] = {}

    def publish(
        self,
        robot_id: str,
        source: ControlSource,
        command: RobotInputCommand,
        *,
        ttl_s: float = 0.25,
        priority: int | None = None,
    ) -> CommandEnvelope:
        if ttl_s <= 0.0:
            raise ValueError("command ttl_s must be positive")
        envelope = CommandEnvelope(
            robot_id=robot_id,
            source=source,
            command=command,
            issued_at_s=self._clock(),
            ttl_s=ttl_s,
            priority=SOURCE_PRIORITY[source] if priority is None else priority,
            sequence=next(self._sequence),
        )
        self._commands[(robot_id, source)] = envelope
        return envelope

    def set_source_enabled(
        self,
        robot_id: str,
        source: ControlSource,
        enabled: bool,
    ) -> None:
        self._enabled[(robot_id, source)] = enabled
        if not enabled:
            self._commands.pop((robot_id, source), None)

    def source_enabled(self, robot_id: str, source: ControlSource) -> bool:
        return self._enabled.get((robot_id, source), True)

    def revoke(self, robot_id: str, source: ControlSource) -> None:
        self._commands.pop((robot_id, source), None)

    def resolve(self, robot_id: str) -> CommandEnvelope | None:
        now = self._clock()
        expired = [key for key, value in self._commands.items() if not value.is_alive(now)]
        for key in expired:
            self._commands.pop(key, None)
        candidates = [
            value
            for (candidate_robot, source), value in self._commands.items()
            if candidate_robot == robot_id and self.source_enabled(robot_id, source)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item.priority, item.sequence))

    def emergency_stop(self, robot_id: str, reason: str = "emergency stop") -> None:
        self.publish(
            robot_id,
            ControlSource.EMERGENCY,
            StopCommand(reason),
            ttl_s=86_400.0,
        )

    def clear_emergency_stop(self, robot_id: str) -> None:
        self.revoke(robot_id, ControlSource.EMERGENCY)


def keyboard_twist(
    pressed: Sequence[str],
    *,
    linear_speed_m_s: float,
    angular_speed_rad_s: float,
    lateral_speed_m_s: float = 0.0,
) -> TwistCommand:
    keys = {key.lower() for key in pressed}
    return TwistCommand(
        linear_x_m_s=linear_speed_m_s * (("w" in keys) - ("s" in keys)),
        linear_y_m_s=lateral_speed_m_s * (("e" in keys) - ("q" in keys)),
        angular_z_rad_s=angular_speed_rad_s * (("a" in keys) - ("d" in keys)),
    )


def gamepad_twist(
    forward_axis: float,
    lateral_axis: float,
    yaw_axis: float,
    *,
    linear_speed_m_s: float,
    angular_speed_rad_s: float,
    deadzone: float = 0.08,
) -> TwistCommand:
    def filtered(value: float) -> float:
        return 0.0 if abs(value) < deadzone else float(np.clip(value, -1.0, 1.0))

    return TwistCommand(
        linear_x_m_s=linear_speed_m_s * filtered(forward_axis),
        linear_y_m_s=linear_speed_m_s * filtered(lateral_axis),
        angular_z_rad_s=angular_speed_rad_s * filtered(yaw_axis),
    )


def is_zero_twist(command: TwistCommand, tolerance: float = 1e-9) -> bool:
    return (
        abs(command.linear_x_m_s) <= tolerance
        and abs(command.linear_y_m_s) <= tolerance
        and abs(command.angular_z_rad_s) <= tolerance
    )


def parse_command_text(text: str) -> tuple[str, RobotInputCommand]:
    """Parse: ROBOT twist VX VY WZ | joint MODE NAME=VALUE... | gripper NAME F | stop."""

    tokens = shlex.split(text)
    if len(tokens) < 2:
        raise ValueError("expected ROBOT and command kind")
    robot_id, kind = tokens[0], tokens[1].lower()
    arguments = tokens[2:]
    if kind == "twist" and len(arguments) == 3:
        return robot_id, TwistCommand(*(float(value) for value in arguments))
    if kind == "joint" and len(arguments) >= 2:
        mode = arguments[0].lower()
        names, values = [], []
        for assignment in arguments[1:]:
            name, separator, value = assignment.partition("=")
            if not separator or not name:
                raise ValueError("joint values must use NAME=VALUE")
            names.append(name)
            values.append(float(value))
        return robot_id, JointCommand(tuple(names), tuple(values), mode)
    if kind == "gripper" and len(arguments) == 2:
        return robot_id, GripperCommand(arguments[0], float(arguments[1]))
    if kind == "stop" and not arguments:
        return robot_id, StopCommand("command stop")
    raise ValueError(f"invalid {kind!r} command arguments")


def command_to_wire(
    robot_id: str,
    command: RobotInputCommand,
    *,
    ttl_s: float,
) -> dict[str, object]:
    payload: dict[str, object] = {"robot_id": robot_id, "ttl_s": ttl_s}
    if isinstance(command, TwistCommand):
        payload.update(
            kind="twist",
            linear_x_m_s=command.linear_x_m_s,
            linear_y_m_s=command.linear_y_m_s,
            angular_z_rad_s=command.angular_z_rad_s,
        )
    elif isinstance(command, JointCommand):
        payload.update(
            kind="joint",
            names=list(command.names),
            values=list(command.values),
            mode=command.mode,
        )
    elif isinstance(command, GripperCommand):
        payload.update(
            kind="gripper",
            name=command.name,
            fraction_closed=command.fraction_closed,
        )
    elif isinstance(command, StopCommand):
        payload.update(kind="stop", reason=command.reason)
    else:
        raise TypeError(type(command))
    return payload


def command_from_wire(
    payload: Mapping[str, object],
) -> tuple[str, RobotInputCommand, float]:
    robot_id = str(payload["robot_id"])
    ttl_s = float(payload.get("ttl_s", 0.5))
    kind = str(payload["kind"])
    if kind == "twist":
        command: RobotInputCommand = TwistCommand(
            float(payload.get("linear_x_m_s", 0.0)),
            float(payload.get("linear_y_m_s", 0.0)),
            float(payload.get("angular_z_rad_s", 0.0)),
        )
    elif kind == "joint":
        command = JointCommand(
            tuple(str(value) for value in payload["names"]),
            tuple(float(value) for value in payload["values"]),
            str(payload["mode"]),
        )
    elif kind == "gripper":
        command = GripperCommand(
            str(payload["name"]),
            float(payload["fraction_closed"]),
        )
    elif kind == "stop":
        command = StopCommand(str(payload.get("reason", "command stop")))
    else:
        raise ValueError(f"unknown wire command kind: {kind}")
    return robot_id, command, ttl_s
