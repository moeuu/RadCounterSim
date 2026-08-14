"""Kit keyboard/gamepad events, command socket, and autonomous command routing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from radcounter.core.robots.command_io import RobotCommandServer
from radcounter.core.robots.control import JointCommand, TwistCommand
from radcounter.core.robots.input import (
    ControlSource,
    GripperCommand,
    RobotCommandMux,
    RobotInputCommand,
    StopCommand,
    command_from_wire,
    gamepad_twist,
    is_zero_twist,
    keyboard_twist,
)


class InputRobotController(Protocol):
    def command_joint(self, command: JointCommand) -> None: ...
    def command_twist(self, command: TwistCommand) -> None: ...
    def command_gripper(self, name: str, fraction_closed: float) -> None: ...


AutoProvider = Callable[[float], RobotInputCommand | None]


class IsaacRobotInputRouter:
    def __init__(
        self,
        controllers: Mapping[str, InputRobotController],
        *,
        mux: RobotCommandMux | None = None,
        command_server: RobotCommandServer | None = None,
        linear_speed_m_s: float = 0.8,
        angular_speed_rad_s: float = 1.2,
        gamepad_deadzone: float = 0.08,
    ) -> None:
        self.controllers = dict(controllers)
        self.mux = mux or RobotCommandMux()
        self.command_server = command_server
        self.linear_speed_m_s = linear_speed_m_s
        self.angular_speed_rad_s = angular_speed_rad_s
        self.gamepad_deadzone = gamepad_deadzone
        self._keyboard_robot = next(iter(self.controllers), None)
        self._gamepad_robot = self._keyboard_robot
        self._pressed: set[str] = set()
        self._gamepad_axes = {"forward": 0.0, "lateral": 0.0, "yaw": 0.0}
        self._keyboard_was_active = False
        self._gamepad_was_active = False
        self._last_source: dict[str, ControlSource | None] = {
            robot_id: None for robot_id in self.controllers
        }
        self._auto: dict[str, AutoProvider] = {}
        self._input = None
        self._keyboard = None
        self._gamepad = None
        self._keyboard_subscription = None
        self._gamepad_subscription = None

    def attach_devices(
        self,
        keyboard_robot: str | None = None,
        gamepad_robot: str | None = None,
        gamepad_index: int = 0,
    ) -> None:
        import carb
        import omni.appwindow

        self._keyboard_robot = keyboard_robot or self._keyboard_robot
        self._gamepad_robot = gamepad_robot or self._gamepad_robot
        app_window = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = app_window.get_keyboard()
        self._keyboard_subscription = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard,
        )
        try:
            self._gamepad = app_window.get_gamepad(gamepad_index)
            self._gamepad_subscription = self._input.subscribe_to_gamepad_events(
                self._gamepad,
                self._on_gamepad,
            )
        except Exception:
            self._gamepad = None
            self._gamepad_subscription = None

    def detach_devices(self) -> None:
        if self._input is not None and self._keyboard_subscription is not None:
            self._input.unsubscribe_to_keyboard_events(
                self._keyboard,
                self._keyboard_subscription,
            )
        if self._input is not None and self._gamepad_subscription is not None:
            self._input.unsubscribe_to_gamepad_events(
                self._gamepad,
                self._gamepad_subscription,
            )
        self._keyboard_subscription = None
        self._gamepad_subscription = None

    def close(self) -> None:
        """Release Kit input subscriptions and an optional command server."""

        self.detach_devices()
        if self.command_server is not None:
            self.command_server.close()
            self.command_server = None

    def register_auto_controller(
        self,
        robot_id: str,
        provider: AutoProvider,
        *,
        enabled: bool = True,
    ) -> None:
        if robot_id not in self.controllers:
            raise KeyError(robot_id)
        self._auto[robot_id] = provider
        self.mux.set_source_enabled(robot_id, ControlSource.AUTO, enabled)

    def set_auto_enabled(self, robot_id: str, enabled: bool) -> None:
        self.mux.set_source_enabled(robot_id, ControlSource.AUTO, enabled)

    def publish(
        self,
        robot_id: str,
        command: RobotInputCommand,
        *,
        source: ControlSource = ControlSource.GUI,
        ttl_s: float = 0.25,
    ) -> None:
        self.mux.publish(robot_id, source, command, ttl_s=ttl_s)

    def active_source(self, robot_id: str) -> ControlSource | None:
        envelope = self.mux.resolve(robot_id)
        return envelope.source if envelope is not None else None

    def update(self, dt_s: float) -> None:
        self._drain_commands()
        self._publish_keyboard()
        self._publish_gamepad()
        for robot_id, provider in self._auto.items():
            if self.mux.source_enabled(robot_id, ControlSource.AUTO):
                command = provider(dt_s)
                if command is not None:
                    self.mux.publish(
                        robot_id,
                        ControlSource.AUTO,
                        command,
                        ttl_s=max(0.1, dt_s * 3.0),
                    )
        for robot_id, controller in self.controllers.items():
            envelope = self.mux.resolve(robot_id)
            if envelope is None:
                if self._last_source[robot_id] is not None:
                    controller.command_twist(TwistCommand())
                self._last_source[robot_id] = None
                continue
            self._apply(controller, envelope.command)
            self._last_source[robot_id] = envelope.source

    def _drain_commands(self) -> None:
        if self.command_server is None:
            return
        for payload in self.command_server.drain():
            robot_id, command, ttl_s = command_from_wire(payload)
            if robot_id not in self.controllers:
                continue
            self.mux.publish(
                robot_id,
                ControlSource.COMMAND,
                command,
                ttl_s=ttl_s,
            )

    def _publish_keyboard(self) -> None:
        if self._keyboard_robot is None:
            return
        command = keyboard_twist(
            self._pressed,
            linear_speed_m_s=self.linear_speed_m_s,
            angular_speed_rad_s=self.angular_speed_rad_s,
            lateral_speed_m_s=self.linear_speed_m_s,
        )
        active = not is_zero_twist(command)
        if active or self._keyboard_was_active:
            self.mux.publish(
                self._keyboard_robot,
                ControlSource.KEYBOARD,
                command,
                ttl_s=0.15,
            )
        self._keyboard_was_active = active

    def _publish_gamepad(self) -> None:
        if self._gamepad_robot is None:
            return
        command = gamepad_twist(
            self._gamepad_axes["forward"],
            self._gamepad_axes["lateral"],
            self._gamepad_axes["yaw"],
            linear_speed_m_s=self.linear_speed_m_s,
            angular_speed_rad_s=self.angular_speed_rad_s,
            deadzone=self.gamepad_deadzone,
        )
        active = not is_zero_twist(command)
        if active or self._gamepad_was_active:
            self.mux.publish(
                self._gamepad_robot,
                ControlSource.GAMEPAD,
                command,
                ttl_s=0.15,
            )
        self._gamepad_was_active = active

    def _on_keyboard(self, event) -> bool:
        import carb

        names = {
            carb.input.KeyboardInput.W: "w",
            carb.input.KeyboardInput.A: "a",
            carb.input.KeyboardInput.S: "s",
            carb.input.KeyboardInput.D: "d",
            carb.input.KeyboardInput.Q: "q",
            carb.input.KeyboardInput.E: "e",
        }
        if (
            event.input == carb.input.KeyboardInput.SPACE
            and event.type == carb.input.KeyboardEventType.KEY_PRESS
        ):
            if self._keyboard_robot is not None:
                self.mux.emergency_stop(self._keyboard_robot, "keyboard space")
            return True
        if (
            event.input == carb.input.KeyboardInput.ENTER
            and event.type == carb.input.KeyboardEventType.KEY_PRESS
        ):
            if self._keyboard_robot is not None:
                self.mux.clear_emergency_stop(self._keyboard_robot)
            return True
        key = names.get(event.input)
        if key is None:
            return False
        if event.type in {
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_REPEAT,
        }:
            self._pressed.add(key)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed.discard(key)
        return True

    def _on_gamepad(self, event) -> bool:
        import carb

        axes = {
            carb.input.GamepadInput.LEFT_STICK_UP: ("forward", 1.0),
            carb.input.GamepadInput.LEFT_STICK_DOWN: ("forward", -1.0),
            carb.input.GamepadInput.LEFT_STICK_RIGHT: ("lateral", 1.0),
            carb.input.GamepadInput.LEFT_STICK_LEFT: ("lateral", -1.0),
            carb.input.GamepadInput.RIGHT_STICK_RIGHT: ("yaw", -1.0),
            carb.input.GamepadInput.RIGHT_STICK_LEFT: ("yaw", 1.0),
        }
        mapping = axes.get(event.input)
        if mapping is None:
            return False
        axis, sign = mapping
        value = sign * float(event.value)
        current = self._gamepad_axes[axis]
        if abs(value) >= abs(current) or event.value == 0.0:
            self._gamepad_axes[axis] = value
        return True

    @staticmethod
    def _apply(controller: InputRobotController, command: RobotInputCommand) -> None:
        if isinstance(command, TwistCommand):
            controller.command_twist(command)
        elif isinstance(command, JointCommand):
            controller.command_joint(command)
        elif isinstance(command, GripperCommand):
            controller.command_gripper(command.name, command.fraction_closed)
        elif isinstance(command, StopCommand):
            controller.command_twist(TwistCommand())
        else:
            raise TypeError(type(command))
