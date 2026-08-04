import socket

from radcounter.core.robots.command_io import RobotCommandServer, send_command
from radcounter.core.robots.control import TwistCommand
from radcounter.core.robots.input import (
    ControlSource,
    RobotCommandMux,
    StopCommand,
    command_from_wire,
    command_to_wire,
    gamepad_twist,
    keyboard_twist,
    parse_command_text,
)


class _Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


def test_manual_source_preempts_auto_then_expires() -> None:
    clock = _Clock()
    mux = RobotCommandMux(clock)
    mux.publish("robot", ControlSource.AUTO, TwistCommand(0.2), ttl_s=10.0)
    mux.publish("robot", ControlSource.KEYBOARD, TwistCommand(1.0), ttl_s=0.2)
    assert mux.resolve("robot").source is ControlSource.KEYBOARD
    clock.now = 0.21
    assert mux.resolve("robot").source is ControlSource.AUTO


def test_emergency_stop_preempts_every_input() -> None:
    mux = RobotCommandMux()
    mux.publish("robot", ControlSource.GAMEPAD, TwistCommand(1.0), ttl_s=10.0)
    mux.emergency_stop("robot")
    assert isinstance(mux.resolve("robot").command, StopCommand)
    mux.clear_emergency_stop("robot")
    assert mux.resolve("robot").source is ControlSource.GAMEPAD


def test_keyboard_and_gamepad_mapping() -> None:
    keyboard = keyboard_twist(
        ("w", "a"),
        linear_speed_m_s=2.0,
        angular_speed_rad_s=1.0,
    )
    assert keyboard == TwistCommand(2.0, 0.0, 1.0)
    gamepad = gamepad_twist(
        0.5,
        0.01,
        -0.25,
        linear_speed_m_s=2.0,
        angular_speed_rad_s=4.0,
    )
    assert gamepad == TwistCommand(1.0, 0.0, -1.0)


def test_text_and_wire_commands_round_trip() -> None:
    robot_id, command = parse_command_text("r1 joint position j1=0.5 j2=-1")
    payload = command_to_wire(robot_id, command, ttl_s=2.0)
    decoded_robot, decoded_command, ttl = command_from_wire(payload)
    assert decoded_robot == robot_id
    assert decoded_command == command
    assert ttl == 2.0


def test_local_command_server_accepts_json_line() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = RobotCommandServer(port=port)
    server.start()
    try:
        payload = command_to_wire("robot", TwistCommand(0.5), ttl_s=1.0)
        assert send_command(payload, port=port)["ok"] is True
        assert server.drain() == (payload,)
    finally:
        server.close()
