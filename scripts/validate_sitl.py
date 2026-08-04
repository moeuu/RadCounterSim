#!/usr/bin/env python3
"""Launch a pinned flight stack and validate live MAVLink flight telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlopen

from pymavlink import mavutil

from radcounter.validation.aerial import run_rotor_dynamics_gate

PX4_IMAGE = (
    "px4io/px4-sitl@"
    "sha256:bab4270c4849b7027df4bd760c79d743d738c81d7830dde14c4cc5714f781216"
)
ARDUPILOT_URL = (
    "https://firmware.ardupilot.org/Copter/stable/"
    "SITL_x86_64_linux_gnu/arducopter"
)
ARDUPILOT_SHA256 = "6cd15ad7a14de5256e14e9fef9c521ac22a5c003c09a04c08181bfb078b7c416"
ARDUPILOT_DEFAULTS_URL = (
    "https://raw.githubusercontent.com/ArduPilot/ardupilot/"
    "92b0cd788ec29406f26c6f9c31d5ceedbd1cc538/"
    "Tools/autotest/default_params/copter.parm"
)
ARDUPILOT_DEFAULTS_SHA256 = (
    "5e01345b45d1c6190b28bece5638bbdd4cf1cce35e05bbbf480ab24d2b51aa0e"
)


def _download_ardupilot(path: Path) -> None:
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == ARDUPILOT_SHA256:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(ARDUPILOT_URL, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != ARDUPILOT_SHA256:
        raise RuntimeError(f"ArduPilot binary digest changed: {digest}")
    path.write_bytes(payload)
    path.chmod(0o755)


def _download_ardupilot_defaults(path: Path) -> None:
    if (
        path.exists()
        and hashlib.sha256(path.read_bytes()).hexdigest()
        == ARDUPILOT_DEFAULTS_SHA256
    ):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(ARDUPILOT_DEFAULTS_URL, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != ARDUPILOT_DEFAULTS_SHA256:
        raise RuntimeError(f"ArduPilot defaults digest changed: {digest}")
    path.write_bytes(payload)


@contextmanager
def _launch_stack(stack: str, output_directory: Path) -> Iterator[tuple[str, Path]]:
    output_directory.mkdir(parents=True, exist_ok=True)
    log_path = output_directory / f"{stack}.log"
    log_stream = log_path.open("w", encoding="utf-8")
    process: subprocess.Popen[str] | None = None
    container_name: str | None = None
    try:
        if stack == "px4":
            container_name = f"radcounter-px4-{uuid.uuid4().hex[:10]}"
            command = [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "--network",
                "host",
                "--log-driver",
                "local",
                "--log-opt",
                "max-size=1m",
                "--log-opt",
                "max-file=1",
                "--log-opt",
                "compress=false",
                "-e",
                "PX4_SIM_MODEL=sihsim_quadx",
                PX4_IMAGE,
            ]
            endpoint = "udpin:0.0.0.0:14550"
        else:
            binary = output_directory.parent / "downloads" / "arducopter-4.6.3"
            defaults = output_directory.parent / "downloads" / "copter-4.6.3.parm"
            _download_ardupilot(binary)
            _download_ardupilot_defaults(defaults)
            runtime = output_directory / "ardupilot-runtime"
            runtime.mkdir(parents=True, exist_ok=True)
            command = [
                str(binary),
                "--wipe",
                "--model",
                "quad",
                "--speedup",
                "1",
                "--home",
                "35.681236,139.767125,20,0",
                "--serial0",
                "udpclient:127.0.0.1:14555",
                "--defaults",
                str(defaults),
            ]
            endpoint = "udpin:0.0.0.0:14555"
        if stack == "px4":
            subprocess.run(command, check=True, capture_output=True, text=True)
        else:
            process = subprocess.Popen(
                command,
                cwd=runtime,
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
        yield endpoint, log_path
    finally:
        if container_name is not None:
            subprocess.run(
                ["docker", "logs", container_name],
                check=False,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                check=False,
                capture_output=True,
                text=True,
            )
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        log_stream.close()


def _request_message(connection: mavutil.mavfile, message_id: int, interval_us: int) -> None:
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        message_id,
        interval_us,
        0,
        0,
        0,
        0,
        0,
    )


def _set_parameter(
    connection: mavutil.mavfile,
    name: str,
    value: float,
) -> None:
    connection.mav.param_set_send(
        connection.target_system,
        connection.target_component,
        name.encode("ascii"),
        value,
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
    )


def _command(
    connection: mavutil.mavfile,
    command: int,
    parameters: tuple[float, float, float, float, float, float, float],
) -> None:
    connection.mav.command_long_send(
        connection.target_system,
        connection.target_component,
        command,
        0,
        *parameters,
    )


def _set_px4_auto_takeoff(connection: mavutil.mavfile) -> None:
    custom_mode = 6 << 16
    connection.mav.set_mode_send(
        connection.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        custom_mode,
    )


def _send_gcs_heartbeat(connection: mavutil.mavfile) -> None:
    connection.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
        0,
        0,
        mavutil.mavlink.MAV_STATE_ACTIVE,
    )


def _send_local_position_target(
    connection: mavutil.mavfile,
    target: tuple[float, float, float],
    elapsed_s: float,
) -> None:
    connection.mav.set_position_target_local_ned_send(
        int(elapsed_s * 1000.0) & 0xFFFFFFFF,
        connection.target_system,
        connection.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        3576,
        target[0],
        target[1],
        target[2],
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


def _collect_flight(
    endpoint: str,
    stack: str,
    *,
    duration_s: float,
    observe_only: bool,
) -> dict[str, object]:
    connection = mavutil.mavlink_connection(endpoint, source_system=250)
    heartbeat = connection.wait_heartbeat(timeout=35)
    if heartbeat is None:
        raise RuntimeError(f"{stack} produced no MAVLink heartbeat")
    if stack == "ardupilot":
        _set_parameter(connection, "FRAME_CLASS", 1.0)
        _set_parameter(connection, "FRAME_TYPE", 1.0)
    for message_id, interval in (
        (30, 50_000),
        (32, 50_000),
        (33, 50_000),
        (105, 50_000),
        (36, 100_000),
        (193, 100_000),
    ):
        _request_message(connection, message_id, interval)
    counts: Counter[str] = Counter()
    altitudes = []
    roll_angles = []
    acknowledgements = []
    status_text = []
    armed_seen = False
    currently_armed = False
    ekf_status_flags = 0
    start_time = time.monotonic()
    command_time = start_time + (25.0 if stack == "ardupilot" else 5.0)
    command_stage = 0
    latest_local: tuple[float, float, float] | None = None
    offboard_target: tuple[float, float, float] | None = None
    next_arm_attempt = command_time
    last_gcs_heartbeat = 0.0
    last_position_target = 0.0
    end_time = start_time + duration_s
    while time.monotonic() < end_time:
        now = time.monotonic()
        elapsed = now - start_time
        if now - last_gcs_heartbeat >= 1.0:
            _send_gcs_heartbeat(connection)
            last_gcs_heartbeat = now
        if (
            stack == "px4"
            and latest_local is not None
            and now >= command_time - 2.0
            and now - last_position_target >= 0.05
        ):
            _send_local_position_target(
                connection,
                offboard_target or latest_local,
                elapsed,
            )
            last_position_target = now
        if not observe_only and stack == "ardupilot":
            if not currently_armed and now >= next_arm_attempt:
                modes = connection.mode_mapping() or {}
                guided = modes.get("GUIDED")
                if guided is None:
                    raise RuntimeError("ArduPilot did not publish a GUIDED mode mapping")
                connection.mav.set_mode_send(
                    connection.target_system,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    guided,
                )
                _command(
                    connection,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    (1.0, 21196.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                )
                next_arm_attempt = now + 2.0
            elif currently_armed and command_stage < 3:
                _command(
                    connection,
                    mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                    (0.0, 0.0, 0.0, math.nan, math.nan, math.nan, 3.0),
                )
                command_stage = 3
        elif (
            not observe_only
            and stack == "px4"
            and command_stage == 0
            and now >= command_time
        ):
            _set_px4_auto_takeoff(connection)
            command_stage = 1
        elif (
            not observe_only
            and stack == "px4"
            and command_stage == 1
            and now >= command_time + 0.75
        ):
            _command(
                connection,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                (1.0, 21196.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )
            if stack == "px4" and latest_local is not None:
                offboard_target = (
                    latest_local[0],
                    latest_local[1],
                    latest_local[2] - 3.0,
                )
            command_stage = 2
        elif (
            not observe_only
            and stack == "px4"
            and command_stage == 2
            and now >= command_time + 1.5
        ):
            command_stage = 3
        message = connection.recv_match(blocking=True, timeout=0.1)
        if message is None:
            continue
        message_type = message.get_type()
        if message_type == "BAD_DATA":
            continue
        counts[message_type] += 1
        if message_type == "HEARTBEAT":
            currently_armed = bool(
                message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            )
            armed_seen = armed_seen or currently_armed
        elif message_type == "LOCAL_POSITION_NED":
            latest_local = (float(message.x), float(message.y), float(message.z))
            altitudes.append(-float(message.z))
        elif message_type == "GLOBAL_POSITION_INT":
            altitudes.append(float(message.relative_alt) / 1000.0)
        elif message_type == "ATTITUDE":
            roll_angles.append(float(message.roll))
        elif message_type == "COMMAND_ACK":
            acknowledgements.append(
                {
                    "command": int(message.command),
                    "result": int(message.result),
                    "progress": int(getattr(message, "progress", 0)),
                }
            )
        elif message_type == "STATUSTEXT":
            status_text.append(str(message.text))
        elif message_type == "EKF_STATUS_REPORT":
            ekf_status_flags = int(message.flags)
    connection.close()
    altitude_span = max(altitudes) - min(altitudes) if len(altitudes) >= 2 else 0.0
    telemetry_pass = (
        counts["HEARTBEAT"] >= 1
        and counts["ATTITUDE"] >= 10
        and (
            counts["LOCAL_POSITION_NED"] >= 5
            or counts["GLOBAL_POSITION_INT"] >= 5
        )
    )
    flight_pass = observe_only or (armed_seen and altitude_span >= 0.5)
    return {
        "passed": telemetry_pass and flight_pass,
        "stack": stack,
        "endpoint": endpoint,
        "duration_s": duration_s,
        "observe_only": observe_only,
        "heartbeat_autopilot": int(heartbeat.autopilot),
        "message_counts": dict(counts),
        "command_acknowledgements": acknowledgements,
        "status_text": status_text[-40:],
        "ekf_status_flags": ekf_status_flags,
        "armed_seen": armed_seen,
        "altitude_span_m": altitude_span,
        "maximum_absolute_roll_rad": max(map(abs, roll_angles), default=0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", choices=("px4", "ardupilot"), required=True)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--observe-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/external-validation/sitl"),
    )
    args = parser.parse_args()
    stack_output = args.output_dir.resolve() / args.stack
    rotor_result = run_rotor_dynamics_gate()
    with _launch_stack(args.stack, stack_output) as (endpoint, log_path):
        flight_result = _collect_flight(
            endpoint,
            args.stack,
            duration_s=args.duration_s,
            observe_only=args.observe_only,
        )
    result = {
        "passed": rotor_result.passed and bool(flight_result["passed"]),
        "rotor_dynamics": rotor_result.as_dict(),
        "sitl": flight_result,
        "stack_log": str(log_path),
        "evidence_kind": "software_in_the_loop",
        "qualified_as_physical_flight": False,
    }
    result_path = stack_output / "metrics.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
