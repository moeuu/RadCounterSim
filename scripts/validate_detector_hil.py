#!/usr/bin/env python3
"""Validate a detector over loopback, TCP, or a physical serial link."""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from pathlib import Path

from radcounter.core.sensors.plugins import ExternalDetectorReadingBuffer
from radcounter.core.sensors.universal import DetectorReading
from radcounter.validation.detector_hil import (
    DetectorFrameCodec,
    DetectorHILSession,
    HILFrame,
    SerialLineTransport,
    SocketLineTransport,
)


def _loopback_writer(connection: socket.socket, rate_hz: float, duration_s: float) -> None:
    sequence = 0
    end_time = time.monotonic() + duration_s
    period = 1.0 / rate_hz
    next_send = time.monotonic()
    while time.monotonic() < end_time:
        if sequence == 20:
            sequence += 1
        reading = DetectorReading(
            detector_id="loopback_detector",
            model_id="laboratory_custom_czt",
            integration_time_s=period,
            expected_count_rate_cps=1250.0,
            observed_counts=25,
            spectrum_counts=(1, 4, 12, 8),
            energy_bin_edges_kev=(0.0, 200.0, 500.0, 800.0, 1200.0),
            dose_rate_usv_h=3.2,
            metadata={"evidence": "synthetic_loopback"},
        )
        frame = HILFrame(
            sequence=sequence,
            device_time_ns=time.monotonic_ns(),
            host_sent_time_ns=time.time_ns(),
            reading=reading,
        )
        payload = DetectorFrameCodec.encode(frame)
        if sequence == 40:
            decoded = json.loads(payload)
            decoded["crc32"] = "00000000"
            connection.sendall(json.dumps(decoded).encode() + b"\n")
        connection.sendall(payload)
        sequence += 1
        next_send += period
        time.sleep(max(0.0, next_send - time.monotonic()))
    connection.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("loopback", "tcp", "serial"), default="loopback")
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19090)
    parser.add_argument("--device")
    parser.add_argument("--baud-rate", type=int, default=115200)
    parser.add_argument("--hardware-serial")
    parser.add_argument("--calibration-id")
    parser.add_argument("--require-physical", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(".cache/detector-hil.json"))
    args = parser.parse_args()
    if args.require_physical and (
        args.mode != "serial"
        or not args.hardware_serial
        or not args.calibration_id
    ):
        parser.error(
            "--require-physical needs serial mode, --hardware-serial, "
            "and --calibration-id"
        )

    writer: threading.Thread | None = None
    if args.mode == "loopback":
        client, server = socket.socketpair()
        transport = SocketLineTransport(client, evidence_kind="synthetic_loopback")
        writer = threading.Thread(
            target=_loopback_writer,
            args=(server, args.rate_hz, args.duration_s),
            daemon=True,
        )
        writer.start()
    elif args.mode == "tcp":
        transport = SocketLineTransport.connect(args.host, args.port)
    else:
        if not args.device:
            parser.error("--device is required for serial mode")
        transport = SerialLineTransport(args.device, args.baud_rate)

    reading_buffer = ExternalDetectorReadingBuffer(maximum_age_s=2.0)
    try:
        metrics = DetectorHILSession(transport, reading_buffer).run(
            duration_s=args.duration_s,
            minimum_frames=max(1, int(args.rate_hz * args.duration_s * 0.5)),
        )
    finally:
        transport.close()
        if writer is not None:
            writer.join(timeout=2.0)
    metric_payload = metrics.as_dict()
    physical = bool(
        args.mode == "serial"
        and args.hardware_serial
        and args.calibration_id
    )
    expected_faults = (
        args.mode != "loopback"
        or (metrics.crc_errors == 1 and metrics.sequence_gaps == 1)
    )
    passed = (
        metrics.frames_accepted >= max(1, int(args.rate_hz * args.duration_s * 0.5))
        and metrics.malformed_frames == 0
        and expected_faults
        and (physical or not args.require_physical)
    )
    result = {
        "passed": passed,
        "qualified_as_physical_hil": physical,
        "hardware_serial": args.hardware_serial,
        "calibration_id": args.calibration_id,
        "mode": args.mode,
        "metrics": metric_payload,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
