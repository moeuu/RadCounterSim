"""Versioned line protocol and transports for detector hardware-in-the-loop."""

from __future__ import annotations

import json
import socket
import time
import zlib
from dataclasses import asdict, dataclass
from typing import Protocol

from radcounter.core.sensors.plugins import ExternalDetectorReadingBuffer
from radcounter.core.sensors.universal import DetectorReading


class DetectorFrameError(ValueError):
    """Base class for malformed detector frames."""


class DetectorFrameCrcError(DetectorFrameError):
    """Frame payload and CRC do not match."""


@dataclass(frozen=True)
class HILFrame:
    sequence: int
    device_time_ns: int
    reading: DetectorReading
    protocol_version: int = 1
    host_sent_time_ns: int | None = None

    def __post_init__(self) -> None:
        if self.protocol_version != 1:
            raise ValueError("unsupported detector HIL protocol version")
        if self.sequence < 0 or self.device_time_ns < 0:
            raise ValueError("sequence and device time must be non-negative")


class DetectorFrameCodec:
    """Canonical JSON-lines codec with CRC32 over all fields except crc32."""

    @staticmethod
    def _reading_mapping(reading: DetectorReading) -> dict[str, object]:
        value = asdict(reading)
        if reading.metadata is not None:
            value["metadata"] = dict(reading.metadata)
        return value

    @classmethod
    def encode(cls, frame: HILFrame) -> bytes:
        payload: dict[str, object] = {
            "protocol_version": frame.protocol_version,
            "sequence": frame.sequence,
            "device_time_ns": frame.device_time_ns,
            "host_sent_time_ns": frame.host_sent_time_ns,
            "reading": cls._reading_mapping(frame.reading),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        payload["crc32"] = f"{zlib.crc32(canonical) & 0xFFFFFFFF:08x}"
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"

    @staticmethod
    def decode(value: bytes | str) -> HILFrame:
        try:
            text = value.decode("utf-8") if isinstance(value, bytes) else value
            payload = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DetectorFrameError("detector frame is not valid UTF-8 JSON") from error
        if not isinstance(payload, dict):
            raise DetectorFrameError("detector frame must be a JSON object")
        supplied_crc = payload.pop("crc32", None)
        if not isinstance(supplied_crc, str):
            raise DetectorFrameError("detector frame has no crc32")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        expected_crc = f"{zlib.crc32(canonical) & 0xFFFFFFFF:08x}"
        if supplied_crc.lower() != expected_crc:
            raise DetectorFrameCrcError(
                f"detector frame CRC mismatch: {supplied_crc} != {expected_crc}"
            )
        reading_payload = payload.get("reading")
        if not isinstance(reading_payload, dict):
            raise DetectorFrameError("detector frame reading must be an object")
        try:
            reading = DetectorReading(
                detector_id=str(reading_payload["detector_id"]),
                model_id=str(reading_payload["model_id"]),
                integration_time_s=float(reading_payload["integration_time_s"]),
                expected_count_rate_cps=float(
                    reading_payload.get("expected_count_rate_cps", 0.0)
                ),
                observed_counts=int(reading_payload["observed_counts"]),
                spectrum_counts=tuple(
                    int(item) for item in reading_payload.get("spectrum_counts", ())
                ),
                energy_bin_edges_kev=tuple(
                    float(item)
                    for item in reading_payload.get("energy_bin_edges_kev", ())
                ),
                dose_rate_usv_h=float(reading_payload.get("dose_rate_usv_h", 0.0)),
                estimated_direction_world=(
                    tuple(
                        float(item)
                        for item in reading_payload["estimated_direction_world"]
                    )
                    if reading_payload.get("estimated_direction_world") is not None
                    else None
                ),
                saturated=bool(reading_payload.get("saturated", False)),
                metadata=reading_payload.get("metadata"),
            )
            return HILFrame(
                protocol_version=int(payload["protocol_version"]),
                sequence=int(payload["sequence"]),
                device_time_ns=int(payload["device_time_ns"]),
                host_sent_time_ns=(
                    int(payload["host_sent_time_ns"])
                    if payload.get("host_sent_time_ns") is not None
                    else None
                ),
                reading=reading,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise DetectorFrameError("detector frame contains invalid fields") from error


class LineTransport(Protocol):
    evidence_kind: str

    def readline(self, timeout_s: float) -> bytes: ...

    def close(self) -> None: ...


class SocketLineTransport:
    evidence_kind = "tcp_transport"

    def __init__(
        self,
        connection: socket.socket,
        *,
        evidence_kind: str = "tcp_transport",
    ) -> None:
        self._connection = connection
        self._buffer = bytearray()
        self.evidence_kind = evidence_kind

    @classmethod
    def connect(
        cls,
        host: str,
        port: int,
        *,
        connect_timeout_s: float = 5.0,
    ) -> SocketLineTransport:
        return cls(socket.create_connection((host, port), timeout=connect_timeout_s))

    def readline(self, timeout_s: float) -> bytes:
        self._connection.settimeout(timeout_s)
        while b"\n" not in self._buffer:
            try:
                chunk = self._connection.recv(65536)
            except TimeoutError:
                return b""
            if not chunk:
                return b""
            self._buffer.extend(chunk)
        line, _, remaining = self._buffer.partition(b"\n")
        self._buffer = bytearray(remaining)
        return bytes(line) + b"\n"

    def close(self) -> None:
        self._connection.close()


class SerialLineTransport:
    evidence_kind = "serial_transport"

    def __init__(self, device: str, baud_rate: int = 115200) -> None:
        try:
            import serial
        except ImportError as error:
            raise RuntimeError("pyserial is required for serial detector HIL") from error
        self._serial = serial.Serial(device, baud_rate, timeout=0.1)

    def readline(self, timeout_s: float) -> bytes:
        self._serial.timeout = timeout_s
        return bytes(self._serial.readline())

    def close(self) -> None:
        self._serial.close()


@dataclass
class HILMetrics:
    evidence_kind: str
    frames_received: int = 0
    frames_accepted: int = 0
    crc_errors: int = 0
    malformed_frames: int = 0
    sequence_gaps: int = 0
    duplicate_or_reordered: int = 0
    latency_samples_ms: tuple[float, ...] = ()

    def as_dict(self) -> dict[str, object]:
        latencies = sorted(self.latency_samples_ms)
        p95_index = max(0, math_ceil(0.95 * len(latencies)) - 1)
        return {
            "evidence_kind": self.evidence_kind,
            "frames_received": self.frames_received,
            "frames_accepted": self.frames_accepted,
            "crc_errors": self.crc_errors,
            "malformed_frames": self.malformed_frames,
            "sequence_gaps": self.sequence_gaps,
            "duplicate_or_reordered": self.duplicate_or_reordered,
            "latency_mean_ms": (
                sum(latencies) / len(latencies) if latencies else None
            ),
            "latency_p95_ms": latencies[p95_index] if latencies else None,
        }


def math_ceil(value: float) -> int:
    integer = int(value)
    return integer if value == integer else integer + 1


class DetectorHILSession:
    """Consume frames, audit the transport, and feed the existing detector buffer."""

    def __init__(
        self,
        transport: LineTransport,
        reading_buffer: ExternalDetectorReadingBuffer,
    ) -> None:
        self.transport = transport
        self.reading_buffer = reading_buffer

    def run(self, *, duration_s: float, minimum_frames: int = 1) -> HILMetrics:
        end_time = time.monotonic() + duration_s
        metrics = HILMetrics(evidence_kind=self.transport.evidence_kind)
        last_sequence: int | None = None
        latency_samples: list[float] = []
        while time.monotonic() < end_time:
            line = self.transport.readline(min(0.1, max(0.001, end_time - time.monotonic())))
            if not line:
                continue
            metrics.frames_received += 1
            try:
                frame = DetectorFrameCodec.decode(line)
            except DetectorFrameCrcError:
                metrics.crc_errors += 1
                continue
            except DetectorFrameError:
                metrics.malformed_frames += 1
                continue
            if last_sequence is not None:
                if frame.sequence <= last_sequence:
                    metrics.duplicate_or_reordered += 1
                elif frame.sequence > last_sequence + 1:
                    metrics.sequence_gaps += frame.sequence - last_sequence - 1
            last_sequence = max(last_sequence or 0, frame.sequence)
            if frame.host_sent_time_ns is not None:
                latency_samples.append(
                    max(0.0, (time.time_ns() - frame.host_sent_time_ns) / 1.0e6)
                )
            self.reading_buffer.ingest(frame.reading)
            metrics.frames_accepted += 1
        metrics.latency_samples_ms = tuple(latency_samples)
        if metrics.frames_accepted < minimum_frames:
            raise RuntimeError(
                f"detector HIL accepted {metrics.frames_accepted}, expected {minimum_frames}"
            )
        return metrics
