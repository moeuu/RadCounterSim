import json
import socket
import threading
import time

import pytest

from radcounter.core.sensors.plugins import ExternalDetectorReadingBuffer
from radcounter.core.sensors.universal import DetectorReading
from radcounter.validation.detector_hil import (
    DetectorFrameCodec,
    DetectorFrameCrcError,
    DetectorHILSession,
    HILFrame,
    SocketLineTransport,
)


def _frame(sequence: int) -> HILFrame:
    return HILFrame(
        sequence=sequence,
        device_time_ns=time.monotonic_ns(),
        host_sent_time_ns=time.time_ns(),
        reading=DetectorReading(
            detector_id="hil",
            model_id="czt",
            integration_time_s=0.1,
            expected_count_rate_cps=10.0,
            observed_counts=1,
        ),
    )


def test_hil_codec_detects_payload_corruption() -> None:
    encoded = DetectorFrameCodec.encode(_frame(3))
    decoded = DetectorFrameCodec.decode(encoded)
    assert decoded.sequence == 3
    payload = json.loads(encoded)
    payload["sequence"] = 4
    with pytest.raises(DetectorFrameCrcError):
        DetectorFrameCodec.decode(json.dumps(payload))


def test_hil_session_tracks_sequence_gap_and_updates_external_buffer() -> None:
    client, server = socket.socketpair()

    def writer() -> None:
        server.sendall(DetectorFrameCodec.encode(_frame(1)))
        server.sendall(DetectorFrameCodec.encode(_frame(3)))
        server.close()

    thread = threading.Thread(target=writer)
    thread.start()
    buffer = ExternalDetectorReadingBuffer(maximum_age_s=1.0)
    transport = SocketLineTransport(client, evidence_kind="synthetic_loopback")
    metrics = DetectorHILSession(transport, buffer).run(
        duration_s=0.1, minimum_frames=2
    )
    transport.close()
    thread.join()
    assert metrics.frames_accepted == 2
    assert metrics.sequence_gaps == 1
    assert buffer.latest().detector_id == "hil"
