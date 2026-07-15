"""Detector integration and Poisson observation state machine."""

from __future__ import annotations

from enum import StrEnum
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from radcounter.core.models.radiation import DetectorSpec, RadiationMeasurement


class MeasurementState(StrEnum):
    IDLE = "idle"
    INTEGRATING = "integrating"
    FINALIZING = "finalizing"
    READY = "ready"
    CANCELLED = "cancelled"


class OmnidirectionalCounter:
    """Stationary energy-binned counter with explicit lifecycle."""

    def __init__(self, detector: DetectorSpec, rng: np.random.Generator) -> None:
        self.detector = detector
        self._rng = rng
        self.state = MeasurementState.IDLE
        self._latest: RadiationMeasurement | None = None
        self._pending: dict[str, object] | None = None

    def start_measurement(
        self,
        *,
        timestamp_sim_s: float,
        duration_s: float,
        position_world_m: NDArray[np.float64],
        orientation_world_wxyz: NDArray[np.float64],
        expected_rate_cps_per_bin: NDArray[np.float64],
        scene_revision: int,
    ) -> str:
        """Start one stationary integration and return its ID."""

        if self.state == MeasurementState.INTEGRATING:
            raise RuntimeError("detector is already integrating")
        rate = np.asarray(expected_rate_cps_per_bin, dtype=np.float64)
        if duration_s <= 0 or rate.shape != (self.detector.energy_bin_count,) or np.any(rate < 0):
            raise ValueError("duration/rate are invalid")
        measurement_id = str(uuid4())
        self._pending = {
            "measurement_id": measurement_id,
            "timestamp_sim_s": timestamp_sim_s,
            "end_sim_s": timestamp_sim_s + duration_s,
            "duration_s": duration_s,
            "position_world_m": np.asarray(position_world_m, dtype=np.float64).copy(),
            "orientation_world_wxyz": np.asarray(orientation_world_wxyz, dtype=np.float64).copy(),
            "rate": rate.copy(),
            "scene_revision": scene_revision,
        }
        self.state = MeasurementState.INTEGRATING
        return measurement_id

    def update(self, sim_time_s: float) -> None:
        """Finalize the pending integration after its end time."""

        if self.state != MeasurementState.INTEGRATING or self._pending is None:
            return
        if sim_time_s < float(self._pending["end_sim_s"]):
            return
        self.state = MeasurementState.FINALIZING
        duration_s = float(self._pending["duration_s"])
        rate = np.asarray(self._pending["rate"], dtype=np.float64)
        expected = rate * duration_s
        counts = self._rng.poisson(expected).astype(np.float64)
        background = self.detector.background_cps_per_bin * duration_s
        dose_rate = None
        if self.detector.dose_conversion_sv_h_per_cps is not None:
            dose_rate = float(rate @ self.detector.dose_conversion_sv_h_per_cps)
        self._latest = RadiationMeasurement(
            measurement_id=str(self._pending["measurement_id"]),
            detector_id=self.detector.detector_id,
            timestamp_sim_s=float(self._pending["timestamp_sim_s"]),
            duration_s=duration_s,
            position_world_m=np.asarray(self._pending["position_world_m"], dtype=np.float64),
            orientation_world_wxyz=np.asarray(
                self._pending["orientation_world_wxyz"], dtype=np.float64
            ),
            counts_per_bin=counts,
            expected_background_counts=background,
            dose_rate_sv_h=dose_rate,
            covariance=np.diag(np.maximum(expected, 1.0)),
            scene_revision=int(self._pending["scene_revision"]),
        )
        self._pending = None
        self.state = MeasurementState.READY

    def cancel(self) -> None:
        """Cancel a pending integration without creating a measurement."""

        self._pending = None
        self.state = MeasurementState.CANCELLED

    def get_latest(self) -> RadiationMeasurement | None:
        """Return the latest completed immutable measurement."""

        return self._latest
