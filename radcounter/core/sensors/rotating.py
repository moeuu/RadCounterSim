"""Rotating-shield program and dose-rate meter models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from radcounter.core.models.radiation import DetectorSpec

FloatArray = NDArray[np.float64]


class RotatingShieldMode(StrEnum):
    PHYSICAL_GEOMETRY = "physical_geometry"
    RESPONSE_MASK = "response_mask"


@dataclass(frozen=True)
class ShieldProgram:
    """Ordered shield angles and dwell times."""

    angles_deg: FloatArray
    dwell_s: FloatArray

    def __post_init__(self) -> None:
        angles = np.asarray(self.angles_deg, dtype=np.float64)
        dwell = np.asarray(self.dwell_s, dtype=np.float64)
        if angles.ndim != 1 or angles.shape != dwell.shape or len(angles) == 0:
            raise ValueError("shield program arrays must be equal non-empty vectors")
        if not np.all(np.isfinite(angles)) or np.any(dwell <= 0):
            raise ValueError("shield angles must be finite and dwell positive")
        object.__setattr__(self, "angles_deg", angles)
        object.__setattr__(self, "dwell_s", dwell)


@dataclass(frozen=True)
class ShieldProgramMeasurement:
    """Per-posture expected and observed spectral counts."""

    measurement_id: str
    commanded_angles_deg: FloatArray
    encoder_angles_deg: FloatArray
    dwell_s: FloatArray
    expected_counts_per_posture_bin: FloatArray
    observed_counts_per_posture_bin: FloatArray
    mode: RotatingShieldMode


class RotatingShieldCounter:
    """Counter supporting precomputed masks or a physical rate provider."""

    def __init__(
        self,
        detector: DetectorSpec,
        rng: np.random.Generator,
        *,
        mode: RotatingShieldMode,
        response_mask_per_posture_bin: FloatArray | None = None,
        encoder_noise_std_deg: float = 0.0,
    ) -> None:
        if encoder_noise_std_deg < 0:
            raise ValueError("encoder noise must be nonnegative")
        self.detector = detector
        self._rng = rng
        self.mode = mode
        self._mask = None
        if response_mask_per_posture_bin is not None:
            mask = np.asarray(response_mask_per_posture_bin, dtype=np.float64)
            if mask.ndim != 2 or mask.shape[1] != detector.energy_bin_count:
                raise ValueError("response mask must have shape (P, energy_bin_count)")
            if np.any(mask < 0) or np.any(mask > 1):
                raise ValueError("response mask values must be in [0,1]")
            self._mask = mask
        if mode == RotatingShieldMode.RESPONSE_MASK and self._mask is None:
            raise ValueError("response-mask mode requires a response mask")
        self._encoder_noise_std_deg = encoder_noise_std_deg

    def measure_program(
        self,
        program: ShieldProgram,
        *,
        unshielded_rate_cps_per_bin: FloatArray | None = None,
        physical_rate_provider: Callable[[float], FloatArray] | None = None,
    ) -> ShieldProgramMeasurement:
        """Execute all postures and sample Poisson counts."""

        encoder_angles = program.angles_deg + self._rng.normal(
            0.0, self._encoder_noise_std_deg, size=len(program.angles_deg)
        )
        expected = np.zeros(
            (len(program.angles_deg), self.detector.energy_bin_count), dtype=np.float64
        )
        if self.mode == RotatingShieldMode.RESPONSE_MASK:
            if unshielded_rate_cps_per_bin is None or self._mask is None:
                raise ValueError("response-mask mode requires unshielded rates")
            rate = np.asarray(unshielded_rate_cps_per_bin, dtype=np.float64)
            if rate.shape != (self.detector.energy_bin_count,):
                raise ValueError("unshielded rate shape does not match detector bins")
            if len(self._mask) != len(program.angles_deg):
                raise ValueError("response mask posture count does not match program")
            expected = self._mask * rate[None, :] * program.dwell_s[:, None]
        else:
            if physical_rate_provider is None:
                raise ValueError("physical-geometry mode requires a rate provider")
            for index, angle_deg in enumerate(encoder_angles):
                rate = np.asarray(physical_rate_provider(float(angle_deg)), dtype=np.float64)
                if rate.shape != (self.detector.energy_bin_count,) or np.any(rate < 0):
                    raise ValueError("physical rate provider returned an invalid spectrum")
                expected[index] = rate * program.dwell_s[index]
        return ShieldProgramMeasurement(
            str(uuid4()),
            program.angles_deg.copy(),
            encoder_angles,
            program.dwell_s.copy(),
            expected,
            self._rng.poisson(expected).astype(np.float64),
            self.mode,
        )


class DoseRateMeter:
    """Convert a binned count-rate vector to dose rate."""

    def __init__(self, detector: DetectorSpec) -> None:
        if detector.dose_conversion_sv_h_per_cps is None:
            raise ValueError("dose-rate meter requires dose conversion factors")
        self.detector = detector

    def dose_rate_sv_h(self, count_rate_cps_per_bin: FloatArray) -> float:
        """Return scalar dose rate for one spectrum."""

        count_rate = np.asarray(count_rate_cps_per_bin, dtype=np.float64)
        if count_rate.shape != (self.detector.energy_bin_count,) or np.any(count_rate < 0):
            raise ValueError("count-rate vector does not match detector bins")
        assert self.detector.dose_conversion_sv_h_per_cps is not None
        return float(count_rate @ self.detector.dose_conversion_sv_h_per_cps)
