"""Rotating-shield program and dose-rate meter models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import yaml
from numpy.typing import NDArray

from radcounter.core.models.radiation import DetectorSpec
from radcounter.core.radiation.data import LoadedPhysicsData, load_detector_data

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
        if (
            not np.all(np.isfinite(angles))
            or not np.all(np.isfinite(dwell))
            or np.any(dwell <= 0)
        ):
            raise ValueError("shield angles must be finite and dwell finite and positive")
        object.__setattr__(self, "angles_deg", angles)
        object.__setattr__(self, "dwell_s", dwell)


@dataclass(frozen=True)
class ShieldProgramMeasurement:
    """Per-posture expected and observed spectral counts."""

    measurement_id: str
    commanded_angles_deg: FloatArray
    actual_angles_deg: FloatArray
    encoder_angles_deg: FloatArray
    dwell_s: FloatArray
    expected_counts_per_posture_bin: FloatArray
    observed_counts_per_posture_bin: FloatArray
    mode: RotatingShieldMode


@dataclass(frozen=True)
class PhysicalGeometryBinding:
    """USD paths needed to drive a rotating attenuation geometry."""

    detector_prim_path: str
    shield_rotation_prim_path: str
    rotation_attribute: str
    source_prim_paths: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("detector_prim_path", self.detector_prim_path),
            ("shield_rotation_prim_path", self.shield_rotation_prim_path),
        ):
            if not value.startswith("/"):
                raise ValueError(f"{name} must be an absolute USD prim path")
        detector_rotates_with_shield = self.detector_prim_path.startswith(
            self.shield_rotation_prim_path.rstrip("/") + "/"
        )
        if (
            self.detector_prim_path == self.shield_rotation_prim_path
            or detector_rotates_with_shield
        ):
            raise ValueError("the detector must not rotate with its shielding geometry")
        if self.rotation_attribute not in {
            "xformOp:rotateX",
            "xformOp:rotateY",
            "xformOp:rotateZ",
        }:
            raise ValueError("rotation_attribute must select a scalar USD rotation op")
        if self.source_prim_paths is not None and (
            not self.source_prim_paths
            or any(not path.startswith("/") for path in self.source_prim_paths)
        ):
            raise ValueError("source_prim_paths must contain absolute USD prim paths")


@dataclass(frozen=True)
class RotatingShieldConfiguration:
    """Validated detector, posture program, and optional USD binding."""

    detector_data: LoadedPhysicsData
    mode: RotatingShieldMode
    program: ShieldProgram
    response_mask_per_posture_bin: FloatArray | None
    encoder_noise_std_deg: float
    posture_error_std_deg: float
    physical_geometry: PhysicalGeometryBinding | None

    @property
    def detector(self) -> DetectorSpec:
        value = self.detector_data.value
        if not isinstance(value, DetectorSpec):
            raise TypeError("rotating-shield detector data did not contain a DetectorSpec")
        return value

    @property
    def data_status(self) -> str:
        return self.detector_data.data_status

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        required_status: str | None = None,
    ) -> RotatingShieldConfiguration:
        """Load one fail-closed rotating-shield configuration."""

        source_path = Path(path).expanduser().resolve()
        payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("rotating-shield configuration root must be a mapping")
        detector_data = load_detector_data(source_path, required_status=required_status)
        try:
            mode = RotatingShieldMode(str(payload["mode"]))
            program_payload = _mapping(payload["program"], name="program")
            program = ShieldProgram(
                np.asarray(program_payload["angles_deg"], dtype=np.float64),
                np.asarray(program_payload["dwell_s"], dtype=np.float64),
            )
        except KeyError as error:
            raise ValueError(
                f"rotating-shield configuration is missing {error.args[0]!r}"
            ) from error
        mask_payload = payload.get("response_mask_per_posture_bin")
        mask = (
            None
            if mask_payload is None
            else np.asarray(mask_payload, dtype=np.float64)
        )
        binding_payload = payload.get("physical_geometry")
        binding = None
        if binding_payload is not None:
            values = _mapping(binding_payload, name="physical_geometry")
            source_paths = values.get("source_prim_paths")
            binding = PhysicalGeometryBinding(
                detector_prim_path=str(values.get("detector_prim_path", "")),
                shield_rotation_prim_path=str(values.get("shield_rotation_prim_path", "")),
                rotation_attribute=str(values.get("rotation_attribute", "")),
                source_prim_paths=(
                    None
                    if source_paths is None
                    else tuple(
                        str(item)
                        for item in _sequence(source_paths, name="source_prim_paths")
                    )
                ),
            )
        if mode is RotatingShieldMode.PHYSICAL_GEOMETRY:
            if binding is None:
                raise ValueError("physical-geometry mode requires physical_geometry")
            if mask is not None:
                raise ValueError("physical-geometry mode cannot use a response mask")
        elif binding is not None:
            raise ValueError("response-mask mode cannot declare physical_geometry")
        if mask is not None and (
            mask.ndim != 2 or mask.shape[0] != len(program.angles_deg)
        ):
            raise ValueError("response mask posture count must match the configured program")
        configuration = cls(
            detector_data=detector_data,
            mode=mode,
            program=program,
            response_mask_per_posture_bin=mask,
            encoder_noise_std_deg=float(payload.get("encoder_noise_std_deg", 0.0)),
            posture_error_std_deg=float(payload.get("posture_error_std_deg", 0.0)),
            physical_geometry=binding,
        )
        # Reuse the runtime class' complete dimensional and range validation.
        configuration.create_counter(np.random.default_rng(0))
        return configuration

    def create_counter(self, rng: np.random.Generator) -> RotatingShieldCounter:
        return RotatingShieldCounter(
            self.detector,
            rng,
            mode=self.mode,
            response_mask_per_posture_bin=self.response_mask_per_posture_bin,
            encoder_noise_std_deg=self.encoder_noise_std_deg,
            posture_error_std_deg=self.posture_error_std_deg,
        )


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, *, name: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    return value


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
        posture_error_std_deg: float = 0.0,
    ) -> None:
        if (
            not np.isfinite(encoder_noise_std_deg)
            or encoder_noise_std_deg < 0
            or not np.isfinite(posture_error_std_deg)
            or posture_error_std_deg < 0
        ):
            raise ValueError("encoder noise and posture error must be finite and nonnegative")
        self.detector = detector
        self._rng = rng
        self.mode = mode
        self._mask = None
        if response_mask_per_posture_bin is not None:
            mask = np.asarray(response_mask_per_posture_bin, dtype=np.float64)
            if (
                mask.ndim != 2
                or mask.shape[1] != detector.energy_bin_count
                or not np.all(np.isfinite(mask))
            ):
                raise ValueError("response mask must have shape (P, energy_bin_count)")
            if np.any(mask < 0) or np.any(mask > 1):
                raise ValueError("response mask values must be in [0,1]")
            self._mask = mask
        if mode == RotatingShieldMode.RESPONSE_MASK and self._mask is None:
            raise ValueError("response-mask mode requires a response mask")
        if mode == RotatingShieldMode.PHYSICAL_GEOMETRY and self._mask is not None:
            raise ValueError("physical-geometry mode cannot accept a response mask")
        self._encoder_noise_std_deg = encoder_noise_std_deg
        self._posture_error_std_deg = posture_error_std_deg

    def measure_program(
        self,
        program: ShieldProgram,
        *,
        unshielded_rate_cps_per_bin: FloatArray | None = None,
        physical_rate_provider: Callable[[float], FloatArray] | None = None,
    ) -> ShieldProgramMeasurement:
        """Execute all postures and sample Poisson counts."""

        actual_angles = program.angles_deg.copy()
        if self.mode is RotatingShieldMode.PHYSICAL_GEOMETRY:
            actual_angles += self._rng.normal(
                0.0, self._posture_error_std_deg, size=len(program.angles_deg)
            )
        encoder_angles = actual_angles + self._rng.normal(
            0.0, self._encoder_noise_std_deg, size=len(program.angles_deg)
        )
        expected = np.zeros(
            (len(program.angles_deg), self.detector.energy_bin_count), dtype=np.float64
        )
        if self.mode == RotatingShieldMode.RESPONSE_MASK:
            if unshielded_rate_cps_per_bin is None or self._mask is None:
                raise ValueError("response-mask mode requires unshielded rates")
            rate = np.asarray(unshielded_rate_cps_per_bin, dtype=np.float64)
            if (
                rate.shape != (self.detector.energy_bin_count,)
                or not np.all(np.isfinite(rate))
                or np.any(rate < 0)
            ):
                raise ValueError("unshielded rate shape does not match detector bins")
            if len(self._mask) != len(program.angles_deg):
                raise ValueError("response mask posture count does not match program")
            expected = self._mask * rate[None, :] * program.dwell_s[:, None]
        else:
            if physical_rate_provider is None:
                raise ValueError("physical-geometry mode requires a rate provider")
            for index, angle_deg in enumerate(actual_angles):
                rate = np.asarray(physical_rate_provider(float(angle_deg)), dtype=np.float64)
                if (
                    rate.shape != (self.detector.energy_bin_count,)
                    or not np.all(np.isfinite(rate))
                    or np.any(rate < 0)
                ):
                    raise ValueError("physical rate provider returned an invalid spectrum")
                expected[index] = rate * program.dwell_s[index]
        return ShieldProgramMeasurement(
            measurement_id=str(uuid4()),
            commanded_angles_deg=program.angles_deg.copy(),
            actual_angles_deg=actual_angles,
            encoder_angles_deg=encoder_angles,
            dwell_s=program.dwell_s.copy(),
            expected_counts_per_posture_bin=expected,
            observed_counts_per_posture_bin=self._rng.poisson(expected).astype(np.float64),
            mode=self.mode,
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
