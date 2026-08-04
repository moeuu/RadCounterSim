"""Composable multi-detector radiation response models."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np


class RadiationType(StrEnum):
    GAMMA = "gamma"
    X_RAY = "x_ray"
    NEUTRON = "neutron"
    ALPHA = "alpha"
    BETA = "beta"


class DetectorFamily(StrEnum):
    GEIGER_MULLER = "geiger_muller"
    ION_CHAMBER = "ion_chamber"
    SCINTILLATOR = "scintillator"
    SEMICONDUCTOR = "semiconductor"
    PROPORTIONAL_COUNTER = "proportional_counter"
    SURFACE_CONTAMINATION = "surface_contamination"
    NEUTRON_COUNTER = "neutron_counter"
    PERSONAL_DOSIMETER = "personal_dosimeter"
    GAMMA_IMAGER = "gamma_imager"


class Directionality(StrEnum):
    OMNIDIRECTIONAL = "omnidirectional"
    COSINE = "cosine"
    COLLIMATED = "collimated"
    ROTATING_COLLIMATOR = "rotating_collimator"
    CODED_APERTURE = "coded_aperture"
    COMPTON = "compton"


class DetectorOutput(StrEnum):
    COUNTS = "counts"
    COUNT_RATE = "count_rate"
    SPECTRUM = "spectrum"
    DOSE_RATE = "dose_rate"
    DIRECTION = "direction"
    IMAGE = "image"


class DeadTimeModel(StrEnum):
    NONE = "none"
    NONPARALYZABLE = "nonparalyzable"
    PARALYZABLE = "paralyzable"


@dataclass(frozen=True)
class ResponseCurve:
    energies_kev: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.energies_kev) != len(self.values) or not self.values:
            raise ValueError("response energy and value arrays must have equal non-zero length")
        if any(value < 0.0 for value in self.values):
            raise ValueError("response values must be non-negative")
        if tuple(sorted(self.energies_kev)) != self.energies_kev:
            raise ValueError("response energies must be sorted")

    def at(self, energy_kev: float) -> float:
        return float(
            np.interp(
                energy_kev,
                self.energies_kev,
                self.values,
                left=self.values[0],
                right=self.values[-1],
            )
        )


@dataclass(frozen=True)
class ParticleResponse:
    radiation_type: RadiationType
    intrinsic_efficiency: ResponseCurve
    maximum_range_m: float | None = None


@dataclass(frozen=True)
class DetectorDescriptor:
    model_id: str
    display_name: str
    family: DetectorFamily
    directionality: Directionality
    outputs: tuple[DetectorOutput, ...]
    particle_responses: tuple[ParticleResponse, ...]
    active_area_m2: float
    background_cps: float = 0.0
    dead_time_s: float = 0.0
    dead_time_model: DeadTimeModel = DeadTimeModel.NONPARALYZABLE
    maximum_count_rate_cps: float | None = None
    energy_resolution_fwhm_fraction_at_662kev: float | None = None
    energy_bin_edges_kev: tuple[float, ...] = ()
    dose_conversion_usv_h_per_count_kev: float = 0.0
    field_of_view_half_angle_deg: float = 180.0
    off_axis_leakage_fraction: float = 1.0
    angular_power: float = 1.0
    metadata: Mapping[str, str | float | int | bool] | None = None

    def __post_init__(self) -> None:
        if not self.model_id or self.active_area_m2 <= 0.0:
            raise ValueError("detector model ID and positive active area are required")
        if self.background_cps < 0.0 or self.dead_time_s < 0.0:
            raise ValueError("background and dead time must be non-negative")
        if not 0.0 <= self.off_axis_leakage_fraction <= 1.0:
            raise ValueError("off-axis leakage must be in [0, 1]")
        if not 0.0 < self.field_of_view_half_angle_deg <= 180.0:
            raise ValueError("field of view must be in (0, 180]")
        if self.energy_bin_edges_kev and len(self.energy_bin_edges_kev) < 2:
            raise ValueError("a spectrum requires at least two bin edges")

    def response_for(self, radiation_type: RadiationType) -> ParticleResponse | None:
        return next(
            (
                response
                for response in self.particle_responses
                if response.radiation_type is radiation_type
            ),
            None,
        )


@dataclass(frozen=True)
class DetectorPose:
    detector_id: str
    position_world_m: tuple[float, float, float]
    forward_world: tuple[float, float, float] = (1.0, 0.0, 0.0)
    up_world: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        forward = np.asarray(self.forward_world, dtype=np.float64)
        up = np.asarray(self.up_world, dtype=np.float64)
        if np.linalg.norm(forward) <= 1e-12 or np.linalg.norm(up) <= 1e-12:
            raise ValueError("detector forward and up vectors cannot be zero")
        if abs(float(np.dot(forward / np.linalg.norm(forward), up / np.linalg.norm(up)))) > 0.999:
            raise ValueError("detector forward and up vectors cannot be parallel")

    @property
    def normalized_forward(self) -> np.ndarray:
        value = np.asarray(self.forward_world, dtype=np.float64)
        return value / np.linalg.norm(value)


@dataclass(frozen=True)
class RadiationSample:
    position_world_m: tuple[float, float, float]
    emission_rate_per_s: float
    energy_kev: float
    radiation_type: RadiationType = RadiationType.GAMMA
    source_id: str = "source"

    def __post_init__(self) -> None:
        if self.emission_rate_per_s < 0.0 or self.energy_kev < 0.0:
            raise ValueError("radiation emission and energy must be non-negative")


@dataclass(frozen=True)
class ShieldPanel:
    shield_id: str
    center_world_m: tuple[float, float, float]
    normal_world: tuple[float, float, float]
    up_world: tuple[float, float, float]
    width_m: float
    height_m: float
    thickness_m: float
    attenuation_by_radiation: Mapping[RadiationType, ResponseCurve]

    def __post_init__(self) -> None:
        if self.width_m <= 0.0 or self.height_m <= 0.0 or self.thickness_m <= 0.0:
            raise ValueError("shield dimensions must be positive")
        normal = self._normal
        up = self._up
        if abs(float(np.dot(normal, up))) > 0.999:
            raise ValueError("shield normal and up vectors cannot be parallel")

    @property
    def _normal(self) -> np.ndarray:
        value = np.asarray(self.normal_world, dtype=np.float64)
        if np.linalg.norm(value) <= 1e-12:
            raise ValueError("shield normal cannot be zero")
        return value / np.linalg.norm(value)

    @property
    def _up(self) -> np.ndarray:
        value = np.asarray(self.up_world, dtype=np.float64)
        if np.linalg.norm(value) <= 1e-12:
            raise ValueError("shield up cannot be zero")
        value -= self._normal * np.dot(value, self._normal)
        return value / np.linalg.norm(value)

    def path_length_m(
        self,
        source_world_m: Sequence[float],
        detector_world_m: Sequence[float],
    ) -> float:
        source = np.asarray(source_world_m, dtype=np.float64)
        detector = np.asarray(detector_world_m, dtype=np.float64)
        segment = detector - source
        normal = self._normal
        denominator = float(np.dot(segment, normal))
        if abs(denominator) <= 1e-12:
            return 0.0
        fraction = float(np.dot(np.asarray(self.center_world_m) - source, normal) / denominator)
        if fraction <= 0.0 or fraction >= 1.0:
            return 0.0
        hit = source + fraction * segment
        relative = hit - np.asarray(self.center_world_m)
        up = self._up
        horizontal = np.cross(up, normal)
        if abs(float(np.dot(relative, horizontal))) > self.width_m * 0.5:
            return 0.0
        if abs(float(np.dot(relative, up))) > self.height_m * 0.5:
            return 0.0
        direction = segment / np.linalg.norm(segment)
        cosine = abs(float(np.dot(direction, normal)))
        return self.thickness_m / max(cosine, 1e-6)

    def transmission(
        self,
        sample: RadiationSample,
        detector_world_m: Sequence[float],
    ) -> float:
        curve = self.attenuation_by_radiation.get(sample.radiation_type)
        if curve is None:
            return 1.0
        path = self.path_length_m(sample.position_world_m, detector_world_m)
        return math.exp(-curve.at(sample.energy_kev) * path)


@dataclass(frozen=True)
class DetectorReading:
    detector_id: str
    model_id: str
    integration_time_s: float
    expected_count_rate_cps: float
    observed_counts: int
    spectrum_counts: tuple[int, ...] = ()
    energy_bin_edges_kev: tuple[float, ...] = ()
    dose_rate_usv_h: float = 0.0
    estimated_direction_world: tuple[float, float, float] | None = None
    saturated: bool = False
    metadata: Mapping[str, float | str | bool] | None = None


@dataclass(frozen=True)
class MeasurementRequest:
    pose: DetectorPose
    samples: tuple[RadiationSample, ...]
    shields: tuple[ShieldPanel, ...]
    integration_time_s: float
    rng: np.random.Generator


class DetectorModel(Protocol):
    descriptor: DetectorDescriptor

    def measure(self, request: MeasurementRequest) -> DetectorReading: ...


class ParametricDetectorModel:
    def __init__(self, descriptor: DetectorDescriptor) -> None:
        self.descriptor = descriptor

    def measure(self, request: MeasurementRequest) -> DetectorReading:
        if request.integration_time_s <= 0.0:
            raise ValueError("integration time must be positive")
        detector_position = np.asarray(request.pose.position_world_m, dtype=np.float64)
        forward = request.pose.normalized_forward
        contributions: list[tuple[RadiationSample, float, np.ndarray]] = []
        ideal_cps = self.descriptor.background_cps
        weighted_direction = np.zeros(3, dtype=np.float64)
        dose_rate = 0.0

        for sample in request.samples:
            response = self.descriptor.response_for(sample.radiation_type)
            if response is None or sample.emission_rate_per_s <= 0.0:
                continue
            source_position = np.asarray(sample.position_world_m, dtype=np.float64)
            vector = source_position - detector_position
            distance = float(np.linalg.norm(vector))
            if distance <= 1e-6:
                distance = 1e-6
            if response.maximum_range_m is not None and distance > response.maximum_range_m:
                continue
            unit_direction = vector / distance
            angular_gain = self._angular_gain(float(np.dot(forward, unit_direction)))
            if angular_gain <= 0.0:
                continue
            transmission = math.prod(
                shield.transmission(sample, detector_position) for shield in request.shields
            )
            efficiency = response.intrinsic_efficiency.at(sample.energy_kev)
            geometric = self.descriptor.active_area_m2 / (4.0 * math.pi * distance * distance)
            cps = sample.emission_rate_per_s * geometric * efficiency * angular_gain * transmission
            contributions.append((sample, cps, unit_direction))
            ideal_cps += cps
            weighted_direction += cps * unit_direction
            dose_rate += (
                cps * sample.energy_kev * self.descriptor.dose_conversion_usv_h_per_count_kev
            )

        measured_cps = self._apply_dead_time(ideal_cps)
        saturated = False
        if (
            self.descriptor.maximum_count_rate_cps is not None
            and measured_cps > self.descriptor.maximum_count_rate_cps
        ):
            measured_cps = self.descriptor.maximum_count_rate_cps
            saturated = True
        observed = int(request.rng.poisson(measured_cps * request.integration_time_s))
        spectrum = self._spectrum(contributions, request)
        direction = None
        if (
            DetectorOutput.DIRECTION in self.descriptor.outputs
            or DetectorOutput.IMAGE in self.descriptor.outputs
        ) and np.linalg.norm(weighted_direction) > 0.0:
            unit = weighted_direction / np.linalg.norm(weighted_direction)
            direction = tuple(float(value) for value in unit)
        return DetectorReading(
            detector_id=request.pose.detector_id,
            model_id=self.descriptor.model_id,
            integration_time_s=request.integration_time_s,
            expected_count_rate_cps=measured_cps,
            observed_counts=observed,
            spectrum_counts=spectrum,
            energy_bin_edges_kev=self.descriptor.energy_bin_edges_kev,
            dose_rate_usv_h=dose_rate,
            estimated_direction_world=direction,
            saturated=saturated,
            metadata={"ideal_count_rate_cps": ideal_cps},
        )

    def _angular_gain(self, cosine: float) -> float:
        cosine = float(np.clip(cosine, -1.0, 1.0))
        mode = self.descriptor.directionality
        if mode is Directionality.OMNIDIRECTIONAL:
            return 1.0
        if mode is Directionality.COSINE:
            return max(0.0, cosine) ** self.descriptor.angular_power
        theta_deg = math.degrees(math.acos(cosine))
        half_angle = self.descriptor.field_of_view_half_angle_deg
        leakage = self.descriptor.off_axis_leakage_fraction
        if theta_deg > half_angle:
            return leakage
        sigma = max(half_angle / 2.355, 1e-6)
        main_lobe = math.exp(-0.5 * (theta_deg / sigma) ** 2)
        return leakage + (1.0 - leakage) * main_lobe

    def _apply_dead_time(self, ideal_cps: float) -> float:
        tau = self.descriptor.dead_time_s
        if tau <= 0.0 or self.descriptor.dead_time_model is DeadTimeModel.NONE:
            return ideal_cps
        if self.descriptor.dead_time_model is DeadTimeModel.PARALYZABLE:
            return ideal_cps * math.exp(-ideal_cps * tau)
        return ideal_cps / (1.0 + ideal_cps * tau)

    def _spectrum(
        self,
        contributions: Sequence[tuple[RadiationSample, float, np.ndarray]],
        request: MeasurementRequest,
    ) -> tuple[int, ...]:
        edges = self.descriptor.energy_bin_edges_kev
        if DetectorOutput.SPECTRUM not in self.descriptor.outputs or len(edges) < 2:
            return ()
        expected = np.zeros(len(edges) - 1, dtype=np.float64)
        resolution = self.descriptor.energy_resolution_fwhm_fraction_at_662kev
        for sample, cps, _direction in contributions:
            counts = cps * request.integration_time_s
            if resolution is None or resolution <= 0.0:
                index = np.searchsorted(edges, sample.energy_kev, side="right") - 1
                if 0 <= index < len(expected):
                    expected[index] += counts
                continue
            fwhm = resolution * math.sqrt(max(sample.energy_kev, 1e-9) / 662.0) * 662.0
            sigma = max(fwhm / 2.355, 1e-6)
            centers = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) * 0.5
            weights = np.exp(-0.5 * ((centers - sample.energy_kev) / sigma) ** 2)
            if weights.sum() > 0.0:
                expected += counts * weights / weights.sum()
        expected += self.descriptor.background_cps * request.integration_time_s / len(expected)
        return tuple(int(value) for value in request.rng.poisson(expected))


class DetectorArray:
    def __init__(self) -> None:
        self._models: dict[str, DetectorModel] = {}
        self._poses: dict[str, DetectorPose] = {}

    def add(self, model: DetectorModel, pose: DetectorPose) -> None:
        if pose.detector_id in self._models:
            raise ValueError(f"duplicate detector ID: {pose.detector_id}")
        self._models[pose.detector_id] = model
        self._poses[pose.detector_id] = pose

    def remove(self, detector_id: str) -> None:
        self._models.pop(detector_id)
        self._poses.pop(detector_id)

    def set_pose(self, pose: DetectorPose) -> None:
        if pose.detector_id not in self._models:
            raise KeyError(pose.detector_id)
        self._poses[pose.detector_id] = pose

    @property
    def detector_ids(self) -> tuple[str, ...]:
        return tuple(self._models)

    def measure(
        self,
        samples: Sequence[RadiationSample],
        *,
        shields: Sequence[ShieldPanel] = (),
        integration_time_s: float = 1.0,
        seed: int | None = None,
    ) -> dict[str, DetectorReading]:
        root_rng = np.random.default_rng(seed)
        readings = {}
        for detector_id, model in self._models.items():
            child_seed = int(root_rng.integers(0, np.iinfo(np.int64).max))
            readings[detector_id] = model.measure(
                MeasurementRequest(
                    pose=self._poses[detector_id],
                    samples=tuple(samples),
                    shields=tuple(shields),
                    integration_time_s=integration_time_s,
                    rng=np.random.default_rng(child_seed),
                )
            )
        return readings
