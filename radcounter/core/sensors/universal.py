"""Detector electronics and response models consuming transported particle fields."""

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


@dataclass(frozen=True, slots=True)
class ResponseCurve:
    energies_kev: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        energies = np.asarray(self.energies_kev, dtype=np.float64)
        values = np.asarray(self.values, dtype=np.float64)
        if len(energies) < 2 or energies.shape != values.shape:
            raise ValueError("response arrays must have equal length of at least two")
        if (
            not np.all(np.isfinite(energies))
            or not np.all(np.isfinite(values))
            or np.any(energies <= 0.0)
            or np.any(np.diff(energies) <= 0.0)
            or np.any(values < 0.0)
        ):
            raise ValueError("response energies/values must be finite, ordered, and nonnegative")

    def at(self, energy_kev: float) -> float:
        if not math.isfinite(energy_kev):
            raise ValueError("response energy must be finite")
        if energy_kev < self.energies_kev[0] or energy_kev > self.energies_kev[-1]:
            raise ValueError(f"response curve does not cover incident energy {energy_kev:g} keV")
        return float(np.interp(energy_kev, self.energies_kev, self.values))


@dataclass(frozen=True, slots=True)
class ParticleResponse:
    radiation_type: RadiationType
    effective_area_m2: ResponseCurve


@dataclass(frozen=True, slots=True)
class DetectorDescriptor:
    model_id: str
    display_name: str
    family: DetectorFamily
    directionality: Directionality
    outputs: tuple[DetectorOutput, ...]
    particle_responses: tuple[ParticleResponse, ...]
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
    response_data_status: str = "synthetic_validation_only"
    response_provenance: Mapping[str, str] | None = None
    metadata: Mapping[str, str | float | int | bool] | None = None

    def __post_init__(self) -> None:
        if not self.model_id or not self.particle_responses:
            raise ValueError("detector model ID and particle responses are required")
        types = tuple(response.radiation_type for response in self.particle_responses)
        if len(types) != len(set(types)):
            raise ValueError("detector has duplicate responses for one radiation type")
        if self.background_cps < 0.0 or self.dead_time_s < 0.0:
            raise ValueError("background and dead time must be nonnegative")
        if not 0.0 <= self.off_axis_leakage_fraction <= 1.0:
            raise ValueError("off-axis leakage must be in [0, 1]")
        if not 0.0 < self.field_of_view_half_angle_deg <= 180.0:
            raise ValueError("field of view must be in (0, 180]")
        edges = np.asarray(self.energy_bin_edges_kev, dtype=np.float64)
        if len(edges) == 1 or (len(edges) >= 2 and np.any(np.diff(edges) <= 0.0)):
            raise ValueError("spectrum bin edges must be empty or strictly increasing")
        if self.response_data_status not in {
            "synthetic_validation_only",
            "experimentally_calibrated",
        }:
            raise ValueError("unsupported detector response data status")

    def response_for(self, radiation_type: RadiationType) -> ParticleResponse | None:
        return next(
            (
                response
                for response in self.particle_responses
                if response.radiation_type is radiation_type
            ),
            None,
        )


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class IncidentParticleFluence:
    """One already-transported monoenergetic contribution at a detector."""

    radiation_type: RadiationType
    energy_kev: float
    fluence_rate_m2_s: float
    arrival_direction_world: tuple[float, float, float]
    source_id: str = "source"

    def __post_init__(self) -> None:
        direction = np.asarray(self.arrival_direction_world, dtype=np.float64)
        if (
            not math.isfinite(self.energy_kev)
            or self.energy_kev <= 0.0
            or not math.isfinite(self.fluence_rate_m2_s)
            or self.fluence_rate_m2_s < 0.0
            or not np.all(np.isfinite(direction))
            or np.linalg.norm(direction) <= 1e-12
        ):
            raise ValueError("incident particle contribution is invalid")

    @property
    def normalized_arrival_direction(self) -> np.ndarray:
        value = np.asarray(self.arrival_direction_world, dtype=np.float64)
        return value / np.linalg.norm(value)


def contribution_weighted_sample_without_replacement(
    weights: Sequence[float],
    sample_count: int,
    *,
    seed: int,
) -> tuple[int, ...]:
    """Return a reproducible probability-proportional-to-size sample.

    The Gumbel-top-k construction samples positive-weight indices without
    replacement. Reusing the same seed gives every source a stable random key,
    and the selected indices are returned in canonical order so unchanged
    sample sets do not reshuffle visualization slots.
    """

    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("contribution weights must be one-dimensional")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 0:
        raise ValueError("sample count must be a nonnegative integer")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("contribution weights must be finite and nonnegative")
    positive_indices = np.flatnonzero(values > 0.0)
    selected_count = min(sample_count, len(positive_indices))
    if selected_count == 0:
        return ()

    uniforms = np.random.default_rng(seed).random(len(values))
    uniforms = np.clip(uniforms[positive_indices], np.finfo(np.float64).tiny, 1.0)
    scores = np.log(values[positive_indices]) - np.log(-np.log(uniforms))
    ranking = np.argsort(-scores, kind="stable")[:selected_count]
    return tuple(sorted(int(index) for index in positive_indices[ranking]))


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class MeasurementRequest:
    pose: DetectorPose
    incident_fluence: tuple[IncidentParticleFluence, ...]
    integration_time_s: float
    rng: np.random.Generator


class DetectorModel(Protocol):
    descriptor: DetectorDescriptor

    def measure(self, request: MeasurementRequest) -> DetectorReading: ...


class ParametricDetectorModel:
    """Apply detector response/electronics without performing radiation transport."""

    def __init__(self, descriptor: DetectorDescriptor) -> None:
        self.descriptor = descriptor

    def measure(self, request: MeasurementRequest) -> DetectorReading:
        if request.integration_time_s <= 0.0:
            raise ValueError("integration time must be positive")
        forward = request.pose.normalized_forward
        contributions: list[tuple[IncidentParticleFluence, float, np.ndarray]] = []
        ideal_cps = self.descriptor.background_cps
        weighted_direction = np.zeros(3, dtype=np.float64)
        ideal_dose_rate = 0.0
        for incident in request.incident_fluence:
            response = self.descriptor.response_for(incident.radiation_type)
            if response is None or incident.fluence_rate_m2_s <= 0.0:
                continue
            direction = incident.normalized_arrival_direction
            angular_gain = self._angular_gain(float(np.dot(forward, direction)))
            if angular_gain <= 0.0:
                continue
            effective_area = response.effective_area_m2.at(incident.energy_kev)
            cps = incident.fluence_rate_m2_s * effective_area * angular_gain
            contributions.append((incident, cps, direction))
            ideal_cps += cps
            weighted_direction += cps * direction
            ideal_dose_rate += (
                cps * incident.energy_kev * self.descriptor.dose_conversion_usv_h_per_count_kev
            )
        dead_time_cps = self._apply_dead_time(ideal_cps)
        measured_cps = dead_time_cps
        saturated = False
        if (
            self.descriptor.maximum_count_rate_cps is not None
            and measured_cps > self.descriptor.maximum_count_rate_cps
        ):
            measured_cps = self.descriptor.maximum_count_rate_cps
            saturated = True
        electronics_scale = measured_cps / ideal_cps if ideal_cps > 0.0 else 1.0
        observed = int(request.rng.poisson(measured_cps * request.integration_time_s))
        spectrum = self._spectrum(contributions, request, electronics_scale)
        direction_result = None
        if (
            DetectorOutput.DIRECTION in self.descriptor.outputs
            or DetectorOutput.IMAGE in self.descriptor.outputs
        ) and np.linalg.norm(weighted_direction) > 0.0:
            unit = weighted_direction / np.linalg.norm(weighted_direction)
            direction_result = tuple(float(value) for value in unit)
        return DetectorReading(
            detector_id=request.pose.detector_id,
            model_id=self.descriptor.model_id,
            integration_time_s=request.integration_time_s,
            expected_count_rate_cps=measured_cps,
            observed_counts=observed,
            spectrum_counts=spectrum,
            energy_bin_edges_kev=self.descriptor.energy_bin_edges_kev,
            dose_rate_usv_h=ideal_dose_rate * electronics_scale,
            estimated_direction_world=direction_result,
            saturated=saturated,
            metadata={
                "ideal_count_rate_cps": ideal_cps,
                "input_representation": "transported_particle_fluence",
                "response_data_status": self.descriptor.response_data_status,
            },
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
        contributions: Sequence[tuple[IncidentParticleFluence, float, np.ndarray]],
        request: MeasurementRequest,
        electronics_scale: float,
    ) -> tuple[int, ...]:
        edges = self.descriptor.energy_bin_edges_kev
        if DetectorOutput.SPECTRUM not in self.descriptor.outputs or len(edges) < 2:
            return ()
        expected = np.zeros(len(edges) - 1, dtype=np.float64)
        resolution = self.descriptor.energy_resolution_fwhm_fraction_at_662kev
        for incident, cps, _direction in contributions:
            counts = cps * electronics_scale * request.integration_time_s
            if resolution is None or resolution <= 0.0:
                index = np.searchsorted(edges, incident.energy_kev, side="right") - 1
                if 0 <= index < len(expected):
                    expected[index] += counts
                continue
            fwhm = resolution * math.sqrt(incident.energy_kev / 662.0) * 662.0
            sigma = max(fwhm / 2.355, 1e-6)
            centers = (np.asarray(edges[:-1]) + np.asarray(edges[1:])) * 0.5
            weights = np.exp(-0.5 * ((centers - incident.energy_kev) / sigma) ** 2)
            if weights.sum() > 0.0:
                expected += counts * weights / weights.sum()
        expected += (
            self.descriptor.background_cps
            * electronics_scale
            * request.integration_time_s
            / len(expected)
        )
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

    @property
    def detector_positions(self) -> Mapping[str, tuple[float, float, float]]:
        return {key: pose.position_world_m for key, pose in self._poses.items()}

    def measure(
        self,
        incident_by_detector: Mapping[str, Sequence[IncidentParticleFluence]],
        *,
        integration_time_s: float = 1.0,
        seed: int | None = None,
    ) -> dict[str, DetectorReading]:
        missing = set(self._models).difference(incident_by_detector)
        extra = set(incident_by_detector).difference(self._models)
        if missing or extra:
            raise ValueError(
                f"incident-field detector IDs do not match array; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        root_rng = np.random.default_rng(seed)
        readings = {}
        for detector_id, model in self._models.items():
            child_seed = int(root_rng.integers(0, np.iinfo(np.int64).max))
            readings[detector_id] = model.measure(
                MeasurementRequest(
                    pose=self._poses[detector_id],
                    incident_fluence=tuple(incident_by_detector[detector_id]),
                    integration_time_s=integration_time_s,
                    rng=np.random.default_rng(child_seed),
                )
            )
        return readings
