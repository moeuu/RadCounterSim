"""Quantitative source and attenuation validation against structured observations."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PhysicsObservation:
    measurement_kind: str
    energy_kev: float
    observed_counts: float
    live_time_s: float
    background_counts: float = 0.0
    background_live_time_s: float = 1.0
    reference_counts: float = 0.0
    reference_live_time_s: float = 1.0
    reference_background_counts: float = 0.0
    distance_m: float = 0.0
    reference_distance_m: float = 0.0
    material: str = ""
    thickness_mm: float = 0.0
    mass_attenuation_cm2_g: float = 0.0
    density_g_cm3: float = 0.0


@dataclass(frozen=True)
class PhysicsPointResult:
    measurement_kind: str
    energy_kev: float
    observed_value: float
    predicted_value: float
    standard_uncertainty: float
    normalized_residual: float
    material: str


@dataclass(frozen=True)
class PhysicsValidationResult:
    passed: bool
    evidence_kind: str
    point_count: int
    reduced_chi_square: float
    maximum_absolute_z: float
    mean_relative_error: float
    attenuation_mu_relative_error: Mapping[str, float]
    reasons: tuple[str, ...]
    points: tuple[PhysicsPointResult, ...]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["points"] = [asdict(point) for point in self.points]
        return payload


def _float(row: Mapping[str, str], name: str, default: float = 0.0) -> float:
    value = row.get(name, "")
    return float(value) if value not in (None, "") else default


def load_validation_dataset(
    observations_path: str | Path,
    metadata_path: str | Path,
) -> tuple[tuple[PhysicsObservation, ...], Mapping[str, object]]:
    observations = []
    with Path(observations_path).open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            observations.append(
                PhysicsObservation(
                    measurement_kind=row["measurement_kind"],
                    energy_kev=_float(row, "energy_kev"),
                    observed_counts=_float(row, "observed_counts"),
                    live_time_s=_float(row, "live_time_s"),
                    background_counts=_float(row, "background_counts"),
                    background_live_time_s=_float(row, "background_live_time_s", 1.0),
                    reference_counts=_float(row, "reference_counts"),
                    reference_live_time_s=_float(row, "reference_live_time_s", 1.0),
                    reference_background_counts=_float(
                        row, "reference_background_counts"
                    ),
                    distance_m=_float(row, "distance_m"),
                    reference_distance_m=_float(row, "reference_distance_m"),
                    material=row.get("material", ""),
                    thickness_mm=_float(row, "thickness_mm"),
                    mass_attenuation_cm2_g=_float(
                        row, "mass_attenuation_cm2_g"
                    ),
                    density_g_cm3=_float(row, "density_g_cm3"),
                )
            )
    metadata = yaml.safe_load(Path(metadata_path).read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("physics validation metadata must be a mapping")
    return tuple(observations), metadata


def _corrected_rate(
    counts: float,
    live_time_s: float,
    background_counts: float,
    background_live_time_s: float,
) -> tuple[float, float]:
    if live_time_s <= 0.0 or background_live_time_s <= 0.0:
        raise ValueError("measurement and background live times must be positive")
    rate = counts / live_time_s - background_counts / background_live_time_s
    variance = counts / live_time_s**2 + background_counts / background_live_time_s**2
    if rate <= 0.0:
        raise ValueError("background-corrected count rate must be positive")
    return rate, math.sqrt(max(variance, 1.0e-18))


def _ratio_with_uncertainty(
    numerator: tuple[float, float],
    denominator: tuple[float, float],
) -> tuple[float, float]:
    ratio = numerator[0] / denominator[0]
    relative_variance = (
        (numerator[1] / numerator[0]) ** 2
        + (denominator[1] / denominator[0]) ** 2
    )
    return ratio, ratio * math.sqrt(relative_variance)


def _nested_float(metadata: Mapping[str, object], group: str, key: str) -> float:
    group_value = metadata.get(group)
    if not isinstance(group_value, dict) or key not in group_value:
        raise ValueError(f"physics metadata requires {group}.{key}")
    return float(group_value[key])


def _validate_measured_metadata(metadata: Mapping[str, object]) -> tuple[str, ...]:
    required = (
        ("source", "isotope"),
        ("source", "certificate_id"),
        ("source", "activity_bq"),
        ("source", "reference_date"),
        ("detector", "model"),
        ("detector", "serial_number"),
        ("detector", "calibration_id"),
        ("detector", "calibration_date"),
        ("geometry", "procedure_id"),
        ("environment", "temperature_c"),
    )
    missing = []
    for group, key in required:
        value = metadata.get(group)
        if not isinstance(value, dict) or value.get(key) in (None, ""):
            missing.append(f"{group}.{key}")
    return tuple(missing)


def validate_physics_dataset(
    observations: Sequence[PhysicsObservation],
    metadata: Mapping[str, object],
    *,
    require_measured: bool = False,
    maximum_reduced_chi_square: float = 5.0,
    maximum_absolute_z: float = 5.0,
    maximum_mu_relative_error: float = 0.10,
) -> PhysicsValidationResult:
    evidence_kind = str(metadata.get("evidence_kind", "unknown"))
    reasons: list[str] = []
    if evidence_kind == "measured":
        missing = _validate_measured_metadata(metadata)
        if missing:
            reasons.append("missing measured metadata: " + ", ".join(missing))
    elif require_measured:
        reasons.append(f"physical evidence required, received {evidence_kind}")

    points = []
    attenuation_values: dict[str, list[tuple[float, float, float]]] = {}
    for observation in observations:
        observed_rate = _corrected_rate(
            observation.observed_counts,
            observation.live_time_s,
            observation.background_counts,
            observation.background_live_time_s,
        )
        if observation.measurement_kind == "absolute_source":
            activity = _nested_float(metadata, "source", "activity_bq")
            photon_yield = _nested_float(
                metadata, "source", "photon_yield_per_decay"
            )
            active_area = _nested_float(metadata, "detector", "active_area_m2")
            efficiency = _nested_float(
                metadata, "detector", "intrinsic_efficiency"
            )
            predicted = (
                activity
                * photon_yield
                * active_area
                * efficiency
                / (4.0 * math.pi * observation.distance_m**2)
            )
            observed, uncertainty = observed_rate
        else:
            reference_rate = _corrected_rate(
                observation.reference_counts,
                observation.reference_live_time_s,
                observation.reference_background_counts,
                observation.background_live_time_s,
            )
            observed, uncertainty = _ratio_with_uncertainty(
                observed_rate, reference_rate
            )
            if observation.measurement_kind == "inverse_square":
                predicted = (
                    observation.reference_distance_m / observation.distance_m
                ) ** 2
            elif observation.measurement_kind == "attenuation":
                linear_mu = (
                    observation.mass_attenuation_cm2_g
                    * observation.density_g_cm3
                )
                thickness_cm = observation.thickness_mm / 10.0
                predicted = math.exp(-linear_mu * thickness_cm)
                if thickness_cm > 0.0:
                    attenuation_values.setdefault(observation.material, []).append(
                        (thickness_cm, observed, linear_mu)
                    )
            else:
                raise ValueError(
                    f"unknown measurement kind: {observation.measurement_kind}"
                )
        normalized_residual = (observed - predicted) / max(uncertainty, 1.0e-12)
        points.append(
            PhysicsPointResult(
                measurement_kind=observation.measurement_kind,
                energy_kev=observation.energy_kev,
                observed_value=observed,
                predicted_value=predicted,
                standard_uncertainty=uncertainty,
                normalized_residual=normalized_residual,
                material=observation.material,
            )
        )

    if not points:
        raise ValueError("physics validation dataset is empty")
    chi_square = sum(point.normalized_residual**2 for point in points)
    reduced_chi_square = chi_square / max(1, len(points) - 1)
    maximum_z = max(abs(point.normalized_residual) for point in points)
    mean_relative_error = sum(
        abs(point.observed_value - point.predicted_value)
        / max(abs(point.predicted_value), 1.0e-12)
        for point in points
    ) / len(points)
    mu_errors = {}
    for material, values in attenuation_values.items():
        numerator = sum(
            thickness_cm * -math.log(max(transmission, 1.0e-15))
            for thickness_cm, transmission, _ in values
        )
        denominator = sum(thickness_cm**2 for thickness_cm, _, _ in values)
        fitted_mu = numerator / denominator
        expected_mu = sum(value[2] for value in values) / len(values)
        mu_errors[material] = abs(fitted_mu - expected_mu) / expected_mu

    if reduced_chi_square > maximum_reduced_chi_square:
        reasons.append(
            f"reduced chi-square {reduced_chi_square:.3f} exceeds "
            f"{maximum_reduced_chi_square:.3f}"
        )
    if maximum_z > maximum_absolute_z:
        reasons.append(
            f"maximum |z| {maximum_z:.3f} exceeds {maximum_absolute_z:.3f}"
        )
    for material, error in mu_errors.items():
        if error > maximum_mu_relative_error:
            reasons.append(
                f"{material} attenuation coefficient error {error:.3f} exceeds "
                f"{maximum_mu_relative_error:.3f}"
            )
    return PhysicsValidationResult(
        passed=not reasons,
        evidence_kind=evidence_kind,
        point_count=len(points),
        reduced_chi_square=reduced_chi_square,
        maximum_absolute_z=maximum_z,
        mean_relative_error=mean_relative_error,
        attenuation_mu_relative_error=mu_errors,
        reasons=tuple(reasons),
        points=tuple(points),
    )
