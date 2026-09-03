"""Shared geometry transport for photons, neutrons, alpha, and beta particles."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import yaml

from radcounter.core.radiation.backend import PathLengthBatch
from radcounter.core.radiation.materials import MaterialTable
from radcounter.core.radiation.scatter import PhotonBuildupModel, PrimaryOnlyPhotonModel
from radcounter.core.sensors.universal import (
    IncidentParticleFluence,
    RadiationType,
    ResponseCurve,
)


class PathLengthProvider(Protocol):
    def trace_path_lengths(
        self, origins_m: np.ndarray, targets_m: np.ndarray
    ) -> PathLengthBatch: ...


@dataclass(frozen=True, slots=True)
class ParticleEmissionSample:
    position_world_m: tuple[float, float, float]
    emission_rate_per_s: float
    energy_kev: float
    radiation_type: RadiationType
    source_id: str

    def __post_init__(self) -> None:
        position = np.asarray(self.position_world_m, dtype=np.float64)
        if (
            position.shape != (3,)
            or not np.all(np.isfinite(position))
            or not math.isfinite(self.emission_rate_per_s)
            or self.emission_rate_per_s < 0.0
            or not math.isfinite(self.energy_kev)
            or self.energy_kev <= 0.0
            or not self.source_id
        ):
            raise ValueError("particle emission sample is invalid")


@dataclass(frozen=True, slots=True)
class ParticleTransportData:
    """Versioned material kernels used after one shared path-length trace."""

    photon_materials: MaterialTable
    neutron_removal_m_inv: Mapping[str, ResponseCurve]
    charged_particle_range_m: Mapping[RadiationType, Mapping[str, ResponseCurve]]
    open_medium_material_id: str = "air"
    data_status: str = "synthetic_validation_only"
    provenance: Mapping[str, str] | None = None
    file_sha256: str = ""
    numeric_sha256: str = ""
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if self.data_status not in {
            "synthetic_validation_only",
            "reference_calibrated",
        }:
            raise ValueError("unsupported particle transport data status")
        if not self.open_medium_material_id:
            raise ValueError("open-medium material ID is required")
        if set(self.charged_particle_range_m).difference(
            {RadiationType.ALPHA, RadiationType.BETA}
        ):
            raise ValueError("charged-particle range data may contain only alpha and beta")


class MultiParticleTransport:
    """Generate detector-local incident fields from one geometry path provider."""

    def __init__(
        self,
        path_provider: PathLengthProvider,
        data: ParticleTransportData,
        *,
        photon_buildup: PhotonBuildupModel | None = None,
        minimum_distance_m: float = 1.0e-3,
    ) -> None:
        if minimum_distance_m <= 0.0 or not math.isfinite(minimum_distance_m):
            raise ValueError("minimum transport distance must be positive")
        self.path_provider = path_provider
        self.data = data
        self.photon_buildup = photon_buildup or PrimaryOnlyPhotonModel()
        self.minimum_distance_m = minimum_distance_m

    def transport(
        self,
        emissions: tuple[ParticleEmissionSample, ...],
        detector_positions: Mapping[str, tuple[float, float, float]],
    ) -> dict[str, tuple[IncidentParticleFluence, ...]]:
        if not detector_positions:
            raise ValueError("particle transport requires at least one detector")
        if not emissions:
            return {detector_id: () for detector_id in detector_positions}
        origins = np.asarray([item.position_world_m for item in emissions], dtype=np.float64)
        result: dict[str, tuple[IncidentParticleFluence, ...]] = {}
        for detector_id, detector_position_value in detector_positions.items():
            detector_position = np.asarray(detector_position_value, dtype=np.float64)
            if detector_position.shape != (3,) or not np.all(np.isfinite(detector_position)):
                raise ValueError(f"detector {detector_id!r} has an invalid position")
            targets = np.repeat(detector_position[None, :], len(emissions), axis=0)
            paths = self.path_provider.trace_path_lengths(origins, targets)
            self._validate_paths(paths, len(emissions))
            displacement = targets - origins
            distance = np.maximum(np.linalg.norm(displacement, axis=1), self.minimum_distance_m)
            incident: list[IncidentParticleFluence] = []
            for index, emission in enumerate(emissions):
                transmission = self._transmission(emission, paths, index, distance[index])
                fluence = (
                    emission.emission_rate_per_s
                    * transmission
                    / (4.0 * math.pi * distance[index] ** 2)
                )
                arrival = -displacement[index] / distance[index]
                incident.append(
                    IncidentParticleFluence(
                        emission.radiation_type,
                        emission.energy_kev,
                        fluence,
                        tuple(float(value) for value in arrival),
                        emission.source_id,
                    )
                )
            result[detector_id] = tuple(incident)
        return result

    @staticmethod
    def _validate_paths(paths: PathLengthBatch, ray_count: int) -> None:
        if paths.lengths_m.shape != (ray_count, len(paths.material_ids)):
            raise ValueError("particle transport path dimensions are inconsistent")
        if paths.error_flags.shape != (ray_count,) or np.any(paths.error_flags):
            raise ValueError("particle transport cannot use rays with trace errors")
        if (
            not np.all(np.isfinite(paths.lengths_m))
            or np.any(paths.lengths_m < 0.0)
            or len(paths.material_ids) != len(set(paths.material_ids))
        ):
            raise ValueError("particle transport path data are invalid")

    def _transmission(
        self,
        emission: ParticleEmissionSample,
        paths: PathLengthBatch,
        row: int,
        distance_m: float,
    ) -> float:
        if emission.radiation_type in {RadiationType.GAMMA, RadiationType.X_RAY}:
            return self._photon_transmission(emission.energy_kev, paths, row)
        if emission.radiation_type is RadiationType.NEUTRON:
            return self._neutron_transmission(emission.energy_kev, paths, row)
        if emission.radiation_type in {RadiationType.ALPHA, RadiationType.BETA}:
            return self._charged_particle_transmission(
                emission.radiation_type,
                emission.energy_kev,
                paths,
                row,
                distance_m,
            )
        raise ValueError(f"unsupported radiation type: {emission.radiation_type}")

    def _photon_transmission(
        self, energy_kev: float, paths: PathLengthBatch, row: int
    ) -> float:
        exponent = 0.0
        for material_index, material_id in enumerate(paths.material_ids):
            length = float(paths.lengths_m[row, material_index])
            if length <= 0.0:
                continue
            attenuation = self.data.photon_materials.attenuation_m_inv(
                material_id, np.asarray((energy_kev,))
            )[0]
            exponent += length * float(attenuation)
        one_ray = PathLengthBatch(
            paths.material_ids,
            paths.lengths_m[row : row + 1],
            paths.error_flags[row : row + 1],
        )
        buildup = float(
            self.photon_buildup.factors(one_ray, np.asarray((energy_kev,)))[0, 0]
        )
        return float(np.clip(math.exp(-min(exponent, 700.0)) * buildup, 0.0, 1.0))

    def _neutron_transmission(
        self, energy_kev: float, paths: PathLengthBatch, row: int
    ) -> float:
        exponent = 0.0
        for material_index, material_id in enumerate(paths.material_ids):
            length = float(paths.lengths_m[row, material_index])
            if length <= 0.0:
                continue
            try:
                curve = self.data.neutron_removal_m_inv[material_id]
            except KeyError as error:
                raise ValueError(
                    f"no neutron removal data for material {material_id!r}"
                ) from error
            exponent += length * curve.at(energy_kev)
        return math.exp(-min(exponent, 700.0))

    def _charged_particle_transmission(
        self,
        radiation_type: RadiationType,
        energy_kev: float,
        paths: PathLengthBatch,
        row: int,
        distance_m: float,
    ) -> float:
        try:
            ranges = self.data.charged_particle_range_m[radiation_type]
            open_range = ranges[self.data.open_medium_material_id].at(energy_kev)
        except KeyError as error:
            raise ValueError(
                f"missing {radiation_type.value} range data for the open medium"
            ) from error
        solid_distance = float(np.sum(paths.lengths_m[row]))
        range_fraction = max(0.0, distance_m - solid_distance) / open_range
        for material_index, material_id in enumerate(paths.material_ids):
            length = float(paths.lengths_m[row, material_index])
            if length <= 0.0:
                continue
            try:
                material_range = ranges[material_id].at(energy_kev)
            except KeyError as error:
                raise ValueError(
                    f"missing {radiation_type.value} range data for material {material_id!r}"
                ) from error
            range_fraction += length / material_range
        return max(0.0, 1.0 - range_fraction)


def load_particle_transport_data(
    path: str | Path,
    photon_materials: MaterialTable,
    *,
    expected_file_sha256: str,
    required_solid_material_ids: set[str],
    required_status: str | None = None,
) -> ParticleTransportData:
    """Load neutron-removal and charged-particle range kernels with provenance."""

    source_path = Path(path).expanduser().resolve()
    if not expected_file_sha256:
        raise ValueError("particle transport data SHA256 is required")
    file_bytes = source_path.read_bytes()
    file_sha256 = hashlib.sha256(file_bytes).hexdigest()
    if file_sha256 != expected_file_sha256:
        raise ValueError("particle transport data file SHA256 mismatch")
    payload = yaml.safe_load(file_bytes)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("particle transport data require schema_version 1")
    status = str(payload.get("data_status", ""))
    if status not in {"synthetic_validation_only", "reference_calibrated"}:
        raise ValueError("particle transport data_status is invalid")
    if required_status is not None and status != required_status:
        raise ValueError(
            f"particle transport data_status={status!r}; required {required_status!r}"
        )
    open_medium = str(payload.get("open_medium_material_id", ""))
    neutron_payload = payload.get("neutron_removal_m_inv")
    charged_payload = payload.get("charged_particle_range_m")
    if not isinstance(neutron_payload, dict) or not isinstance(charged_payload, dict):
        raise ValueError("particle data require neutron and charged-particle sections")
    for material_id in required_solid_material_ids:
        try:
            photon_materials.get(material_id)
        except KeyError as error:
            raise ValueError(
                f"photon attenuation data are missing material {material_id!r}"
            ) from error

    def curve(item: object, context: str) -> ResponseCurve:
        if not isinstance(item, dict):
            raise ValueError(f"{context} must be a mapping")
        return ResponseCurve(
            tuple(float(value) for value in item["energies_kev"]),
            tuple(float(value) for value in item["values"]),
        )

    neutron = {
        str(material_id): curve(item, f"neutron {material_id}")
        for material_id, item in neutron_payload.items()
    }
    if set(neutron) != required_solid_material_ids:
        raise ValueError("neutron material coverage does not match required solid materials")
    charged: dict[RadiationType, dict[str, ResponseCurve]] = {}
    required_charged = required_solid_material_ids | {open_medium}
    for radiation_type in (RadiationType.ALPHA, RadiationType.BETA):
        type_payload = charged_payload.get(radiation_type.value)
        if not isinstance(type_payload, dict) or set(type_payload) != required_charged:
            raise ValueError(
                f"{radiation_type.value} range coverage does not match required materials"
            )
        charged[radiation_type] = {
            str(material_id): curve(item, f"{radiation_type.value} {material_id}")
            for material_id, item in type_payload.items()
        }
    numeric_payload = {
        "open_medium_material_id": open_medium,
        "neutron_removal_m_inv": neutron_payload,
        "charged_particle_range_m": charged_payload,
    }
    numeric_sha256 = hashlib.sha256(
        json.dumps(
            numeric_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    provenance_payload = payload.get("provenance")
    if not isinstance(provenance_payload, dict):
        raise ValueError("particle transport provenance is required")
    provenance = {str(key): str(value) for key, value in provenance_payload.items()}
    for key in ("source", "calibration_method", "measured_date"):
        if not provenance.get(key):
            raise ValueError(f"particle transport provenance.{key} is required")
    if status == "reference_calibrated" and provenance.get("numeric_sha256") != numeric_sha256:
        raise ValueError("particle transport numeric payload digest mismatch")
    return ParticleTransportData(
        photon_materials,
        neutron,
        charged,
        open_medium_material_id=open_medium,
        data_status=status,
        provenance=provenance,
        file_sha256=file_sha256,
        numeric_sha256=numeric_sha256,
        source_path=source_path,
    )
