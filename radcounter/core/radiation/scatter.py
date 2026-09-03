"""Reference-calibrated photon buildup corrections over traced material paths."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml
from numpy.typing import NDArray

from radcounter.core.radiation.backend import PathLengthBatch
from radcounter.core.radiation.materials import MaterialTable

FloatArray = NDArray[np.float64]


class PhotonBuildupModel(Protocol):
    """Path-aware multiplicative correction applied before detector response."""

    model_name: str

    def factors(
        self,
        path_lengths: PathLengthBatch,
        energies_keV: FloatArray,
    ) -> FloatArray: ...


@dataclass(frozen=True, slots=True)
class PrimaryOnlyPhotonModel:
    """Explicitly represent primary photons without a scatter correction."""

    model_name: str = "primary_only"

    def factors(
        self,
        path_lengths: PathLengthBatch,
        energies_keV: FloatArray,
    ) -> FloatArray:
        energies = np.asarray(energies_keV, dtype=np.float64)
        return np.ones((len(path_lengths.lengths_m), len(energies)), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class MaterialBuildupSurface:
    """Buildup factor tabulated against energy and optical depth ``mu*x``."""

    material_id: str
    energies_keV: FloatArray
    optical_depths: FloatArray
    factors_by_energy_depth: FloatArray

    def __post_init__(self) -> None:
        energies = np.asarray(self.energies_keV, dtype=np.float64)
        depths = np.asarray(self.optical_depths, dtype=np.float64)
        factors = np.asarray(self.factors_by_energy_depth, dtype=np.float64)
        if not self.material_id:
            raise ValueError("buildup surface requires a material ID")
        if (
            energies.ndim != 1
            or depths.ndim != 1
            or len(energies) < 2
            or len(depths) < 2
            or np.any(energies <= 0.0)
            or np.any(np.diff(energies) <= 0.0)
            or depths[0] != 0.0
            or np.any(np.diff(depths) <= 0.0)
            or factors.shape != (len(energies), len(depths))
            or not np.all(np.isfinite(factors))
            or np.any(factors < 1.0)
        ):
            raise ValueError(f"invalid buildup surface for {self.material_id!r}")
        if not np.allclose(factors[:, 0], 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("buildup factor at zero optical depth must equal one")
        object.__setattr__(self, "energies_keV", energies)
        object.__setattr__(self, "optical_depths", depths)
        object.__setattr__(self, "factors_by_energy_depth", factors)

    def at(self, energies_keV: FloatArray, optical_depth: FloatArray) -> FloatArray:
        """Bilinearly interpolate without extrapolating either calibrated axis."""

        energies = np.asarray(energies_keV, dtype=np.float64)
        depths = np.asarray(optical_depth, dtype=np.float64)
        if depths.ndim != 2 or depths.shape[1] != len(energies):
            raise ValueError("optical depth must have shape (ray_count, energy_count)")
        if np.any(energies < self.energies_keV[0]) or np.any(
            energies > self.energies_keV[-1]
        ):
            raise ValueError(
                f"buildup data for {self.material_id!r} do not cover requested energies"
            )
        if np.any(depths < 0.0) or np.any(depths > self.optical_depths[-1]):
            raise ValueError(
                f"buildup data for {self.material_id!r} do not cover traced optical depth"
            )
        result = np.empty_like(depths)
        log_grid = np.log(self.energies_keV)
        for energy_index, energy in enumerate(energies):
            upper = int(np.searchsorted(self.energies_keV, energy, side="right"))
            upper = min(max(upper, 1), len(self.energies_keV) - 1)
            lower = upper - 1
            denominator = log_grid[upper] - log_grid[lower]
            weight = (np.log(energy) - log_grid[lower]) / denominator
            lower_factor = np.interp(
                depths[:, energy_index],
                self.optical_depths,
                self.factors_by_energy_depth[lower],
            )
            upper_factor = np.interp(
                depths[:, energy_index],
                self.optical_depths,
                self.factors_by_energy_depth[upper],
            )
            result[:, energy_index] = lower_factor + weight * (
                upper_factor - lower_factor
            )
        return result


@dataclass(frozen=True, slots=True)
class ReferenceCalibratedBuildupModel:
    """Compose per-material corrections using traced optical path lengths."""

    material_table: MaterialTable
    surfaces: tuple[MaterialBuildupSurface, ...]
    model_name: str = "reference_calibrated_optical_depth_buildup"

    def __post_init__(self) -> None:
        identifiers = [surface.material_id for surface in self.surfaces]
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError("buildup surfaces require unique material IDs")

    def factors(
        self,
        path_lengths: PathLengthBatch,
        energies_keV: FloatArray,
    ) -> FloatArray:
        energies = np.asarray(energies_keV, dtype=np.float64)
        if energies.ndim != 1 or np.any(energies <= 0.0):
            raise ValueError("buildup energies must be a positive vector")
        if path_lengths.lengths_m.shape != (
            len(path_lengths.error_flags),
            len(path_lengths.material_ids),
        ):
            raise ValueError("path-length batch dimensions are inconsistent")
        if np.any(path_lengths.error_flags):
            raise ValueError("cannot apply buildup to rays with transport errors")
        by_material = {surface.material_id: surface for surface in self.surfaces}
        correction = np.ones(
            (len(path_lengths.lengths_m), len(energies)), dtype=np.float64
        )
        for material_index, material_id in enumerate(path_lengths.material_ids):
            lengths = path_lengths.lengths_m[:, material_index]
            active = lengths > 0.0
            if not np.any(active):
                continue
            try:
                surface = by_material[material_id]
            except KeyError as error:
                raise ValueError(
                    f"no buildup calibration is available for material {material_id!r}"
                ) from error
            attenuation = self.material_table.attenuation_m_inv(material_id, energies)
            optical_depth = lengths[:, None] * attenuation[None, :]
            material_factor = surface.at(energies, optical_depth)
            correction[active] *= material_factor[active]
        return correction


@dataclass(frozen=True, slots=True)
class LoadedBuildupData:
    data_status: str
    provenance: dict[str, Any]
    payload_sha256: str
    source_path: Path
    surfaces: tuple[MaterialBuildupSurface, ...]


def _payload_hash(payload: object) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def load_buildup_data(
    path: str | Path,
    *,
    required_status: str | None = None,
) -> LoadedBuildupData:
    """Load versioned buildup tables with provenance and numeric-payload integrity."""

    source_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("buildup data root must be a mapping")
    status = str(payload.get("data_status", ""))
    if not status:
        raise ValueError("buildup data_status is required")
    if required_status is not None and status != required_status:
        raise ValueError(
            f"buildup data_status={status!r}; required {required_status!r}"
        )
    if payload.get("composition_rule") != "product_by_material":
        raise ValueError("buildup composition_rule must be product_by_material")
    numeric_payload = {
        "composition_rule": payload.get("composition_rule"),
        "materials": payload.get("materials"),
    }
    digest = _payload_hash(numeric_payload)
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("buildup provenance must be a mapping")
    if status != "synthetic_validation_only":
        for key in ("source_name", "retrieved_on", "payload_sha256"):
            if provenance.get(key) in (None, ""):
                raise ValueError(f"buildup provenance.{key} is required")
        if provenance["payload_sha256"] != digest:
            raise ValueError("buildup numeric payload digest mismatch")
    materials = payload.get("materials")
    if not isinstance(materials, list) or not materials:
        raise ValueError("buildup materials must be a nonempty list")
    surfaces = tuple(
        MaterialBuildupSurface(
            material_id=str(item["material_id"]),
            energies_keV=np.asarray(item["energies_keV"], dtype=np.float64),
            optical_depths=np.asarray(item["optical_depths"], dtype=np.float64),
            factors_by_energy_depth=np.asarray(
                item["factors_by_energy_depth"], dtype=np.float64
            ),
        )
        for item in materials
    )
    return LoadedBuildupData(status, provenance, digest, source_path, surfaces)
