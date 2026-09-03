"""Radiation domain values with explicit SI/keV unit names."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _float_array(value: object, *, ndim: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-D array")
    return array


class SourceType(StrEnum):
    """Supported source geometry types."""

    POINT = "point"
    SURFACE = "surface"
    VOLUME = "volume"


@dataclass(frozen=True)
class EmissionLine:
    """One isotope photon line."""

    energy_keV: float
    photons_per_decay: float

    def __post_init__(self) -> None:
        if self.energy_keV <= 0 or self.photons_per_decay < 0:
            raise ValueError("emission energy must be positive and yield nonnegative")


@dataclass(frozen=True)
class IsotopeSpec:
    """Emission data for an isotope or synthetic validation nuclide."""

    isotope_id: str
    emission_lines: tuple[EmissionLine, ...]

    def __post_init__(self) -> None:
        if not self.isotope_id or not self.emission_lines:
            raise ValueError("isotope_id and at least one emission line are required")


@dataclass
class PointSourceState:
    """Mutable truth or belief point-source state."""

    source_id: str
    position_world_m: FloatArray
    activity_bq: float
    isotope_id: str
    enabled: bool = True
    attached_prim_path: str | None = None

    def __post_init__(self) -> None:
        self.position_world_m = _float_array(self.position_world_m, ndim=1, name="position_world_m")
        if self.position_world_m.shape != (3,):
            raise ValueError("position_world_m must have shape (3,)")
        if not self.source_id or not self.isotope_id or self.activity_bq < 0:
            raise ValueError("source id/isotope are required and activity must be nonnegative")


@dataclass
class SurfaceSourceState:
    """Per-triangle surface activity state."""

    source_id: str
    prim_path: str
    triangle_indices: IntArray
    activity_bq_per_triangle: FloatArray
    isotope_id: str
    enabled: bool = True

    def __post_init__(self) -> None:
        self.triangle_indices = np.asarray(self.triangle_indices, dtype=np.int64)
        self.activity_bq_per_triangle = _float_array(
            self.activity_bq_per_triangle, ndim=1, name="activity_bq_per_triangle"
        )
        if self.triangle_indices.ndim != 1:
            raise ValueError("triangle_indices must be one-dimensional")
        if self.triangle_indices.shape != self.activity_bq_per_triangle.shape:
            raise ValueError("triangle and activity arrays must have equal shape")
        if np.any(self.triangle_indices < 0) or np.any(self.activity_bq_per_triangle < 0):
            raise ValueError("triangle indices and activity must be nonnegative")


@dataclass(frozen=True)
class MaterialSpec:
    """Energy-dependent linear attenuation data."""

    material_id: str
    energies_keV: FloatArray
    linear_attenuation_m_inv: FloatArray
    geometry_mode: str = "solid"
    explicit_thickness_m: float | None = None

    def __post_init__(self) -> None:
        energies = _float_array(self.energies_keV, ndim=1, name="energies_keV")
        attenuation = _float_array(
            self.linear_attenuation_m_inv, ndim=1, name="linear_attenuation_m_inv"
        )
        if energies.shape != attenuation.shape or len(energies) < 2:
            raise ValueError(
                "material energy and attenuation grids must match and contain 2+ points"
            )
        if np.any(energies <= 0) or np.any(np.diff(energies) <= 0) or np.any(attenuation < 0):
            raise ValueError("material grid must be increasing, positive, and nonnegative")
        if self.geometry_mode not in {"solid", "thin_sheet"}:
            raise ValueError("geometry_mode must be solid or thin_sheet")
        if self.geometry_mode == "thin_sheet" and (
            self.explicit_thickness_m is None or self.explicit_thickness_m <= 0
        ):
            raise ValueError("thin_sheet materials require positive explicit_thickness_m")
        object.__setattr__(self, "energies_keV", energies)
        object.__setattr__(self, "linear_attenuation_m_inv", attenuation)


@dataclass(frozen=True)
class DetectorSpec:
    """Energy-binned effective-area response and electronics model.

    ``effective_area_m2_per_bin`` maps incident particle fluence rate at each
    response energy to an output count-rate spectrum.  Keeping the response in
    square metres makes the forward-model units explicit::

        particles / (m2 s) * effective area (m2) = counts / s

    A row may distribute one incident energy across several output bins, so the
    same representation covers integral counters and spectrometers without a
    detector-specific transport path.
    """

    detector_id: str
    energy_bin_edges_keV: FloatArray
    response_energy_keV: FloatArray
    effective_area_m2_per_bin: FloatArray
    background_cps_per_bin: FloatArray
    dead_time_s: float = 0.0
    dose_conversion_sv_h_per_cps: FloatArray | None = None

    def __post_init__(self) -> None:
        if not self.detector_id:
            raise ValueError("detector_id is required")
        edges = _float_array(self.energy_bin_edges_keV, ndim=1, name="energy_bin_edges_keV")
        energy = _float_array(self.response_energy_keV, ndim=1, name="response_energy_keV")
        response = _float_array(
            self.effective_area_m2_per_bin,
            ndim=2,
            name="effective_area_m2_per_bin",
        )
        background = _float_array(
            self.background_cps_per_bin, ndim=1, name="background_cps_per_bin"
        )
        if len(edges) < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError("energy bin edges must be strictly increasing")
        if len(energy) < 2 or np.any(energy <= 0) or np.any(np.diff(energy) <= 0):
            raise ValueError("detector response energies must be positive and strictly increasing")
        if response.shape != (len(energy), len(edges) - 1):
            raise ValueError(
                "effective-area response must have shape "
                "(response_energy_count, energy_bin_count)"
            )
        if background.shape != (len(edges) - 1,):
            raise ValueError("background must have one value per energy bin")
        if np.any(response < 0) or np.any(background < 0):
            raise ValueError("effective area and background must be nonnegative")
        if self.dead_time_s < 0:
            raise ValueError("dead_time_s must be nonnegative")
        conversion = self.dose_conversion_sv_h_per_cps
        if conversion is not None:
            conversion = _float_array(conversion, ndim=1, name="dose_conversion")
            if conversion.shape != background.shape or np.any(conversion < 0):
                raise ValueError("dose conversion must be nonnegative and match bins")
        object.__setattr__(self, "energy_bin_edges_keV", edges)
        object.__setattr__(self, "response_energy_keV", energy)
        object.__setattr__(self, "effective_area_m2_per_bin", response)
        object.__setattr__(self, "background_cps_per_bin", background)
        object.__setattr__(self, "dose_conversion_sv_h_per_cps", conversion)

    @property
    def energy_bin_count(self) -> int:
        """Return the number of output energy bins."""

        return len(self.energy_bin_edges_keV) - 1

    def effective_area_m2_at(self, energy_keV: float) -> FloatArray:
        """Interpolate the effective-area output vector for one incident energy.

        Response data are not extrapolated.  Callers must provide a detector
        dataset covering every modeled emission energy; otherwise evaluation
        fails explicitly instead of silently clamping or inventing a response.
        """

        if not np.isfinite(energy_keV):
            raise ValueError("incident energy must be finite")
        if energy_keV < self.response_energy_keV[0] or energy_keV > self.response_energy_keV[-1]:
            raise ValueError(
                f"incident energy {energy_keV:g} keV is outside detector "
                f"response range [{self.response_energy_keV[0]:g}, "
                f"{self.response_energy_keV[-1]:g}] keV"
            )
        return np.asarray(
            [
                np.interp(
                    energy_keV,
                    self.response_energy_keV,
                    self.effective_area_m2_per_bin[:, bin_index],
                )
                for bin_index in range(self.energy_bin_count)
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class RadiationMeasurement:
    """Completed detector integration."""

    measurement_id: str
    detector_id: str
    timestamp_sim_s: float
    duration_s: float
    position_world_m: FloatArray
    orientation_world_wxyz: FloatArray
    counts_per_bin: FloatArray
    expected_background_counts: FloatArray
    dose_rate_sv_h: float | None
    covariance: FloatArray
    scene_revision: int
