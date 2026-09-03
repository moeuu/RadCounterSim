"""Material attenuation interpolation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radcounter.core.models.radiation import MaterialSpec


def interpolate_attenuation_m_inv(
    energies_keV: ArrayLike,
    tabulated_energies_keV: ArrayLike,
    tabulated_attenuation_m_inv: ArrayLike,
    *,
    material_id: str = "material",
) -> NDArray[np.float64]:
    """Log-log interpolate attenuation without extrapolation or silent repair."""

    energies = np.asarray(energies_keV, dtype=np.float64)
    grid = np.asarray(tabulated_energies_keV, dtype=np.float64)
    attenuation = np.asarray(tabulated_attenuation_m_inv, dtype=np.float64)
    if energies.ndim != 1 or np.any(energies <= 0.0) or not np.all(np.isfinite(energies)):
        raise ValueError("energies_keV must be a finite positive 1-D array")
    if (
        grid.ndim != 1
        or attenuation.ndim != 1
        or grid.shape != attenuation.shape
        or len(grid) < 2
        or np.any(grid <= 0.0)
        or np.any(np.diff(grid) <= 0.0)
        or not np.all(np.isfinite(grid))
        or not np.all(np.isfinite(attenuation))
        or np.any(attenuation < 0.0)
    ):
        raise ValueError(f"invalid attenuation table for {material_id!r}")
    if np.any(energies < grid[0]) or np.any(energies > grid[-1]):
        raise ValueError(
            f"material {material_id!r} attenuation range [{grid[0]:g}, {grid[-1]:g}] keV "
            "does not cover all requested energies"
        )
    if np.all(attenuation == 0.0):
        return np.zeros_like(energies)
    if np.any(attenuation <= 0.0):
        raise ValueError(
            f"material {material_id!r} mixes zero and positive attenuation values; "
            "use an all-zero validation material or a positive physical table"
        )
    return np.exp(np.interp(np.log(energies), np.log(grid), np.log(attenuation)))


class MaterialTable:
    """Validated material lookup with log-energy interpolation."""

    def __init__(self, materials: list[MaterialSpec] | tuple[MaterialSpec, ...]) -> None:
        self._materials = {item.material_id: item for item in materials}
        if len(self._materials) != len(materials):
            raise ValueError("material_id values must be unique")

    def get(self, material_id: str) -> MaterialSpec:
        """Return one material or raise a useful error."""

        try:
            return self._materials[material_id]
        except KeyError as exc:
            raise KeyError(f"unknown material_id: {material_id}") from exc

    def attenuation_m_inv(self, material_id: str, energies_keV: ArrayLike) -> NDArray[np.float64]:
        """Interpolate linear attenuation and reject out-of-range energies."""

        material = self.get(material_id)
        return interpolate_attenuation_m_inv(
            energies_keV,
            material.energies_keV,
            material.linear_attenuation_m_inv,
            material_id=material.material_id,
        )
