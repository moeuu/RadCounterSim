"""Material attenuation interpolation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from radcounter.core.models.radiation import MaterialSpec


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
        """Interpolate linear attenuation and clamp to endpoint values."""

        material = self.get(material_id)
        energies = np.asarray(energies_keV, dtype=np.float64)
        if np.any(energies <= 0) or not np.all(np.isfinite(energies)):
            raise ValueError("energies_keV must be finite and positive")
        positive = np.maximum(material.linear_attenuation_m_inv, np.finfo(float).tiny)
        values = np.exp(
            np.interp(
                np.log(energies),
                np.log(material.energies_keV),
                np.log(positive),
            )
        )
        return np.where(material.linear_attenuation_m_inv.max() == 0, 0.0, values)
