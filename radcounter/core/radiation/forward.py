"""Point-source spectral forward model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from radcounter.core.models.radiation import DetectorSpec, IsotopeSpec, PointSourceState
from radcounter.core.radiation.backend import RayTransportBackend

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CountRatePrediction:
    """Expected detector count rates at one or more poses."""

    count_rate_cps_per_bin: FloatArray
    direct_count_rate_cps_per_bin: FloatArray
    background_cps_per_bin: FloatArray


class RadiationForwardModel:
    """No-scatter, energy-binned point-source prediction."""

    def __init__(self, transport: RayTransportBackend, *, minimum_distance_m: float = 0.01) -> None:
        if minimum_distance_m <= 0:
            raise ValueError("minimum_distance_m must be positive")
        self._transport = transport
        self._minimum_distance_m = minimum_distance_m

    def predict_point_count_rate(
        self,
        detector_positions_world_m: FloatArray,
        sources: Sequence[PointSourceState],
        isotopes: Mapping[str, IsotopeSpec],
        detector: DetectorSpec,
    ) -> CountRatePrediction:
        """Predict direct plus background count rates for stationary poses."""

        positions = np.asarray(detector_positions_world_m, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1:] != (3,):
            raise ValueError("detector_positions_world_m must have shape (D, 3)")
        direct = np.zeros((len(positions), detector.energy_bin_count), dtype=np.float64)
        for source in sources:
            if not source.enabled or source.activity_bq == 0:
                continue
            try:
                isotope = isotopes[source.isotope_id]
            except KeyError as exc:
                raise KeyError(f"missing isotope specification: {source.isotope_id}") from exc
            origins = np.repeat(source.position_world_m[None, :], len(positions), axis=0)
            distance_m = np.maximum(
                np.linalg.norm(positions - origins, axis=1), self._minimum_distance_m
            )
            geometry = 1.0 / (4.0 * np.pi * distance_m**2)
            for line in isotope.emission_lines:
                bin_index = int(
                    np.searchsorted(detector.energy_bin_edges_keV, line.energy_keV, side="right")
                    - 1
                )
                if bin_index < 0 or bin_index >= detector.energy_bin_count:
                    continue
                efficiency = detector.efficiency_at(line.energy_keV)
                if efficiency == 0 or line.photons_per_decay == 0:
                    continue
                transmission = self._transport.trace_transmission(
                    origins, positions, np.asarray([line.energy_keV], dtype=np.float64)
                )[:, 0]
                direct[:, bin_index] += (
                    source.activity_bq
                    * line.photons_per_decay
                    * geometry
                    * transmission
                    * efficiency
                )
        background = np.repeat(detector.background_cps_per_bin[None, :], len(positions), axis=0)
        total = direct + background
        if detector.dead_time_s > 0:
            live_fraction = 1.0 / (1.0 + detector.dead_time_s * total.sum(axis=1))
            total = total * live_fraction[:, None]
            direct = direct * live_fraction[:, None]
            background = background * live_fraction[:, None]
        return CountRatePrediction(total, direct, background)

    @staticmethod
    def predict_dose_rate_sv_h(
        prediction: CountRatePrediction, detector: DetectorSpec
    ) -> FloatArray:
        """Convert binned count rate to dose rate when conversion factors exist."""

        if detector.dose_conversion_sv_h_per_cps is None:
            raise ValueError("detector has no dose conversion")
        return prediction.count_rate_cps_per_bin @ detector.dose_conversion_sv_h_per_cps
