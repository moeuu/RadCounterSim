"""Forward model for point/surface/volume quadrature samples."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from radcounter.core.models.radiation import DetectorSpec, IsotopeSpec
from radcounter.core.radiation.backend import RayTransportBackend
from radcounter.core.radiation.sampling import SourceSampleBatch
from radcounter.core.radiation.scatter import PhotonBuildupModel, PrimaryOnlyPhotonModel

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SampleCountRatePrediction:
    """Expected binned rates from a sample batch."""

    count_rate_cps_per_bin: FloatArray
    direct_count_rate_cps_per_bin: FloatArray
    background_cps_per_bin: FloatArray
    buildup_model_name: str


class SampledRadiationForwardModel:
    """Linear direct model plus explicit scatter/background/electronics."""

    def __init__(
        self,
        transport: RayTransportBackend,
        *,
        minimum_distance_m: float = 0.01,
        buildup_model: PhotonBuildupModel | None = None,
    ) -> None:
        if minimum_distance_m <= 0:
            raise ValueError("minimum_distance_m must be positive")
        self._transport = transport
        self._minimum_distance_m = minimum_distance_m
        self._buildup = (
            buildup_model if buildup_model is not None else PrimaryOnlyPhotonModel()
        )

    def predict_count_rate(
        self,
        detector_positions_world_m: FloatArray,
        source_samples: SourceSampleBatch,
        isotopes: Sequence[IsotopeSpec],
        detector: DetectorSpec,
        *,
        include_background: bool = True,
        apply_dead_time: bool = True,
    ) -> SampleCountRatePrediction:
        """Predict rates while preserving the linear direct component."""

        positions = np.asarray(detector_positions_world_m, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1:] != (3,):
            raise ValueError("detector positions must have shape (D,3)")
        if source_samples.sample_count and np.max(source_samples.isotope_index) >= len(isotopes):
            raise ValueError("source sample isotope index is out of range")
        direct = np.zeros((len(positions), detector.energy_bin_count), dtype=np.float64)
        corrected_source_rate = np.zeros_like(direct)
        for sample_index in range(source_samples.sample_count):
            activity_bq = source_samples.activity_bq[sample_index]
            if activity_bq == 0:
                continue
            isotope = isotopes[source_samples.isotope_index[sample_index]]
            origin = source_samples.positions_world_m[sample_index]
            origins = np.repeat(origin[None, :], len(positions), axis=0)
            distance_m = np.maximum(
                np.linalg.norm(positions - origins, axis=1), self._minimum_distance_m
            )
            geometric_factor = 1.0 / (4.0 * np.pi * distance_m**2)
            energies = np.asarray(
                [line.energy_keV for line in isotope.emission_lines], dtype=np.float64
            )
            yields = np.asarray(
                [line.photons_per_decay for line in isotope.emission_lines],
                dtype=np.float64,
            )
            transmission = self._transport.trace_transmission(origins, positions, energies)
            paths = self._transport.trace_path_lengths(origins, positions)
            buildup = self._buildup.factors(paths, energies)
            if transmission.shape != buildup.shape:
                raise ValueError("transport and buildup result dimensions do not match")
            primary_fluence = (
                activity_bq
                * yields[None, :]
                * geometric_factor[:, None]
                * transmission
            )
            corrected_fluence = primary_fluence * buildup
            for line_index, line in enumerate(isotope.emission_lines):
                effective_area_m2 = detector.effective_area_m2_at(line.energy_keV)
                if not np.any(effective_area_m2) or yields[line_index] == 0.0:
                    continue
                direct += (
                    primary_fluence[:, line_index, None]
                    * effective_area_m2[None, :]
                )
                corrected_source_rate += (
                    corrected_fluence[:, line_index, None]
                    * effective_area_m2[None, :]
                )
        background = np.zeros_like(corrected_source_rate)
        if include_background:
            background[:] = detector.background_cps_per_bin[None, :]
        total = corrected_source_rate + background
        if apply_dead_time and detector.dead_time_s > 0:
            live_fraction = 1.0 / (1.0 + detector.dead_time_s * total.sum(axis=1))
            total *= live_fraction[:, None]
            direct *= live_fraction[:, None]
            background *= live_fraction[:, None]
        return SampleCountRatePrediction(
            total,
            direct,
            background,
            self._buildup.model_name,
        )

    def build_transfer_matrix(
        self,
        detector_positions_world_m: FloatArray,
        basis_samples: SourceSampleBatch,
        isotopes: Sequence[IsotopeSpec],
        detector: DetectorSpec,
    ) -> FloatArray:
        """Build a pre-background, pre-dead-time count response matrix."""

        detector_count = len(np.asarray(detector_positions_world_m))
        matrix = np.zeros(
            (detector_count * detector.energy_bin_count, basis_samples.sample_count),
            dtype=np.float64,
        )
        unit_activity = np.zeros(basis_samples.sample_count, dtype=np.float64)
        for column in range(basis_samples.sample_count):
            unit_activity[column] = 1.0
            unit_basis = basis_samples.with_activity_bq(unit_activity)
            prediction = self.predict_count_rate(
                detector_positions_world_m,
                unit_basis,
                isotopes,
                detector,
                include_background=False,
                apply_dead_time=False,
            )
            matrix[:, column] = prediction.count_rate_cps_per_bin.reshape(-1)
            unit_activity[column] = 0.0
        return matrix
