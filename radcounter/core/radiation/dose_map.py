"""Chunked planar/3-D dose-map evaluation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from radcounter.core.models.radiation import DetectorSpec, IsotopeSpec
from radcounter.core.models.state import RevisionState
from radcounter.core.radiation.sampled_forward import SampledRadiationForwardModel
from radcounter.core.radiation.sampling import SourceSampleBatch

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EvaluationGrid:
    """World evaluation points and occupancy mask."""

    points_world_m: FloatArray
    shape: tuple[int, ...]
    occupied: NDArray[np.bool_]


@dataclass(frozen=True)
class DoseMap:
    """Dose-rate result associated with a scene revision."""

    points_world_m: FloatArray
    dose_rate_sv_h: FloatArray
    standard_deviation_sv_h: FloatArray | None
    revision: RevisionState
    grid_shape: tuple[int, ...]


class DoseMapEvaluator:
    """Evaluate source samples in bounded chunks."""

    def __init__(self, forward_model: SampledRadiationForwardModel) -> None:
        self._forward_model = forward_model

    @staticmethod
    def create_planar_grid(
        bounds_xy_m: tuple[float, float, float, float],
        *,
        z_m: float,
        resolution_m: float,
    ) -> EvaluationGrid:
        """Create an inclusive XY grid at fixed height."""

        x_min, x_max, y_min, y_max = bounds_xy_m
        if x_max < x_min or y_max < y_min or resolution_m <= 0:
            raise ValueError("grid bounds/resolution are invalid")
        x = np.arange(x_min, x_max + 0.5 * resolution_m, resolution_m)
        y = np.arange(y_min, y_max + 0.5 * resolution_m, resolution_m)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        points = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, z_m)))
        return EvaluationGrid(points, xx.shape, np.zeros(len(points), dtype=np.bool_))

    @staticmethod
    def create_3d_grid(
        bounds_xyz_m: tuple[float, float, float, float, float, float],
        *,
        resolution_m: float,
    ) -> EvaluationGrid:
        """Create an inclusive XYZ grid."""

        x_min, x_max, y_min, y_max, z_min, z_max = bounds_xyz_m
        if min(x_max - x_min, y_max - y_min, z_max - z_min) < 0 or resolution_m <= 0:
            raise ValueError("grid bounds/resolution are invalid")
        axes = [
            np.arange(lower, upper + 0.5 * resolution_m, resolution_m)
            for lower, upper in ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        ]
        xx, yy, zz = np.meshgrid(*axes, indexing="ij")
        points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        return EvaluationGrid(points, xx.shape, np.zeros(len(points), dtype=np.bool_))

    @staticmethod
    def mask_occupied_cells(
        grid: EvaluationGrid, is_occupied: Callable[[FloatArray], NDArray[np.bool_]]
    ) -> EvaluationGrid:
        """Apply a collision-scene occupancy callback."""

        occupied = np.asarray(is_occupied(grid.points_world_m), dtype=np.bool_)
        if occupied.shape != (len(grid.points_world_m),):
            raise ValueError("occupancy callback returned the wrong shape")
        return EvaluationGrid(grid.points_world_m, grid.shape, occupied)

    def evaluate(
        self,
        grid: EvaluationGrid,
        source_samples: SourceSampleBatch,
        isotopes: Sequence[IsotopeSpec],
        detector: DetectorSpec,
        revision: RevisionState,
        *,
        source_strength_covariance: FloatArray | None = None,
        chunk_size: int = 4096,
    ) -> DoseMap:
        """Evaluate dose while excluding occupied cells and bounding memory."""

        if detector.dose_conversion_sv_h_per_cps is None:
            raise ValueError("dose map requires detector dose conversion factors")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        dose = np.full(len(grid.points_world_m), np.nan, dtype=np.float64)
        standard_deviation = None
        if source_strength_covariance is not None:
            covariance = np.asarray(source_strength_covariance, dtype=np.float64)
            expected_shape = (source_samples.sample_count, source_samples.sample_count)
            if covariance.shape != expected_shape:
                raise ValueError("source covariance does not match source samples")
            standard_deviation = np.full(len(grid.points_world_m), np.nan, dtype=np.float64)
        active_indexes = np.flatnonzero(~grid.occupied)
        conversion = detector.dose_conversion_sv_h_per_cps
        for start in range(0, len(active_indexes), chunk_size):
            indexes = active_indexes[start : start + chunk_size]
            points = grid.points_world_m[indexes]
            prediction = self._forward_model.predict_count_rate(
                points, source_samples, isotopes, detector
            )
            dose[indexes] = prediction.count_rate_cps_per_bin @ conversion
            if standard_deviation is not None:
                matrix = self._forward_model.build_transfer_matrix(
                    points, source_samples, isotopes, detector
                )
                matrix = matrix.reshape(len(points), detector.energy_bin_count, -1)
                dose_response = np.einsum("dbs,b->ds", matrix, conversion)
                variances = np.einsum(
                    "ds,st,dt->d", dose_response, covariance, dose_response, optimize=True
                )
                standard_deviation[indexes] = np.sqrt(np.maximum(variances, 0.0))
        return DoseMap(
            grid.points_world_m,
            dose,
            standard_deviation,
            revision.copy(),
            grid.shape,
        )
