"""Selective source-detector ray cache for local geometry updates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class AxisAlignedBounds:
    minimum_m: FloatArray
    maximum_m: FloatArray

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum_m, dtype=np.float64)
        maximum = np.asarray(self.maximum_m, dtype=np.float64)
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("axis-aligned bounds require two 3-D vectors")
        if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
            raise ValueError("axis-aligned bounds must be finite")
        if np.any(maximum < minimum):
            raise ValueError("axis-aligned maximum must not be below minimum")
        object.__setattr__(self, "minimum_m", minimum)
        object.__setattr__(self, "maximum_m", maximum)

    def expanded(self, padding_m: float) -> AxisAlignedBounds:
        if padding_m < 0.0:
            raise ValueError("bounds padding must be nonnegative")
        return AxisAlignedBounds(self.minimum_m - padding_m, self.maximum_m + padding_m)


@dataclass(frozen=True)
class TransferRayLayout:
    """Detector-major flattening of every detector-source segment."""

    detector_count: int
    source_count: int

    def __post_init__(self) -> None:
        if self.detector_count < 1 or self.source_count < 1:
            raise ValueError("transfer layout dimensions must be positive")

    @property
    def ray_count(self) -> int:
        return self.detector_count * self.source_count

    @classmethod
    def build_segments(
        cls,
        detector_positions_m: FloatArray,
        source_positions_m: FloatArray,
    ) -> tuple[TransferRayLayout, FloatArray, FloatArray]:
        detectors = np.asarray(detector_positions_m, dtype=np.float64)
        sources = np.asarray(source_positions_m, dtype=np.float64)
        if detectors.ndim != 2 or detectors.shape[1:] != (3,):
            raise ValueError("detector positions must have shape (D, 3)")
        if sources.ndim != 2 or sources.shape[1:] != (3,):
            raise ValueError("source positions must have shape (S, 3)")
        layout = cls(len(detectors), len(sources))
        origins = np.tile(sources, (len(detectors), 1))
        targets = np.repeat(detectors, len(sources), axis=0)
        return layout, origins, targets


class SelectiveRayResponseCache:
    """Patch only rays whose finite segments intersect changed geometry bounds."""

    def __init__(
        self,
        origins_m: FloatArray,
        targets_m: FloatArray,
        response: FloatArray,
        *,
        layout: TransferRayLayout | None = None,
        geometry_revision: int = 0,
    ) -> None:
        origins = np.asarray(origins_m, dtype=np.float64)
        targets = np.asarray(targets_m, dtype=np.float64)
        values = np.asarray(response, dtype=np.float64)
        if origins.ndim != 2 or origins.shape[1:] != (3,) or targets.shape != origins.shape:
            raise ValueError("ray origins and targets must have equal shape (R, 3)")
        if values.ndim < 1 or values.shape[0] != len(origins):
            raise ValueError("response first dimension must match the ray count")
        if not np.all(np.isfinite(origins)) or not np.all(np.isfinite(targets)):
            raise ValueError("ray endpoints must be finite")
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("ray response values must be finite and nonnegative")
        if layout is not None and layout.ray_count != len(origins):
            raise ValueError("transfer layout does not match the ray count")
        if geometry_revision < 0:
            raise ValueError("geometry revision must be nonnegative")
        self.origins_m = origins
        self.targets_m = targets
        self.response = values
        self.layout = layout
        self.geometry_revision = geometry_revision
        self.patch_count = 0
        self.patched_ray_count = 0

    def intersecting_ray_indices(
        self, bounds: AxisAlignedBounds, *, padding_m: float = 0.0
    ) -> IntArray:
        """Return finite segments intersecting an expanded AABB using slab tests."""

        expanded = bounds.expanded(padding_m)
        direction = self.targets_m - self.origins_m
        t_enter = np.zeros(len(direction), dtype=np.float64)
        t_exit = np.ones(len(direction), dtype=np.float64)
        intersects = np.ones(len(direction), dtype=np.bool_)
        epsilon = np.finfo(np.float64).eps * 32.0
        for axis in range(3):
            origin = self.origins_m[:, axis]
            delta = direction[:, axis]
            parallel = np.abs(delta) <= epsilon
            intersects &= ~(
                parallel
                & ((origin < expanded.minimum_m[axis]) | (origin > expanded.maximum_m[axis]))
            )
            nonparallel = ~parallel
            inverse = np.zeros_like(delta)
            inverse[nonparallel] = 1.0 / delta[nonparallel]
            first = (expanded.minimum_m[axis] - origin) * inverse
            second = (expanded.maximum_m[axis] - origin) * inverse
            near = np.minimum(first, second)
            far = np.maximum(first, second)
            t_enter[nonparallel] = np.maximum(t_enter[nonparallel], near[nonparallel])
            t_exit[nonparallel] = np.minimum(t_exit[nonparallel], far[nonparallel])
        intersects &= t_exit >= t_enter
        return np.flatnonzero(intersects).astype(np.int64)

    def patch_bounds(
        self,
        bounds: AxisAlignedBounds,
        recompute: Callable[[FloatArray, FloatArray], FloatArray],
        *,
        geometry_revision: int,
        padding_m: float = 0.0,
    ) -> IntArray:
        """Recompute affected rows and return their ray indexes."""

        if geometry_revision <= self.geometry_revision:
            raise ValueError("patched geometry revision must increase monotonically")
        indexes = self.intersecting_ray_indices(bounds, padding_m=padding_m)
        if len(indexes):
            replacement = np.asarray(
                recompute(self.origins_m[indexes], self.targets_m[indexes]),
                dtype=np.float64,
            )
            if replacement.shape != self.response[indexes].shape:
                raise ValueError("recomputed response shape does not match affected rays")
            if np.any(replacement < 0.0) or not np.all(np.isfinite(replacement)):
                raise ValueError("recomputed response must be finite and nonnegative")
            self.response[indexes] = replacement
        self.geometry_revision = geometry_revision
        self.patch_count += 1
        self.patched_ray_count += len(indexes)
        return indexes

    def replace_all(
        self,
        recompute: Callable[[FloatArray, FloatArray], FloatArray],
        *,
        geometry_revision: int,
    ) -> None:
        if geometry_revision <= self.geometry_revision:
            raise ValueError("replacement geometry revision must increase monotonically")
        replacement = np.asarray(recompute(self.origins_m, self.targets_m), dtype=np.float64)
        if replacement.shape != self.response.shape:
            raise ValueError("replacement response shape does not match the cache")
        if np.any(replacement < 0.0) or not np.all(np.isfinite(replacement)):
            raise ValueError("replacement response must be finite and nonnegative")
        self.response[:] = replacement
        self.geometry_revision = geometry_revision
        self.patch_count += 1
        self.patched_ray_count += len(self.origins_m)

    def expected_detector_response(self, source_activity_bq: FloatArray) -> FloatArray:
        """Apply cached transfer values without any ray tracing."""

        if self.layout is None:
            raise RuntimeError("a transfer layout is required for activity projection")
        activity = np.asarray(source_activity_bq, dtype=np.float64)
        if activity.shape != (self.layout.source_count,) or np.any(activity < 0.0):
            raise ValueError("source activity does not match the transfer layout")
        trailing_shape = self.response.shape[1:]
        matrix = self.response.reshape(
            self.layout.detector_count,
            self.layout.source_count,
            *trailing_shape,
        )
        return np.tensordot(matrix, activity, axes=([1], [0]))
