"""Transfer-matrix keys, cache, and linear evaluation."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from radcounter.core.models.state import RevisionState

FloatArray = NDArray[np.float64]


def hash_array(value: NDArray[np.generic]) -> str:
    """Create a shape/dtype-sensitive SHA-256 for a contiguous array."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TransferMatrixKey:
    """Cache key deliberately excluding source activity revision."""

    detector_pose_hash: str
    basis_hash: str
    geometry_revision: int
    material_revision: int
    detector_revision: int
    energy_grid_hash: str

    @classmethod
    def from_arrays(
        cls,
        *,
        detector_poses: FloatArray,
        basis_positions: FloatArray,
        revision: RevisionState,
        energy_grid_keV: FloatArray,
    ) -> TransferMatrixKey:
        """Build a cache key using only response-changing revisions."""

        return cls(
            detector_pose_hash=hash_array(np.asarray(detector_poses)),
            basis_hash=hash_array(np.asarray(basis_positions)),
            geometry_revision=revision.geometry_revision,
            material_revision=revision.material_revision,
            detector_revision=revision.detector_revision,
            energy_grid_hash=hash_array(np.asarray(energy_grid_keV)),
        )


@dataclass(frozen=True)
class CacheStatistics:
    hits: int
    misses: int
    entries: int


class TransferMatrixCache:
    """Bounded LRU cache for immutable transfer matrices."""

    def __init__(self, maximum_entries: int = 8) -> None:
        if maximum_entries < 1:
            raise ValueError("maximum_entries must be positive")
        self._maximum_entries = maximum_entries
        self._entries: OrderedDict[TransferMatrixKey, FloatArray] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: TransferMatrixKey) -> FloatArray | None:
        """Return a read-only matrix and update LRU statistics."""

        matrix = self._entries.get(key)
        if matrix is None:
            self._misses += 1
            return None
        self._entries.move_to_end(key)
        self._hits += 1
        return matrix

    def put(self, key: TransferMatrixKey, matrix: FloatArray) -> None:
        """Store an immutable contiguous matrix."""

        value = np.ascontiguousarray(matrix, dtype=np.float64)
        if value.ndim != 2 or not np.all(np.isfinite(value)):
            raise ValueError("transfer matrix must be finite and two-dimensional")
        value.setflags(write=False)
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self._maximum_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        """Drop all cached matrices without resetting lifetime statistics."""

        self._entries.clear()

    @property
    def statistics(self) -> CacheStatistics:
        """Return cache counters."""

        return CacheStatistics(self._hits, self._misses, len(self._entries))


def evaluate_linear_counts(
    transfer_matrix: FloatArray,
    source_activity_bq: FloatArray,
    background_count_rate_cps: FloatArray | None = None,
) -> FloatArray:
    """Evaluate `H @ x + b` without any transport call."""

    matrix = np.asarray(transfer_matrix, dtype=np.float64)
    activity = np.asarray(source_activity_bq, dtype=np.float64)
    if matrix.ndim != 2 or activity.shape != (matrix.shape[1],) or np.any(activity < 0):
        raise ValueError("transfer matrix and activity dimensions are inconsistent")
    result = matrix @ activity
    if background_count_rate_cps is not None:
        background = np.asarray(background_count_rate_cps, dtype=np.float64)
        if background.shape != result.shape or np.any(background < 0):
            raise ValueError("background must match transfer-matrix rows")
        result = result + background
    return result
