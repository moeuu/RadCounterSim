"""Transport protocol and deterministic analytic fallback."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from radcounter.core.radiation.materials import MaterialTable

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AnalyticSlab:
    """Infinite planar slab used for analytic validation."""

    slab_id: str
    material_id: str
    point_on_midplane_m: FloatArray
    normal: FloatArray
    thickness_m: float

    def __post_init__(self) -> None:
        point = np.asarray(self.point_on_midplane_m, dtype=np.float64)
        normal = np.asarray(self.normal, dtype=np.float64)
        norm = np.linalg.norm(normal)
        if point.shape != (3,) or normal.shape != (3,) or norm <= 0 or self.thickness_m <= 0:
            raise ValueError("slab requires 3-D point/normal and positive thickness")
        object.__setattr__(self, "point_on_midplane_m", point)
        object.__setattr__(self, "normal", normal / norm)


@dataclass(frozen=True)
class PathLengthBatch:
    """Per-material path lengths for a ray batch."""

    material_ids: tuple[str, ...]
    lengths_m: FloatArray
    error_flags: NDArray[np.bool_]


class RayTransportBackend(Protocol):
    """Batch finite-segment transport contract implemented by all backends."""

    def build_scene(self, meshes: Sequence[object], material_table: MaterialTable) -> None: ...

    def commit_updates(self) -> int: ...

    def trace_path_lengths(
        self, origins_m: FloatArray, targets_m: FloatArray
    ) -> PathLengthBatch: ...

    def trace_transmission(
        self, origins_m: FloatArray, targets_m: FloatArray, energies_keV: FloatArray
    ) -> FloatArray: ...


class AnalyticTransportBackend:
    """Free-space or infinite-slab backend for CI and analytic checks."""

    def __init__(self) -> None:
        self._slabs: tuple[AnalyticSlab, ...] = ()
        self._materials = MaterialTable(())
        self._revision = 0

    def build_scene(self, meshes: Sequence[object], material_table: MaterialTable) -> None:
        """Accept only explicit analytic slabs; never pretend to trace meshes."""

        if any(not isinstance(item, AnalyticSlab) for item in meshes):
            raise TypeError("AnalyticTransportBackend accepts AnalyticSlab objects only")
        self._slabs = tuple(meshes)  # type: ignore[assignment]
        self._materials = material_table
        self._revision += 1

    def commit_updates(self) -> int:
        """Return the current analytic scene revision."""

        return self._revision

    @staticmethod
    def _validated_rays(
        origins_m: FloatArray, targets_m: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        origins = np.asarray(origins_m, dtype=np.float64)
        targets = np.asarray(targets_m, dtype=np.float64)
        if origins.ndim != 2 or origins.shape[1:] != (3,) or origins.shape != targets.shape:
            raise ValueError("origins_m and targets_m must both have shape (R, 3)")
        if not np.all(np.isfinite(origins)) or not np.all(np.isfinite(targets)):
            raise ValueError("ray endpoints must be finite")
        return origins, targets

    def trace_path_lengths(self, origins_m: FloatArray, targets_m: FloatArray) -> PathLengthBatch:
        """Compute exact path lengths through configured infinite slabs."""

        origins, targets = self._validated_rays(origins_m, targets_m)
        material_ids = tuple(dict.fromkeys(slab.material_id for slab in self._slabs))
        lengths = np.zeros((len(origins), len(material_ids)), dtype=np.float64)
        material_index = {material_id: index for index, material_id in enumerate(material_ids)}
        segments = targets - origins
        segment_lengths = np.linalg.norm(segments, axis=1)
        directions = np.divide(
            segments,
            segment_lengths[:, None],
            out=np.zeros_like(segments),
            where=segment_lengths[:, None] > 0,
        )
        for slab in self._slabs:
            signed_origin = (origins - slab.point_on_midplane_m) @ slab.normal
            signed_target = (targets - slab.point_on_midplane_m) @ slab.normal
            half = slab.thickness_m / 2.0
            crosses = (np.minimum(signed_origin, signed_target) <= half) & (
                np.maximum(signed_origin, signed_target) >= -half
            )
            cosine = np.abs(directions @ slab.normal)
            effective = np.divide(
                slab.thickness_m,
                cosine,
                out=np.zeros_like(cosine),
                where=cosine > 1e-12,
            )
            effective = np.minimum(effective, segment_lengths)
            lengths[:, material_index[slab.material_id]] += np.where(crosses, effective, 0.0)
        return PathLengthBatch(material_ids, lengths, np.zeros(len(origins), dtype=np.bool_))

    def trace_transmission(
        self, origins_m: FloatArray, targets_m: FloatArray, energies_keV: FloatArray
    ) -> FloatArray:
        """Return `exp(-sum(mu*length))` for each ray and energy."""

        energies = np.asarray(energies_keV, dtype=np.float64)
        if energies.ndim != 1 or np.any(energies <= 0):
            raise ValueError("energies_keV must be a positive 1-D array")
        batch = self.trace_path_lengths(origins_m, targets_m)
        exponent = np.zeros((batch.lengths_m.shape[0], len(energies)), dtype=np.float64)
        for index, material_id in enumerate(batch.material_ids):
            exponent += (
                batch.lengths_m[:, index, None]
                * self._materials.attenuation_m_inv(material_id, energies)[None, :]
            )
        return np.exp(-np.minimum(exponent, 700.0))
