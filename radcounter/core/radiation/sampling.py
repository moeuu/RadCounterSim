"""Point, surface, and volume source sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SourceSampleBatch:
    """World-space quadrature samples with activity-conserving weights."""

    positions_world_m: FloatArray
    activity_bq: FloatArray
    isotope_index: IntArray
    source_id_index: IntArray
    triangle_index: IntArray

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_world_m, dtype=np.float64)
        activity = np.asarray(self.activity_bq, dtype=np.float64)
        isotope = np.asarray(self.isotope_index, dtype=np.int64)
        source = np.asarray(self.source_id_index, dtype=np.int64)
        triangle = np.asarray(self.triangle_index, dtype=np.int64)
        sample_count = len(positions)
        if positions.shape != (sample_count, 3):
            raise ValueError("positions_world_m must have shape (S, 3)")
        for name, array in (
            ("activity_bq", activity),
            ("isotope_index", isotope),
            ("source_id_index", source),
            ("triangle_index", triangle),
        ):
            if array.shape != (sample_count,):
                raise ValueError(f"{name} must have shape (S,)")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(activity)):
            raise ValueError("sample position and activity must be finite")
        if np.any(activity < 0) or np.any(isotope < 0) or np.any(source < 0):
            raise ValueError("activity and source/isotope indexes must be nonnegative")
        object.__setattr__(self, "positions_world_m", positions)
        object.__setattr__(self, "activity_bq", activity)
        object.__setattr__(self, "isotope_index", isotope)
        object.__setattr__(self, "source_id_index", source)
        object.__setattr__(self, "triangle_index", triangle)

    @property
    def sample_count(self) -> int:
        """Return the number of quadrature samples."""

        return len(self.activity_bq)

    @property
    def total_activity_bq(self) -> float:
        """Return activity represented by all samples."""

        return float(self.activity_bq.sum())

    def with_activity_bq(self, activity_bq: FloatArray) -> SourceSampleBatch:
        """Return the same basis with a new activity vector."""

        return SourceSampleBatch(
            self.positions_world_m,
            activity_bq,
            self.isotope_index,
            self.source_id_index,
            self.triangle_index,
        )

    def transformed(self, world_transform: FloatArray) -> SourceSampleBatch:
        """Apply one homogeneous transform while retaining sample weights."""

        transform = np.asarray(world_transform, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("world_transform must be a finite 4x4 matrix")
        homogeneous = np.column_stack(
            (self.positions_world_m, np.ones(self.sample_count, dtype=np.float64))
        )
        positions = (homogeneous @ transform.T)[:, :3]
        return SourceSampleBatch(
            positions,
            self.activity_bq,
            self.isotope_index,
            self.source_id_index,
            self.triangle_index,
        )


def point_sample_batch(
    positions_world_m: FloatArray,
    activity_bq: FloatArray,
    isotope_index: IntArray,
    source_id_index: IntArray,
) -> SourceSampleBatch:
    """Create point-source samples with triangle index `-1`."""

    count = len(np.asarray(activity_bq))
    return SourceSampleBatch(
        positions_world_m,
        activity_bq,
        isotope_index,
        source_id_index,
        np.full(count, -1, dtype=np.int64),
    )


def _triangle_sample_count(
    vertices: FloatArray,
    detector_positions_world_m: FloatArray,
    maximum_samples_per_triangle: int,
) -> int:
    centroid = vertices.mean(axis=0)
    edge_scale_m = max(
        np.linalg.norm(vertices[1] - vertices[0]),
        np.linalg.norm(vertices[2] - vertices[1]),
        np.linalg.norm(vertices[0] - vertices[2]),
    )
    distance_m = float(
        np.min(np.linalg.norm(detector_positions_world_m - centroid[None, :], axis=1))
    )
    ratio = edge_scale_m / max(distance_m, 1e-6)
    return int(np.clip(np.ceil(1.0 + 4.0 * ratio), 1, maximum_samples_per_triangle))


def sample_surface_triangles(
    *,
    vertices_world_m: FloatArray,
    triangles: IntArray,
    activity_bq_per_triangle: FloatArray,
    isotope_index: int,
    source_id_index: int,
    mode: Literal["centroid", "stratified", "adaptive"] = "centroid",
    samples_per_triangle: int = 4,
    detector_positions_world_m: FloatArray | None = None,
    maximum_samples_per_triangle: int = 16,
    rng: np.random.Generator | None = None,
) -> SourceSampleBatch:
    """Sample a triangulated surface without changing total activity."""

    vertices = np.asarray(vertices_world_m, dtype=np.float64)
    faces = np.asarray(triangles, dtype=np.int64)
    activity = np.asarray(activity_bq_per_triangle, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise ValueError("vertices_world_m must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or activity.shape != (len(faces),):
        raise ValueError("triangles must be (M,3) and activity must be (M,)")
    if np.any(faces < 0) or np.any(faces >= len(vertices)) or np.any(activity < 0):
        raise ValueError("surface indexes/activity are invalid")
    if mode == "stratified" and samples_per_triangle < 1:
        raise ValueError("samples_per_triangle must be positive")
    detectors = None
    if mode == "adaptive":
        if detector_positions_world_m is None or maximum_samples_per_triangle < 1:
            raise ValueError("adaptive mode requires detector positions and a positive cap")
        detectors = np.asarray(detector_positions_world_m, dtype=np.float64)
        if detectors.ndim != 2 or detectors.shape[1:] != (3,) or len(detectors) == 0:
            raise ValueError("detector_positions_world_m must have shape (D,3)")
    generator = rng if rng is not None else np.random.default_rng(0)
    positions: list[FloatArray] = []
    weights: list[float] = []
    triangle_ids: list[int] = []
    for triangle_id, face in enumerate(faces):
        triangle_vertices = vertices[face]
        if mode == "centroid":
            count = 1
        elif mode == "stratified":
            count = samples_per_triangle
        else:
            assert detectors is not None
            count = _triangle_sample_count(
                triangle_vertices, detectors, maximum_samples_per_triangle
            )
        if count == 1:
            triangle_positions = triangle_vertices.mean(axis=0)[None, :]
        else:
            random_uv = generator.random((count, 2))
            root_u = np.sqrt(random_uv[:, 0])
            barycentric = np.column_stack(
                (1.0 - root_u, root_u * (1.0 - random_uv[:, 1]), root_u * random_uv[:, 1])
            )
            triangle_positions = barycentric @ triangle_vertices
        positions.extend(triangle_positions)
        weights.extend([float(activity[triangle_id]) / count] * count)
        triangle_ids.extend([triangle_id] * count)
    sample_count = len(weights)
    return SourceSampleBatch(
        np.asarray(positions, dtype=np.float64).reshape(sample_count, 3),
        np.asarray(weights, dtype=np.float64),
        np.full(sample_count, isotope_index, dtype=np.int64),
        np.full(sample_count, source_id_index, dtype=np.int64),
        np.asarray(triangle_ids, dtype=np.int64),
    )


def volume_sample_batch(
    *,
    voxel_centers_world_m: FloatArray,
    activity_bq_per_voxel: FloatArray,
    isotope_index: int,
    source_id_index: int,
) -> SourceSampleBatch:
    """Convert a voxel source into center-point quadrature samples."""

    count = len(np.asarray(activity_bq_per_voxel))
    return SourceSampleBatch(
        voxel_centers_world_m,
        activity_bq_per_voxel,
        np.full(count, isotope_index, dtype=np.int64),
        np.full(count, source_id_index, dtype=np.int64),
        np.full(count, -1, dtype=np.int64),
    )
