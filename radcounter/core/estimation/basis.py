"""Candidate source bases independent of simulator truth."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class BasisKind(StrEnum):
    GRID_3D = "grid_3d"
    SURFACE_TRIANGLE = "surface_triangle"


@dataclass(frozen=True)
class CandidateBasis:
    """Public candidate geometry and graph connectivity."""

    basis_ids: tuple[str, ...]
    positions_world_m: FloatArray
    adjacency_edges: IntArray
    kind: BasisKind

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_world_m, dtype=np.float64)
        edges = np.asarray(self.adjacency_edges, dtype=np.int64)
        if positions.shape != (len(self.basis_ids), 3):
            raise ValueError("basis positions must match basis_ids and have three coordinates")
        if edges.size == 0:
            edges = np.empty((0, 2), dtype=np.int64)
        if edges.ndim != 2 or edges.shape[1:] != (2,):
            raise ValueError("adjacency_edges must have shape (E,2)")
        if np.any(edges < 0) or np.any(edges >= len(self.basis_ids)):
            raise ValueError("basis adjacency index out of range")
        if len(set(self.basis_ids)) != len(self.basis_ids):
            raise ValueError("basis_ids must be unique")
        object.__setattr__(self, "positions_world_m", positions)
        object.__setattr__(self, "adjacency_edges", edges)

    @property
    def size(self) -> int:
        """Return number of source candidates."""

        return len(self.basis_ids)

    @classmethod
    def regular_grid(
        cls,
        bounds_xyz_m: tuple[float, float, float, float, float, float],
        *,
        spacing_m: float,
        id_prefix: str = "grid",
    ) -> CandidateBasis:
        """Build an inclusive 6-neighbour 3-D grid."""

        x_min, x_max, y_min, y_max, z_min, z_max = bounds_xyz_m
        if spacing_m <= 0 or min(x_max - x_min, y_max - y_min, z_max - z_min) < 0:
            raise ValueError("grid bounds and spacing are invalid")
        axes = tuple(
            np.arange(lower, upper + 0.5 * spacing_m, spacing_m)
            for lower, upper in ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        )
        mesh = np.meshgrid(*axes, indexing="ij")
        positions = np.column_stack(tuple(component.ravel() for component in mesh))
        shape = tuple(len(axis) for axis in axes)
        edges: list[tuple[int, int]] = []
        for index in np.ndindex(shape):
            current = int(np.ravel_multi_index(index, shape))
            for dimension in range(3):
                neighbour = list(index)
                neighbour[dimension] += 1
                if neighbour[dimension] < shape[dimension]:
                    edges.append((current, int(np.ravel_multi_index(tuple(neighbour), shape))))
        ids = tuple(f"{id_prefix}-{index}" for index in range(len(positions)))
        return cls(ids, positions, np.asarray(edges, dtype=np.int64), BasisKind.GRID_3D)

    @classmethod
    def surface(
        cls,
        positions_world_m: FloatArray,
        adjacency_edges: IntArray,
        *,
        id_prefix: str = "surface",
    ) -> CandidateBasis:
        """Build a triangle-centroid surface graph."""

        positions = np.asarray(positions_world_m, dtype=np.float64)
        ids = tuple(f"{id_prefix}-{index}" for index in range(len(positions)))
        return cls(ids, positions, adjacency_edges, BasisKind.SURFACE_TRIANGLE)

    def connected_components(self, active_mask: NDArray[np.bool_]) -> tuple[IntArray, ...]:
        """Return graph components induced by active candidates."""

        active = np.asarray(active_mask, dtype=np.bool_)
        if active.shape != (self.size,):
            raise ValueError("active_mask must match candidate basis")
        neighbours: list[list[int]] = [[] for _ in range(self.size)]
        for left, right in self.adjacency_edges:
            neighbours[int(left)].append(int(right))
            neighbours[int(right)].append(int(left))
        visited = np.zeros(self.size, dtype=np.bool_)
        components: list[IntArray] = []
        for seed in np.flatnonzero(active):
            if visited[seed]:
                continue
            stack = [int(seed)]
            visited[seed] = True
            component: list[int] = []
            while stack:
                node = stack.pop()
                component.append(node)
                for neighbour in neighbours[node]:
                    if active[neighbour] and not visited[neighbour]:
                        visited[neighbour] = True
                        stack.append(neighbour)
            components.append(np.asarray(sorted(component), dtype=np.int64))
        return tuple(components)
