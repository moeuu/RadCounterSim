"""Coordinate normalization and portable triangle construction."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from radcounter.core.environment.models import (
    AxisDirection,
    CoordinateDefaults,
    EnvironmentImportConfig,
    Handedness,
    LengthUnit,
)


def pose_matrix(
    xyz: tuple[float, float, float] | np.ndarray,
    rpy_rad: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_euler("xyz", np.asarray(rpy_rad, dtype=np.float64)).as_matrix()
    matrix[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return matrix


def apply_transform(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    vertices_array = np.asarray(vertices, dtype=np.float64)
    homogeneous = np.column_stack((vertices_array, np.ones(len(vertices_array))))
    return np.ascontiguousarray((homogeneous @ np.asarray(matrix, dtype=np.float64).T)[:, :3])


def _axis_vector(axis: AxisDirection) -> np.ndarray:
    return {
        AxisDirection.POS_X: np.array([1.0, 0.0, 0.0]),
        AxisDirection.NEG_X: np.array([-1.0, 0.0, 0.0]),
        AxisDirection.POS_Y: np.array([0.0, 1.0, 0.0]),
        AxisDirection.NEG_Y: np.array([0.0, -1.0, 0.0]),
        AxisDirection.POS_Z: np.array([0.0, 0.0, 1.0]),
        AxisDirection.NEG_Z: np.array([0.0, 0.0, -1.0]),
    }[axis]


def normalization_matrix(
    config: EnvironmentImportConfig,
    defaults: CoordinateDefaults,
) -> np.ndarray:
    """Return source coordinates to metre, right-handed, Z-up world coordinates."""

    coordinate = config.coordinate_system
    up_axis = defaults.up_axis if coordinate.up_axis == AxisDirection.AUTO else coordinate.up_axis
    forward_axis = (
        defaults.forward_axis
        if coordinate.forward_axis == AxisDirection.AUTO
        else coordinate.forward_axis
    )
    handedness = (
        defaults.handedness
        if coordinate.handedness == Handedness.AUTO
        else coordinate.handedness
    )
    unit_scale = (
        defaults.unit_scale_m
        if coordinate.units == LengthUnit.AUTO
        else coordinate.units.metres
    )
    up = _axis_vector(up_axis)
    forward = _axis_vector(forward_axis)
    if not math.isclose(float(np.dot(up, forward)), 0.0, abs_tol=1e-9):
        raise ValueError("environment up and forward axes must be orthogonal")
    left = np.cross(up, forward)
    if handedness == Handedness.LEFT:
        left *= -1.0
    source_basis = np.column_stack((forward, left, up))
    axis_change = source_basis.T
    local = np.eye(4, dtype=np.float64)
    local[:3, :3] = axis_change * (unit_scale * config.scale)
    world = pose_matrix(
        config.translation_world_m,
        np.radians(np.asarray(config.rotation_world_rpy_deg, dtype=np.float64)),
    )
    return world @ local


def voxel_surface_mesh(
    points: np.ndarray,
    voxel_size: float,
    max_voxels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert occupied point-cloud cells to a watertight boundary mesh."""

    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] < 3:
        raise ValueError("point cloud must contain at least x, y, z columns")
    point_array = point_array[:, :3]
    point_array = point_array[np.isfinite(point_array).all(axis=1)]
    if len(point_array) == 0:
        raise ValueError("point cloud contains no finite points")
    cells = np.unique(np.floor(point_array / voxel_size).astype(np.int64), axis=0)
    if len(cells) > max_voxels:
        raise ValueError(
            f"point cloud creates {len(cells)} voxels; maximum is {max_voxels}. "
            "Increase point_cloud_voxel_size_m or max_point_cloud_voxels."
        )
    occupied = {tuple(cell) for cell in cells}
    directions = (
        ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
        ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
        ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
        ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
        ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
        ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
    )
    vertices: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    for cell in occupied:
        for direction, corners in directions:
            neighbour = tuple(cell[index] + direction[index] for index in range(3))
            if neighbour in occupied:
                continue
            start = len(vertices)
            vertices.extend(
                tuple((cell[index] + corner[index]) * voxel_size for index in range(3))
                for corner in corners
            )
            triangles.extend(((start, start + 1, start + 2), (start, start + 2, start + 3)))
    return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.uint32)
