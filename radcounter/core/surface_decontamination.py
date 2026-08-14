"""Surface-source activity and contact-based decontamination dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def effective_contact_exposure_s(
    dt_s: float,
    surface_speed_m_s: float,
    max_surface_speed_m_s: float,
) -> float:
    """Return one cell's canonical exposure for one verified contact tick."""

    if dt_s <= 0.0 or max_surface_speed_m_s <= 0.0 or surface_speed_m_s < 0.0:
        raise ValueError("contact exposure inputs must be positive and speed nonnegative")
    if surface_speed_m_s > max_surface_speed_m_s:
        return 0.0
    speed_factor = max(0.15, 1.0 - surface_speed_m_s / max_surface_speed_m_s)
    return dt_s * speed_factor


def irregular_deposition_field(cells_x: int = 48, cells_y: int = 28) -> np.ndarray:
    """Return the canonical deterministic ragged contamination activity field.

    This is the flat-surface version used by the August 6, 2026 reference
    renders.  The numerical lattice is rectangular, but inactive cells remain
    absent so the authored source consists of lobes, holes, and detached drops.
    """

    if cells_x < 8 or cells_y < 8:
        raise ValueError("irregular deposition fields require at least 8 x 8 cells")
    x = np.linspace(-1.0, 1.0, cells_x)
    y = np.linspace(-1.0, 1.0, cells_y)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    plume = (
        2.6 * np.exp(-((xx - 0.28) ** 2 / 0.12 + (yy + 0.16) ** 2 / 0.22))
        + 1.9 * np.exp(-((xx + 0.28) ** 2 / 0.30 + (yy - 0.08) ** 2 / 0.10))
        + 1.25 * np.exp(-((xx + 0.68) ** 2 / 0.045 + (yy + 0.30) ** 2 / 0.15))
        + 0.85 * np.exp(-((xx - 0.58) ** 2 / 0.055 + (yy - 0.48) ** 2 / 0.055))
    )
    roughness = (
        0.22 * np.sin(9.0 * xx + 2.2 * np.sin(5.0 * yy))
        + 0.18 * np.cos(11.0 * yy - 1.8 * xx)
        + 0.11 * np.sin(19.0 * (xx + yy))
    )
    contaminated = plume + roughness > 0.70
    clean_holes = (((xx - 0.05) / 0.16) ** 2 + ((yy + 0.02) / 0.12) ** 2 < 1.0) | (
        ((xx + 0.48) / 0.09) ** 2 + ((yy - 0.03) / 0.08) ** 2 < 1.0
    )
    contaminated &= ~clean_holes
    detached_droplets = (
        ((xx + 0.78) ** 2 + (yy - 0.58) ** 2 < 0.018)
        | ((xx - 0.72) ** 2 + (yy + 0.64) ** 2 < 0.012)
        | ((xx + 0.12) ** 2 + (yy - 0.72) ** 2 < 0.007)
    )
    contaminated |= detached_droplets
    local_activity = 92_000.0 * np.maximum(0.22, plume + 0.65 * roughness)
    return np.where(contaminated, local_activity, 0.0).reshape(-1)


@dataclass(frozen=True)
class DecontaminationTool:
    length_m: float
    width_m: float
    rate_constant_s_inv: float
    max_contact_distance_m: float
    max_surface_speed_m_s: float

    def __post_init__(self) -> None:
        values = (
            self.length_m,
            self.width_m,
            self.rate_constant_s_inv,
            self.max_contact_distance_m,
            self.max_surface_speed_m_s,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("decontamination tool parameters must be positive")


@dataclass(frozen=True)
class DecontaminationStep:
    contacted_cells: tuple[int, ...]
    activity_before_bq: float
    activity_after_bq: float
    removed_activity_bq: float
    effective_dwell_s: float


class SurfaceSourceGrid:
    """A surface-aligned numerical lattice of independent activity cells."""

    def __init__(
        self,
        *,
        cells_x: int,
        cells_y: int,
        size_x_m: float,
        size_y_m: float,
        center_world_m: tuple[float, float, float],
        activity_bq_per_cell: float | np.ndarray,
        efficiency_field: np.ndarray | None = None,
        surface_u_world: tuple[float, float, float] = (1.0, 0.0, 0.0),
        surface_v_world: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> None:
        if cells_x <= 0 or cells_y <= 0:
            raise ValueError("surface source dimensions must be positive")
        if size_x_m <= 0.0 or size_y_m <= 0.0:
            raise ValueError("surface source size must be positive")
        self.cells_x = cells_x
        self.cells_y = cells_y
        self.size_x_m = size_x_m
        self.size_y_m = size_y_m
        self.center_world_m = np.asarray(center_world_m, dtype=np.float64)
        self.cell_size_x_m = size_x_m / cells_x
        self.cell_size_y_m = size_y_m / cells_y

        surface_u = np.asarray(surface_u_world, dtype=np.float64)
        surface_v = np.asarray(surface_v_world, dtype=np.float64)
        if np.linalg.norm(surface_u) <= 1e-12 or np.linalg.norm(surface_v) <= 1e-12:
            raise ValueError("surface basis vectors cannot be zero")
        surface_u /= np.linalg.norm(surface_u)
        surface_v /= np.linalg.norm(surface_v)
        if abs(float(np.dot(surface_u, surface_v))) > 1e-6:
            raise ValueError("surface basis vectors must be orthogonal")
        surface_normal = np.cross(surface_u, surface_v)
        if np.linalg.norm(surface_normal) <= 1e-12:
            raise ValueError("surface basis vectors cannot be parallel")
        self.surface_u_world = surface_u
        self.surface_v_world = surface_v
        self.surface_normal_world = surface_normal / np.linalg.norm(surface_normal)

        x = (np.arange(cells_x, dtype=np.float64) + 0.5) * self.cell_size_x_m - size_x_m * 0.5
        y = (np.arange(cells_y, dtype=np.float64) + 0.5) * self.cell_size_y_m - size_y_m * 0.5
        xx, yy = np.meshgrid(x, y, indexing="xy")
        self.centers_surface_uv_m = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
        self.centers_world_m = (
            self.center_world_m[None, :]
            + self.centers_surface_uv_m[:, :1] * self.surface_u_world[None, :]
            + self.centers_surface_uv_m[:, 1:] * self.surface_v_world[None, :]
        )

        initial = np.asarray(activity_bq_per_cell, dtype=np.float64)
        if initial.ndim == 0:
            initial = np.full(xx.size, float(initial), dtype=np.float64)
        initial = initial.reshape(-1)
        if len(initial) != xx.size or np.any(initial < 0.0):
            raise ValueError("activity must be non-negative and match the grid")
        self.initial_activity_bq = initial.copy()
        self.activity_bq = initial.copy()
        self.cumulative_exposure_s = np.zeros(xx.size, dtype=np.float64)

        if efficiency_field is None:
            efficiency = np.ones(xx.size, dtype=np.float64)
        else:
            efficiency = np.asarray(efficiency_field, dtype=np.float64).reshape(-1)
        if len(efficiency) != xx.size or np.any(efficiency < 0.0):
            raise ValueError("efficiency field must be non-negative and match the grid")
        self.efficiency = efficiency

    @property
    def initial_total_activity_bq(self) -> float:
        return float(self.initial_activity_bq.sum())

    @property
    def total_activity_bq(self) -> float:
        return float(self.activity_bq.sum())

    @property
    def remaining_fraction(self) -> float:
        initial = self.initial_total_activity_bq
        return self.total_activity_bq / initial if initial > 0.0 else 0.0

    @property
    def removed_fraction(self) -> float:
        return 1.0 - self.remaining_fraction

    @property
    def treated_fraction(self) -> float:
        return float(np.count_nonzero(self.cumulative_exposure_s > 0.0) / len(self.activity_bq))

    def apply_tool(
        self,
        tool: DecontaminationTool,
        *,
        tool_center_world_m: tuple[float, float, float],
        tool_yaw_rad: float,
        surface_speed_m_s: float,
        dt_s: float,
    ) -> DecontaminationStep:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        center = np.asarray(tool_center_world_m, dtype=np.float64)
        height_error = abs(float(np.dot(center - self.center_world_m, self.surface_normal_world)))
        activity_before = self.total_activity_bq
        if (
            height_error > tool.max_contact_distance_m
            or surface_speed_m_s > tool.max_surface_speed_m_s
        ):
            return DecontaminationStep((), activity_before, activity_before, 0.0, 0.0)

        relative_world = self.centers_world_m - center
        relative_u = relative_world @ self.surface_u_world
        relative_v = relative_world @ self.surface_v_world
        cosine = math.cos(tool_yaw_rad)
        sine = math.sin(tool_yaw_rad)
        local_x = cosine * relative_u + sine * relative_v
        local_y = -sine * relative_u + cosine * relative_v
        contacted = (np.abs(local_x) <= tool.length_m * 0.5) & (
            np.abs(local_y) <= tool.width_m * 0.5
        )
        indices = np.flatnonzero(contacted)
        if not len(indices):
            return DecontaminationStep((), activity_before, activity_before, 0.0, 0.0)

        effective_dwell = effective_contact_exposure_s(
            dt_s,
            surface_speed_m_s,
            tool.max_surface_speed_m_s,
        )
        self.cumulative_exposure_s[indices] += effective_dwell
        exponent = (
            -tool.rate_constant_s_inv
            * self.efficiency[indices]
            * self.cumulative_exposure_s[indices]
        )
        self.activity_bq[indices] = self.initial_activity_bq[indices] * np.exp(exponent)
        activity_after = self.total_activity_bq
        return DecontaminationStep(
            contacted_cells=tuple(int(index) for index in indices),
            activity_before_bq=activity_before,
            activity_after_bq=activity_after,
            removed_activity_bq=max(0.0, activity_before - activity_after),
            effective_dwell_s=effective_dwell,
        )

    def color_rgb(self) -> np.ndarray:
        """Return green-clean to yellow/red-contaminated colors for visualization."""

        fraction = np.divide(
            self.activity_bq,
            self.initial_activity_bq,
            out=np.zeros_like(self.activity_bq),
            where=self.initial_activity_bq > 0.0,
        )
        fraction = np.clip(fraction, 0.0, 1.0)
        clean = np.asarray([0.05, 0.48, 0.16])
        middle = np.asarray([0.95, 0.62, 0.04])
        hot = np.asarray([0.88, 0.035, 0.02])
        colors = np.empty((len(fraction), 3), dtype=np.float64)
        low = fraction <= 0.5
        low_weight = (fraction[low] * 2.0)[:, None]
        colors[low] = clean + low_weight * (middle - clean)
        high_weight = ((fraction[~low] - 0.5) * 2.0)[:, None]
        colors[~low] = middle + high_weight * (hot - middle)
        return colors
