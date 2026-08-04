"""Surface-source activity and contact-based decontamination dynamics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


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
    """A rectangular surface source discretized into independent activity cells."""

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

        x = (np.arange(cells_x, dtype=np.float64) + 0.5) * self.cell_size_x_m - size_x_m * 0.5
        y = (np.arange(cells_y, dtype=np.float64) + 0.5) * self.cell_size_y_m - size_y_m * 0.5
        xx, yy = np.meshgrid(x, y, indexing="xy")
        self.centers_world_m = np.column_stack(
            (
                xx.reshape(-1) + self.center_world_m[0],
                yy.reshape(-1) + self.center_world_m[1],
                np.full(xx.size, self.center_world_m[2]),
            )
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
        height_error = abs(center[2] - self.center_world_m[2])
        activity_before = self.total_activity_bq
        if (
            height_error > tool.max_contact_distance_m
            or surface_speed_m_s > tool.max_surface_speed_m_s
        ):
            return DecontaminationStep((), activity_before, activity_before, 0.0, 0.0)

        relative = self.centers_world_m[:, :2] - center[:2]
        cosine = math.cos(tool_yaw_rad)
        sine = math.sin(tool_yaw_rad)
        local_x = cosine * relative[:, 0] + sine * relative[:, 1]
        local_y = -sine * relative[:, 0] + cosine * relative[:, 1]
        contacted = (np.abs(local_x) <= tool.length_m * 0.5) & (
            np.abs(local_y) <= tool.width_m * 0.5
        )
        indices = np.flatnonzero(contacted)
        if not len(indices):
            return DecontaminationStep((), activity_before, activity_before, 0.0, 0.0)

        speed_factor = max(0.15, 1.0 - surface_speed_m_s / tool.max_surface_speed_m_s)
        effective_dwell = dt_s * speed_factor
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
