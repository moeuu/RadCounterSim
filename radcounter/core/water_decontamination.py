"""Pressure-water decontamination with recovery, runoff, and redeposition."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .surface_decontamination import SurfaceSourceGrid
from .treatment import WaterJetTreatment

FloatArray = np.ndarray
IntArray = np.ndarray


@dataclass(frozen=True)
class WaterJetSpec:
    flow_rate_l_min: float
    pressure_mpa: float
    reference_pressure_mpa: float
    spray_cone_angle_deg: float
    minimum_footprint_radius_m: float
    min_standoff_m: float
    max_standoff_m: float
    max_incidence_angle_deg: float
    max_surface_speed_m_s: float
    water_recovery_fraction: float
    surface_water_retention_fraction: float = 0.08
    require_wastewater_collection: bool = True

    def __post_init__(self) -> None:
        positive = (
            self.flow_rate_l_min,
            self.pressure_mpa,
            self.reference_pressure_mpa,
            self.minimum_footprint_radius_m,
            self.min_standoff_m,
            self.max_standoff_m,
            self.max_surface_speed_m_s,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("water-jet physical parameters must be positive")
        if self.max_standoff_m < self.min_standoff_m:
            raise ValueError("max_standoff_m must be at least min_standoff_m")
        if not 0.0 < self.spray_cone_angle_deg < 180.0:
            raise ValueError("spray_cone_angle_deg must be in (0, 180)")
        if not 0.0 <= self.max_incidence_angle_deg < 90.0:
            raise ValueError("max_incidence_angle_deg must be in [0, 90)")
        fractions = (
            self.water_recovery_fraction,
            self.surface_water_retention_fraction,
        )
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("water recovery and activity fractions must be in [0, 1]")
        if self.water_recovery_fraction + self.surface_water_retention_fraction > 1.0:
            raise ValueError("recovered and retained water fractions cannot exceed one")


@dataclass
class WaterDecontaminationState:
    supply_remaining_l: float
    wastewater_capacity_l: float
    wastewater_volume_l: float = 0.0
    applied_water_l: float = 0.0
    recovered_water_l: float = 0.0
    retained_surface_water_l: float = 0.0
    discharged_water_l: float = 0.0
    captured_activity_bq: float = 0.0
    discharged_activity_bq: float = 0.0
    redeposited_activity_bq: float = 0.0
    surface_water_l_by_cell: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.supply_remaining_l < 0.0 or self.wastewater_capacity_l < 0.0:
            raise ValueError("water supply and wastewater capacity must be non-negative")
        if self.wastewater_volume_l > self.wastewater_capacity_l:
            raise ValueError("initial wastewater exceeds tank capacity")

    @property
    def wastewater_capacity_remaining_l(self) -> float:
        return max(0.0, self.wastewater_capacity_l - self.wastewater_volume_l)


@dataclass(frozen=True)
class WaterDecontaminationStep:
    contacted_cells: tuple[int, ...]
    impact_world_m: tuple[float, float, float] | None
    standoff_m: float | None
    incidence_angle_deg: float | None
    applied_water_l: float
    recovered_water_l: float
    removed_activity_bq: float
    captured_activity_bq: float
    redeposited_activity_bq: float
    discharged_activity_bq: float
    blocked_reason: str | None = None


class WaterSurfaceDecontaminator:
    """Applies a finite water cone to a surface-source activity grid."""

    def __init__(
        self,
        grid: SurfaceSourceGrid,
        state: WaterDecontaminationState,
        treatment: WaterJetTreatment,
        *,
        washability: np.ndarray | None = None,
        surface_normal_world: tuple[float, float, float] | None = None,
        runoff_direction_world_xy: tuple[float, float] = (1.0, 0.0),
    ) -> None:
        self.grid = grid
        self.state = state
        self.treatment = treatment
        self.reference_activity_bq = (
            grid.total_activity_bq + state.captured_activity_bq + state.discharged_activity_bq
        )
        normal = np.asarray(
            grid.surface_normal_world if surface_normal_world is None else surface_normal_world,
            dtype=np.float64,
        )
        if np.linalg.norm(normal) <= 1e-12:
            raise ValueError("surface normal cannot be zero")
        self.surface_normal = normal / np.linalg.norm(normal)
        if abs(float(np.dot(self.surface_normal, grid.surface_normal_world))) < 1.0 - 1e-6:
            raise ValueError("surface normal must align with the source grid plane")
        runoff = np.asarray(runoff_direction_world_xy, dtype=np.float64)
        if np.linalg.norm(runoff) <= 1e-12:
            raise ValueError("runoff direction cannot be zero")
        self.runoff_direction_xy = runoff / np.linalg.norm(runoff)
        if washability is None:
            washability_array = np.ones(len(grid.activity_bq), dtype=np.float64)
        else:
            washability_array = np.asarray(washability, dtype=np.float64).reshape(-1)
        if len(washability_array) != len(grid.activity_bq) or np.any(washability_array < 0.0):
            raise ValueError("washability must be non-negative and match the grid")
        self.washability = washability_array
        if state.surface_water_l_by_cell is None:
            state.surface_water_l_by_cell = np.zeros(
                len(grid.activity_bq),
                dtype=np.float64,
            )
        elif len(state.surface_water_l_by_cell) != len(grid.activity_bq):
            raise ValueError("surface water state does not match the grid")
        self._runoff_targets = self._build_runoff_targets()

    @property
    def accounted_activity_bq(self) -> float:
        return (
            self.grid.total_activity_bq
            + self.state.captured_activity_bq
            + self.state.discharged_activity_bq
        )

    @property
    def mass_balance_error_bq(self) -> float:
        return self.accounted_activity_bq - self.reference_activity_bq

    @property
    def water_balance_error_l(self) -> float:
        return self.state.applied_water_l - (
            self.state.recovered_water_l
            + self.state.retained_surface_water_l
            + self.state.discharged_water_l
        )

    def apply(
        self,
        spec: WaterJetSpec,
        *,
        nozzle_world_m: tuple[float, float, float],
        jet_direction_world: tuple[float, float, float],
        surface_speed_m_s: float,
        dt_s: float,
    ) -> WaterDecontaminationStep:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        nozzle = np.asarray(nozzle_world_m, dtype=np.float64)
        direction = np.asarray(jet_direction_world, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm <= 1e-12:
            raise ValueError("jet direction cannot be zero")
        direction /= norm
        denominator = float(np.dot(direction, self.surface_normal))
        incidence_cosine = -denominator
        if incidence_cosine <= 1e-9:
            return self._blocked("jet points away from the surface")
        incidence_angle = math.degrees(math.acos(float(np.clip(incidence_cosine, 0.0, 1.0))))
        if incidence_angle > spec.max_incidence_angle_deg:
            return self._blocked("incidence angle exceeds tool limit", incidence_angle)
        surface_point = self.grid.center_world_m
        standoff = float(np.dot(surface_point - nozzle, self.surface_normal) / denominator)
        if standoff < spec.min_standoff_m or standoff > spec.max_standoff_m:
            return self._blocked("standoff is outside tool range", incidence_angle, standoff)
        impact = nozzle + direction * standoff

        radius = max(
            spec.minimum_footprint_radius_m,
            standoff * math.tan(math.radians(spec.spray_cone_angle_deg * 0.5)),
        )
        relative = self.grid.centers_world_m - impact
        normal_distance = relative @ self.surface_normal
        in_plane = relative - normal_distance[:, None] * self.surface_normal
        radial_distance = np.linalg.norm(in_plane, axis=1)
        effective_radius = radius / max(incidence_cosine, 0.2)
        contacted = radial_distance <= effective_radius
        indices = np.flatnonzero(contacted)
        if not len(indices):
            return self._blocked(
                "spray footprint misses source cells",
                incidence_angle,
                standoff,
                tuple(float(value) for value in impact),
            )

        desired_water_l = spec.flow_rate_l_min / 60.0 * dt_s
        available_water_l = min(desired_water_l, self.state.supply_remaining_l)
        if spec.require_wastewater_collection and spec.water_recovery_fraction > 0.0:
            tank_limited = self.state.wastewater_capacity_remaining_l / spec.water_recovery_fraction
            available_water_l = min(available_water_l, tank_limited)
        if available_water_l <= 1e-12:
            return self._blocked(
                "water supply empty or wastewater tank full",
                incidence_angle,
                standoff,
                tuple(float(value) for value in impact),
            )

        sigma = max(radius * 0.45, min(self.grid.cell_size_x_m, self.grid.cell_size_y_m))
        weights = np.exp(-0.5 * (radial_distance[indices] / sigma) ** 2)
        weights /= weights.sum()
        water_by_cell_l = available_water_l * weights
        cell_area_m2 = self.grid.cell_size_x_m * self.grid.cell_size_y_m
        pressure_factor = math.sqrt(spec.pressure_mpa / spec.reference_pressure_mpa)
        speed_factor = min(
            1.0,
            spec.max_surface_speed_m_s / max(surface_speed_m_s, 1e-9),
        )
        exponent = (
            self.treatment.removal_coefficient_m2_per_l
            * water_by_cell_l
            / cell_area_m2
            * pressure_factor
            * incidence_cosine
            * speed_factor
            * self.washability[indices]
        )
        before = self.grid.activity_bq[indices].copy()
        after = before * np.exp(-exponent)
        removed = before - after
        self.grid.activity_bq[indices] = after

        captured = removed * self.treatment.activity_capture_fraction
        uncaptured = removed - captured
        requested_redeposit = (
            uncaptured * self.treatment.runoff_redeposition_fraction
        )
        redeposited = 0.0
        for source_index, amount in zip(indices, requested_redeposit, strict=True):
            target_index = self._runoff_targets[source_index]
            if target_index >= 0:
                self.grid.activity_bq[target_index] += amount
                redeposited += float(amount)
        captured_total = float(captured.sum())
        removed_total = float(removed.sum())
        discharged_activity = float(uncaptured.sum()) - redeposited

        recovered_water = available_water_l * spec.water_recovery_fraction
        retained_water = available_water_l * spec.surface_water_retention_fraction
        discharged_water = available_water_l - recovered_water - retained_water
        self.state.supply_remaining_l -= available_water_l
        self.state.applied_water_l += available_water_l
        self.state.recovered_water_l += recovered_water
        self.state.wastewater_volume_l += recovered_water
        self.state.retained_surface_water_l += retained_water
        self.state.discharged_water_l += discharged_water
        self.state.captured_activity_bq += captured_total
        self.state.discharged_activity_bq += discharged_activity
        self.state.redeposited_activity_bq += redeposited
        retained_weights = retained_water * weights
        self.state.surface_water_l_by_cell[indices] += retained_weights
        equivalent_dwell_s = water_by_cell_l / max(
            spec.flow_rate_l_min / 60.0,
            1e-12,
        )
        self.grid.cumulative_exposure_s[indices] += equivalent_dwell_s

        return WaterDecontaminationStep(
            contacted_cells=tuple(int(index) for index in indices),
            impact_world_m=tuple(float(value) for value in impact),
            standoff_m=standoff,
            incidence_angle_deg=incidence_angle,
            applied_water_l=available_water_l,
            recovered_water_l=recovered_water,
            removed_activity_bq=removed_total,
            captured_activity_bq=captured_total,
            redeposited_activity_bq=redeposited,
            discharged_activity_bq=discharged_activity,
        )

    def visual_color_rgb(self, wetness_at_full_color_l_m2: float = 0.8) -> np.ndarray:
        if wetness_at_full_color_l_m2 <= 0.0:
            raise ValueError("wetness scale must be positive")
        activity_colors = self.grid.color_rgb()
        cell_area = self.grid.cell_size_x_m * self.grid.cell_size_y_m
        wetness = np.clip(
            self.state.surface_water_l_by_cell / cell_area / wetness_at_full_color_l_m2,
            0.0,
            1.0,
        )
        water_blue = np.asarray([0.05, 0.38, 0.92])
        blend = (0.30 * wetness)[:, None]
        return activity_colors * (1.0 - blend) + water_blue * blend

    def _build_runoff_targets(self) -> np.ndarray:
        step = min(self.grid.cell_size_x_m, self.grid.cell_size_y_m) * 1.05
        targets = np.full(len(self.grid.activity_bq), -1, dtype=np.int64)
        lower = -np.asarray([self.grid.size_x_m, self.grid.size_y_m]) * 0.5
        upper = np.asarray([self.grid.size_x_m, self.grid.size_y_m]) * 0.5
        for index, center in enumerate(self.grid.centers_surface_uv_m):
            target_point = center + self.runoff_direction_xy * step
            if np.any(target_point < lower) or np.any(target_point > upper):
                continue
            distance = np.linalg.norm(
                self.grid.centers_surface_uv_m - target_point,
                axis=1,
            )
            target = int(np.argmin(distance))
            if target != index:
                targets[index] = target
        return targets

    @staticmethod
    def _blocked(
        reason: str,
        incidence_angle_deg: float | None = None,
        standoff_m: float | None = None,
        impact_world_m: tuple[float, float, float] | None = None,
    ) -> WaterDecontaminationStep:
        return WaterDecontaminationStep(
            contacted_cells=(),
            impact_world_m=impact_world_m,
            standoff_m=standoff_m,
            incidence_angle_deg=incidence_angle_deg,
            applied_water_l=0.0,
            recovered_water_l=0.0,
            removed_activity_bq=0.0,
            captured_activity_bq=0.0,
            redeposited_activity_bq=0.0,
            discharged_activity_bq=0.0,
            blocked_reason=reason,
        )


@dataclass(frozen=True, slots=True)
class TriangleSurfaceGeometry:
    """The exact visible triangulated surface used by treatment and activity."""

    vertices_world_m: FloatArray
    face_vertex_indices: IntArray
    activity_triangle_indices: IntArray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_world_m, dtype=np.float64)
        faces = np.asarray(self.face_vertex_indices, dtype=np.int64)
        activity_indices = np.asarray(self.activity_triangle_indices, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3:
            raise ValueError("surface vertices must have shape (V, 3)")
        if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) == 0:
            raise ValueError("surface faces must have shape (F, 3)")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("surface faces reference invalid vertices")
        if not np.array_equal(activity_indices, np.arange(len(faces), dtype=np.int64)):
            raise ValueError(
                "water treatment requires activity on every and only every visible face"
            )
        triangles = vertices[faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        if not np.all(np.isfinite(vertices)) or np.any(np.linalg.norm(cross, axis=1) <= 1e-12):
            raise ValueError("surface mesh contains non-finite or degenerate triangles")
        object.__setattr__(self, "vertices_world_m", vertices)
        object.__setattr__(self, "face_vertex_indices", faces)
        object.__setattr__(self, "activity_triangle_indices", activity_indices)

    @property
    def triangles_world_m(self) -> FloatArray:
        return self.vertices_world_m[self.face_vertex_indices]

    @property
    def centroids_world_m(self) -> FloatArray:
        return np.mean(self.triangles_world_m, axis=1)

    @property
    def normals_world(self) -> FloatArray:
        triangles = self.triangles_world_m
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        return cross / np.linalg.norm(cross, axis=1)[:, None]

    @property
    def area_m2(self) -> FloatArray:
        triangles = self.triangles_world_m
        return 0.5 * np.linalg.norm(
            np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
            axis=1,
        )

    def first_ray_hit(
        self,
        origin_world_m: FloatArray,
        direction_world: FloatArray,
    ) -> tuple[int, float] | None:
        """Return the nearest intersection with an actual visible activity face."""

        origin = np.asarray(origin_world_m, dtype=np.float64)
        direction = np.asarray(direction_world, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        triangles = self.triangles_world_m
        edge1 = triangles[:, 1] - triangles[:, 0]
        edge2 = triangles[:, 2] - triangles[:, 0]
        h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
        determinant = np.einsum("ij,ij->i", edge1, h)
        valid = np.abs(determinant) > 1e-12
        inverse = np.zeros_like(determinant)
        inverse[valid] = 1.0 / determinant[valid]
        offset = origin[None, :] - triangles[:, 0]
        u = inverse * np.einsum("ij,ij->i", offset, h)
        q = np.cross(offset, edge1)
        v = inverse * (q @ direction)
        distance = inverse * np.einsum("ij,ij->i", edge2, q)
        valid &= (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1.0 + 1e-10)
        valid &= distance > 1e-10
        if not np.any(valid):
            return None
        candidates = np.flatnonzero(valid)
        row = int(candidates[np.argmin(distance[candidates])])
        return int(self.activity_triangle_indices[row]), float(distance[row])


class TriangleWaterSurfaceDecontaminator:
    """Water treatment on the same visible triangles used by radiation sampling."""

    def __init__(
        self,
        activity_map: object,
        geometry: TriangleSurfaceGeometry,
        state: WaterDecontaminationState,
        treatment: WaterJetTreatment,
        *,
        runoff_direction_world: tuple[float, float, float],
        random_seed: int = 0,
    ) -> None:
        from radcounter.core.scene import SurfaceActivityMap

        if not isinstance(activity_map, SurfaceActivityMap):
            raise TypeError("triangle water treatment requires a SurfaceActivityMap")
        if len(activity_map.activity_bq) != len(geometry.face_vertex_indices):
            raise ValueError("activity and visible triangle counts differ")
        self.activity_map = activity_map
        self.geometry = geometry
        self.state = state
        self.treatment = treatment
        runoff = np.asarray(runoff_direction_world, dtype=np.float64)
        if not np.all(np.isfinite(runoff)) or np.linalg.norm(runoff) <= 1e-12:
            raise ValueError("runoff direction must be finite and nonzero")
        self.runoff_direction_world = runoff / np.linalg.norm(runoff)
        rng = np.random.default_rng(random_seed)
        variation = rng.normal(size=len(activity_map.activity_bq))
        self.washability = np.clip(
            treatment.washability_mean + treatment.washability_std * variation,
            0.05,
            None,
        )
        if state.surface_water_l_by_cell is None:
            state.surface_water_l_by_cell = np.zeros(len(activity_map.activity_bq))
        elif len(state.surface_water_l_by_cell) != len(activity_map.activity_bq):
            raise ValueError("surface-water state and visible triangle counts differ")
        self.reference_activity_bq = (
            float(np.sum(activity_map.activity_bq))
            + state.captured_activity_bq
            + state.discharged_activity_bq
        )
        self._runoff_targets = self._build_runoff_targets()

    @property
    def mass_balance_error_bq(self) -> float:
        accounted = (
            float(np.sum(self.activity_map.activity_bq))
            + self.state.captured_activity_bq
            + self.state.discharged_activity_bq
        )
        return accounted - self.reference_activity_bq

    @property
    def water_balance_error_l(self) -> float:
        return self.state.applied_water_l - (
            self.state.recovered_water_l
            + self.state.retained_surface_water_l
            + self.state.discharged_water_l
        )

    def apply(
        self,
        spec: WaterJetSpec,
        *,
        nozzle_world_m: tuple[float, float, float],
        jet_direction_world: tuple[float, float, float],
        surface_speed_m_s: float,
        dt_s: float,
        simulation_step: int,
    ) -> WaterDecontaminationStep:
        if dt_s <= 0.0 or surface_speed_m_s < 0.0:
            raise ValueError("water-treatment time must be positive and speed nonnegative")
        nozzle = np.asarray(nozzle_world_m, dtype=np.float64)
        direction = np.asarray(jet_direction_world, dtype=np.float64)
        if not np.all(np.isfinite(direction)) or np.linalg.norm(direction) <= 1e-12:
            raise ValueError("jet direction must be finite and nonzero")
        direction /= np.linalg.norm(direction)
        hit = self.geometry.first_ray_hit(nozzle, direction)
        if hit is None:
            return WaterSurfaceDecontaminator._blocked("nozzle ray misses visible activity mesh")
        hit_triangle, standoff = hit
        if standoff < spec.min_standoff_m or standoff > spec.max_standoff_m:
            return WaterSurfaceDecontaminator._blocked(
                "standoff is outside tool range", standoff_m=standoff
            )
        normals = self.geometry.normals_world
        center_incidence_cosine = float(np.dot(-direction, normals[hit_triangle]))
        center_angle = math.degrees(
            math.acos(float(np.clip(center_incidence_cosine, 0.0, 1.0)))
        )
        if center_incidence_cosine <= 0.0 or center_angle > spec.max_incidence_angle_deg:
            return WaterSurfaceDecontaminator._blocked(
                "incidence angle exceeds tool limit", center_angle, standoff
            )
        impact = nozzle + direction * standoff
        radius = max(
            spec.minimum_footprint_radius_m,
            standoff * math.tan(math.radians(spec.spray_cone_angle_deg * 0.5)),
        )
        relative = self.geometry.centroids_world_m - impact
        radial = np.linalg.norm(
            relative - (relative @ direction)[:, None] * direction[None, :], axis=1
        )
        incidence_cosine = normals @ (-direction)
        normal_threshold = math.cos(math.radians(spec.max_incidence_angle_deg))
        contacted = (radial <= radius) & (incidence_cosine >= normal_threshold)
        indices = np.flatnonzero(contacted)
        if not len(indices):
            return WaterSurfaceDecontaminator._blocked(
                "spray cone misses visible activity triangles",
                center_angle,
                standoff,
                tuple(float(value) for value in impact),
            )
        desired_water_l = spec.flow_rate_l_min / 60.0 * dt_s
        available_water_l = min(desired_water_l, self.state.supply_remaining_l)
        if spec.require_wastewater_collection and spec.water_recovery_fraction > 0.0:
            available_water_l = min(
                available_water_l,
                self.state.wastewater_capacity_remaining_l / spec.water_recovery_fraction,
            )
        if available_water_l <= 1e-12:
            return WaterSurfaceDecontaminator._blocked(
                "water supply empty or wastewater tank full",
                center_angle,
                standoff,
                tuple(float(value) for value in impact),
            )
        sigma = max(radius * 0.45, math.sqrt(float(np.median(self.geometry.area_m2))))
        weights = np.exp(-0.5 * (radial[indices] / sigma) ** 2)
        weights *= self.geometry.area_m2[indices]
        weights /= np.sum(weights)
        water_by_triangle_l = available_water_l * weights
        pressure_factor = math.sqrt(spec.pressure_mpa / spec.reference_pressure_mpa)
        speed_factor = min(1.0, spec.max_surface_speed_m_s / max(surface_speed_m_s, 1e-9))
        exponent = (
            self.treatment.removal_coefficient_m2_per_l
            * water_by_triangle_l
            / self.geometry.area_m2[indices]
            * pressure_factor
            * incidence_cosine[indices]
            * speed_factor
            * self.washability[indices]
        )
        before = self.activity_map.activity_bq[indices].copy()
        after = before * np.exp(-exponent)
        removed = before - after
        self.activity_map.activity_bq[indices] = after
        captured = removed * self.treatment.activity_capture_fraction
        uncaptured = removed - captured
        requested_redeposit = uncaptured * self.treatment.runoff_redeposition_fraction
        redeposited = 0.0
        for source_index, amount in zip(indices, requested_redeposit, strict=True):
            target = self._runoff_targets[source_index]
            if target >= 0:
                self.activity_map.activity_bq[target] += amount
                redeposited += float(amount)
        captured_total = float(np.sum(captured))
        removed_total = float(np.sum(removed))
        discharged_activity = float(np.sum(uncaptured)) - redeposited
        recovered_water = available_water_l * spec.water_recovery_fraction
        retained_water = available_water_l * spec.surface_water_retention_fraction
        discharged_water = available_water_l - recovered_water - retained_water
        self.state.supply_remaining_l -= available_water_l
        self.state.applied_water_l += available_water_l
        self.state.recovered_water_l += recovered_water
        self.state.wastewater_volume_l += recovered_water
        self.state.retained_surface_water_l += retained_water
        self.state.discharged_water_l += discharged_water
        self.state.captured_activity_bq += captured_total
        self.state.discharged_activity_bq += discharged_activity
        self.state.redeposited_activity_bq += redeposited
        self.state.surface_water_l_by_cell[indices] += retained_water * weights
        equivalent_dwell_s = water_by_triangle_l / max(
            spec.flow_rate_l_min / 60.0, 1e-12
        )
        self.activity_map.cumulative_treatment_exposure[indices] += equivalent_dwell_s
        self.activity_map.last_treated_step[indices] = simulation_step
        return WaterDecontaminationStep(
            contacted_cells=tuple(
                int(self.geometry.activity_triangle_indices[index]) for index in indices
            ),
            impact_world_m=tuple(float(value) for value in impact),
            standoff_m=standoff,
            incidence_angle_deg=center_angle,
            applied_water_l=available_water_l,
            recovered_water_l=recovered_water,
            removed_activity_bq=removed_total,
            captured_activity_bq=captured_total,
            redeposited_activity_bq=redeposited,
            discharged_activity_bq=discharged_activity,
        )

    def _build_runoff_targets(self) -> IntArray:
        centers = self.geometry.centroids_world_m
        characteristic = math.sqrt(float(np.median(self.geometry.area_m2)))
        maximum_distance = characteristic * 4.0
        targets = np.full(len(centers), -1, dtype=np.int64)
        for index, center in enumerate(centers):
            delta = centers - center
            forward = delta @ self.runoff_direction_world
            distance = np.linalg.norm(delta, axis=1)
            candidates = np.flatnonzero(
                (forward > characteristic * 0.2)
                & (distance <= maximum_distance)
                & (np.arange(len(centers)) != index)
            )
            if len(candidates):
                lateral = np.sqrt(
                    np.maximum(
                        distance[candidates] ** 2 - forward[candidates] ** 2,
                        0.0,
                    )
                )
                score = distance[candidates] + 2.0 * lateral
                targets[index] = int(candidates[np.argmin(score)])
        return targets
