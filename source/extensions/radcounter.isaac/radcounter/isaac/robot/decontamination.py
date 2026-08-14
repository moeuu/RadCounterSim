"""Contact-driven surface decontamination using live PhysX scene queries."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from radcounter.core.surface_decontamination import effective_contact_exposure_s

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class DecontaminationConfig:
    footprint_points_local_m: tuple[tuple[float, float, float], ...] = (
        (-0.08, -0.08, 0.0),
        (0.0, -0.08, 0.0),
        (0.08, -0.08, 0.0),
        (-0.08, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.08, 0.0, 0.0),
        (-0.08, 0.08, 0.0),
        (0.0, 0.08, 0.0),
        (0.08, 0.08, 0.0),
    )
    treatment_axis_local: tuple[float, float, float] = (0.0, 0.0, -1.0)
    max_contact_distance_m: float = 0.035
    max_normal_angle_deg: float = 25.0
    max_surface_speed_m_s: float = 0.3
    rate_constant_s_inv: float = 0.9
    efficiency_mean: float = 0.86
    efficiency_std: float = 0.08
    transfer_mode: Literal["discard", "transfer_to_waste"] = "transfer_to_waste"
    waste_source_path: str = "/World/DecontaminationWaste"
    random_seed: int = 19
    recontamination_rate_bq_s: float = 0.0


@dataclass(frozen=True, slots=True)
class TreatmentTickResult:
    accepted_contacts: int
    rejected_contacts: int
    treated_triangle_indices: tuple[int, ...]
    removed_activity_bq: float
    cumulative_removed_activity_bq: float
    recontaminated_activity_bq: float = 0.0


class ContactDrivenDecontaminator:
    """Modify truth activity only when the physical tool satisfies contact constraints."""

    def __init__(
        self,
        stage: Any,
        tool_path: str,
        surface_path: str,
        config: DecontaminationConfig | None = None,
    ) -> None:
        import omni.physx

        self.stage = stage
        self.tool_path = tool_path
        self.surface_path = surface_path
        self.config = config or DecontaminationConfig()
        self._query = omni.physx.get_physx_scene_query_interface()
        self._rng = np.random.default_rng(self.config.random_seed)
        self._previous_center: FloatArray | None = None
        self._cumulative_removed = 0.0
        self._dirty = False
        surface = stage.GetPrimAtPath(surface_path)
        if (
            not surface
            or not surface.IsValid()
            or not bool(self._attribute(surface, "rad:decon:enabled", False))
        ):
            raise ValueError(f"{surface_path} is not an enabled decontamination surface")
        uri = str(
            self._attribute(
                surface,
                "rad:decon:activityMapUri",
                self._attribute(surface, "rad:source:activityMapUri", ""),
            )
        )
        root_layer = Path(stage.GetRootLayer().realPath).resolve()
        self.activity_path = (
            Path(uri) if Path(uri).is_absolute() else (root_layer.parent / uri).resolve()
        )
        expected_digest = str(self._attribute(surface, "rad:decon:activityMapSha256", ""))
        actual_digest = hashlib.sha256(self.activity_path.read_bytes()).hexdigest()
        if expected_digest and expected_digest != actual_digest:
            raise ValueError(f"activity-map digest mismatch for {self.activity_path}")
        with np.load(self.activity_path, allow_pickle=False) as payload:
            self.triangle_indices = np.asarray(payload["triangle_indices"], dtype=np.int64)
            self.activity_bq = np.asarray(payload["activity_bq"], dtype=np.float64)
            self.exposure = np.asarray(payload["cumulative_treatment_exposure"], dtype=np.float64)
            self.last_treated_step = np.asarray(payload["last_treated_step"], dtype=np.int64)
        self.initial_activity_bq = self.activity_bq.copy()
        from pxr import UsdGeom

        authored_colors = UsdGeom.Gprim(surface).GetDisplayColorAttr().Get() or ()
        initial_colors = np.asarray(authored_colors, dtype=np.float64)
        self._initial_display_colors = (
            initial_colors if initial_colors.shape == (len(self.activity_bq), 3) else None
        )
        common = self._rng.normal()
        local = self._rng.normal(size=len(self.activity_bq))
        correlation = 0.75
        variation = correlation * common + math.sqrt(1.0 - correlation**2) * local
        self.truth_efficiency = np.clip(
            self.config.efficiency_mean + self.config.efficiency_std * variation,
            0.05,
            1.0,
        )
        self._row_by_triangle = {int(index): row for row, index in enumerate(self.triangle_indices)}

    @staticmethod
    def _attribute(prim: Any, name: str, default: object = None) -> object:
        attr = prim.GetAttribute(name)
        if not attr or not attr.HasAuthoredValueOpinion():
            return default
        value = attr.Get()
        return default if value is None else value

    def _tool_samples(self) -> tuple[FloatArray, FloatArray, FloatArray]:
        from pxr import Gf, UsdGeom

        tool = self.stage.GetPrimAtPath(self.tool_path)
        transform = UsdGeom.XformCache().GetLocalToWorldTransform(tool)
        samples = np.asarray(
            [
                transform.Transform(Gf.Vec3d(*point))
                for point in self.config.footprint_points_local_m
            ],
            dtype=np.float64,
        )
        center = np.asarray(transform.Transform(Gf.Vec3d()), dtype=np.float64)
        axis_endpoint = np.asarray(
            transform.Transform(Gf.Vec3d(*self.config.treatment_axis_local)), dtype=np.float64
        )
        axis = axis_endpoint - center
        axis /= np.linalg.norm(axis)
        return samples, center, axis

    def _update_surface_visuals(self) -> None:
        """Fade each rendered face using the same remaining truth activity."""

        if self._initial_display_colors is None:
            return
        from pxr import Gf, UsdGeom

        surface = self.stage.GetPrimAtPath(self.surface_path)
        gprim = UsdGeom.Gprim(surface)
        fraction = np.divide(
            self.activity_bq,
            self.initial_activity_bq,
            out=np.zeros_like(self.activity_bq),
            where=self.initial_activity_bq > 0.0,
        )
        fraction = np.clip(fraction, 0.0, 1.0)
        host_color = np.asarray((0.20, 0.27, 0.32), dtype=np.float64)
        colors = host_color + fraction[:, None] * (
            self._initial_display_colors - host_color
        )
        color_attr = gprim.GetDisplayColorAttr()
        color_attr.Set([Gf.Vec3f(*color) for color in colors])
        color_attr.SetMetadata("interpolation", UsdGeom.Tokens.uniform)
        opacity_attr = gprim.GetDisplayOpacityAttr()
        if not opacity_attr:
            opacity_attr = gprim.CreateDisplayOpacityAttr()
        opacity_attr.Set([0.0 if value < 0.10 else 1.0 for value in fraction])
        opacity_attr.SetMetadata("interpolation", UsdGeom.Tokens.uniform)

    def tick(self, dt_s: float, simulation_step: int) -> TreatmentTickResult:
        import carb

        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        samples, center, axis = self._tool_samples()
        speed = (
            0.0
            if self._previous_center is None
            else float(np.linalg.norm(center - self._previous_center) / dt_s)
        )
        self._previous_center = center
        accepted = 0
        rejected = 0
        hit_counts: dict[int, int] = {}
        normal_threshold = math.cos(math.radians(self.config.max_normal_angle_deg))
        for origin in samples:
            hit = self._query.raycast_closest(
                carb.Float3(*origin),
                carb.Float3(*axis),
                self.config.max_contact_distance_m,
                True,
            )
            if not hit.get("hit", False) or str(hit.get("collision", "")) != self.surface_path:
                rejected += 1
                continue
            normal = np.asarray(hit["normal"], dtype=np.float64)
            alignment = float(np.dot(-axis, normal / np.linalg.norm(normal)))
            triangle_index = int(hit["faceIndex"])
            if (
                speed > self.config.max_surface_speed_m_s
                or alignment < normal_threshold
                or triangle_index not in self._row_by_triangle
            ):
                rejected += 1
                continue
            accepted += 1
            hit_counts[triangle_index] = hit_counts.get(triangle_index, 0) + 1
        removed_total = 0.0
        effective_exposure_s = effective_contact_exposure_s(
            dt_s,
            speed,
            self.config.max_surface_speed_m_s,
        )
        for triangle_index in hit_counts:
            row = self._row_by_triangle[triangle_index]
            # Ray count is spatial sampling density, not elapsed time.  Apply
            # one contact tick per hit face so densifying the pad footprint
            # cannot dilute treatment, matching SurfaceSourceGrid.apply_tool().
            incremental_exposure = effective_exposure_s
            removal_fraction = 1.0 - math.exp(
                -self.config.rate_constant_s_inv * incremental_exposure * self.truth_efficiency[row]
            )
            removed = float(self.activity_bq[row] * removal_fraction)
            self.activity_bq[row] -= removed
            self.exposure[row] += incremental_exposure
            self.last_treated_step[row] = simulation_step
            removed_total += removed
        if removed_total > 0:
            self._dirty = True
            self._cumulative_removed += removed_total
            if self.config.transfer_mode == "transfer_to_waste":
                self._transfer_to_waste(removed_total)
            if simulation_step % 10 == 0:
                self._update_surface_visuals()
        recontaminated = self.inject_recontamination(self.config.recontamination_rate_bq_s * dt_s)
        return TreatmentTickResult(
            accepted_contacts=accepted,
            rejected_contacts=rejected,
            treated_triangle_indices=tuple(sorted(hit_counts)),
            removed_activity_bq=removed_total,
            cumulative_removed_activity_bq=self._cumulative_removed,
            recontaminated_activity_bq=recontaminated,
        )

    def inject_recontamination(
        self,
        activity_bq: float,
        triangle_weights: Sequence[float] | None = None,
    ) -> float:
        if activity_bq < 0:
            raise ValueError("recontamination activity must be non-negative")
        if activity_bq == 0:
            return 0.0
        if triangle_weights is None:
            weights = 1.0 / np.maximum(self.truth_efficiency, 0.05)
        else:
            weights = np.asarray(triangle_weights, dtype=np.float64)
            if weights.shape != self.activity_bq.shape or np.any(weights < 0):
                raise ValueError("triangle_weights must be non-negative and match the activity map")
        total_weight = float(np.sum(weights))
        if total_weight <= 0:
            raise ValueError("triangle_weights must contain positive mass")
        self.activity_bq += activity_bq * weights / total_weight
        self._dirty = True
        return float(activity_bq)

    def _transfer_to_waste(self, removed_activity_bq: float) -> None:
        from pxr import Gf, Sdf, UsdGeom

        waste = self.stage.GetPrimAtPath(self.config.waste_source_path)
        if not waste or not waste.IsValid():
            waste = UsdGeom.Xform.Define(self.stage, self.config.waste_source_path).GetPrim()
            waste.CreateAttribute("rad:role", Sdf.ValueTypeNames.String, custom=True).Set("source")
            waste.CreateAttribute("rad:source:type", Sdf.ValueTypeNames.String, custom=True).Set(
                "point"
            )
            surface = self.stage.GetPrimAtPath(self.surface_path)
            isotope = str(self._attribute(surface, "rad:source:isotopeId", ""))
            waste.CreateAttribute(
                "rad:source:isotopeId", Sdf.ValueTypeNames.String, custom=True
            ).Set(isotope)
            waste.CreateAttribute(
                "rad:source:hiddenFromEstimator", Sdf.ValueTypeNames.Bool, custom=True
            ).Set(False)
            waste.CreateAttribute("rad:source:enabled", Sdf.ValueTypeNames.Bool, custom=True).Set(
                True
            )
            zone = self.stage.GetPrimAtPath("/World/DisposalZone")
            if zone and zone.IsValid():
                world = UsdGeom.XformCache().GetLocalToWorldTransform(zone).Transform(Gf.Vec3d())
                UsdGeom.Xformable(waste).AddTranslateOp().Set(Gf.Vec3d(*world))
        activity = waste.GetAttribute("rad:source:activityBq")
        if not activity:
            activity = waste.CreateAttribute(
                "rad:source:activityBq", Sdf.ValueTypeNames.Double, custom=True
            )
        activity.Set(float(activity.Get() or 0.0) + removed_activity_bq)

    def flush(self) -> str:
        if not self._dirty:
            return hashlib.sha256(self.activity_path.read_bytes()).hexdigest()
        self._update_surface_visuals()
        self.activity_path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.activity_path.stem}.",
            suffix=".npz",
            dir=self.activity_path.parent,
        )
        os.close(file_descriptor)
        temporary = Path(temporary_name)
        try:
            np.savez_compressed(
                temporary,
                triangle_indices=self.triangle_indices,
                activity_bq=self.activity_bq,
                cumulative_treatment_exposure=self.exposure,
                last_treated_step=self.last_treated_step,
            )
            os.replace(temporary, self.activity_path)
        finally:
            temporary.unlink(missing_ok=True)
        digest = hashlib.sha256(self.activity_path.read_bytes()).hexdigest()
        surface = self.stage.GetPrimAtPath(self.surface_path)
        for name in ("rad:source:activityMapSha256", "rad:decon:activityMapSha256"):
            attribute = surface.GetAttribute(name)
            if attribute:
                attribute.Set(digest)
        self._dirty = False
        return digest
