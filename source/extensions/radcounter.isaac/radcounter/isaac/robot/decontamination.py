"""Contact-driven surface decontamination using live PhysX scene queries."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from radcounter.core.scene.activity_map import SurfaceActivityMap, resolve_asset_uri
from radcounter.core.surface_decontamination import effective_contact_exposure_s
from radcounter.core.treatment import TreatmentMaterialModel, load_treatment_material_model

from .disposal import DisposalDisposition, disposal_configuration

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
    dwell_qualified_triangle_indices: tuple[int, ...] = ()
    dwell_pending_triangle_indices: tuple[int, ...] = ()
    minimum_tool_dwell_s: float = 0.0
    treatment_model_id: str = ""
    treatment_data_status: str = ""
    treatment_numeric_sha256: str = ""
    activity_balance_error_bq: float = 0.0


class ContactDrivenDecontaminator:
    """Modify truth activity only when the physical tool satisfies contact constraints."""

    treatment_method = "dry_contact"

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
        self._cumulative_recontaminated = 0.0
        self._dirty = False
        surface = stage.GetPrimAtPath(surface_path)
        if (
            not surface
            or not surface.IsValid()
            or not bool(self._attribute(surface, "rad:decon:enabled", False))
        ):
            raise ValueError(f"{surface_path} is not an enabled decontamination surface")
        if (
            not self.config.footprint_points_local_m
            or self.config.max_contact_distance_m <= 0.0
            or self.config.max_surface_speed_m_s <= 0.0
        ):
            raise ValueError("decontamination configuration contains invalid physical values")
        minimum_dwell = self._attribute(surface, "rad:decon:minToolDwellS", None)
        if minimum_dwell is None:
            raise ValueError(f"{surface_path} requires authored rad:decon:minToolDwellS")
        self.minimum_tool_dwell_s = float(minimum_dwell)
        if not math.isfinite(self.minimum_tool_dwell_s) or self.minimum_tool_dwell_s < 0.0:
            raise ValueError("rad:decon:minToolDwellS must be finite and nonnegative")
        uri = str(
            self._attribute(
                surface,
                "rad:decon:activityMapUri",
                self._attribute(surface, "rad:source:activityMapUri", ""),
            )
        )
        root_layer_path = str(stage.GetRootLayer().realPath or "")
        base_directory = Path(root_layer_path).resolve().parent if root_layer_path else Path.cwd()
        substrate_material_id = str(self._attribute(surface, "rad:decon:substrateMaterialId", ""))
        treatment_uri_value = self._attribute(surface, "rad:decon:treatmentModelUri", "")
        treatment_uri = str(getattr(treatment_uri_value, "path", treatment_uri_value))
        treatment_digest = str(self._attribute(surface, "rad:decon:treatmentModelSha256", ""))
        if not substrate_material_id or not treatment_uri or not treatment_digest:
            raise ValueError(
                f"{surface_path} requires substrateMaterialId, treatmentModelUri, and "
                "treatmentModelSha256"
            )
        treatment_path = resolve_asset_uri(treatment_uri, base_directory)
        self.treatment_model: TreatmentMaterialModel = load_treatment_material_model(
            treatment_path,
            expected_file_sha256=treatment_digest,
            expected_substrate_material_id=substrate_material_id,
        )
        self.activity_path = resolve_asset_uri(uri, base_directory)
        expected_digest = str(self._attribute(surface, "rad:decon:activityMapSha256", ""))
        if not expected_digest:
            raise ValueError(f"{surface_path} requires rad:decon:activityMapSha256")
        with np.load(self.activity_path, allow_pickle=False) as payload:
            required = {
                "triangle_indices",
                "activity_bq",
                "cumulative_treatment_exposure",
                "last_treated_step",
                "verified_contact_dwell_s",
            }
            missing = required.difference(payload.files)
            if missing:
                raise ValueError(
                    "decontamination activity map is missing arrays: " + ", ".join(sorted(missing))
                )
        self._activity_map = SurfaceActivityMap.load(
            uri,
            base_directory=base_directory,
            expected_sha256=expected_digest,
        )
        self.triangle_indices = self._activity_map.triangle_indices
        self.activity_bq = self._activity_map.activity_bq
        self.exposure = self._activity_map.cumulative_treatment_exposure
        self.last_treated_step = self._activity_map.last_treated_step
        self.verified_contact_dwell_s = self._activity_map.verified_contact_dwell_s
        self.initial_activity_bq = self.activity_bq.copy()
        self._initial_total_activity_bq = float(np.sum(self.initial_activity_bq))
        from pxr import UsdGeom

        mesh = UsdGeom.Mesh(surface)
        if not mesh:
            raise ValueError("contact decontamination requires the visible source to be a Mesh")
        face_vertex_counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        if len(face_vertex_counts) == 0 or np.any(face_vertex_counts != 3):
            raise ValueError(
                "contact decontamination requires triangulated visible source geometry"
            )
        if not np.array_equal(
            self.triangle_indices,
            np.arange(len(face_vertex_counts), dtype=np.int64),
        ):
            raise ValueError(
                "activity map must cover every and only every face of the visible source mesh"
            )
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
            self.treatment_model.dry_contact.efficiency_mean
            + self.treatment_model.dry_contact.efficiency_std * variation,
            0.05,
            1.0,
        )
        self._row_by_triangle = {int(index): row for row, index in enumerate(self.triangle_indices)}
        self._waste_disposal = None
        if self.config.transfer_mode == "transfer_to_waste":
            self._waste_disposal = disposal_configuration(stage, "/World/DisposalZone")
            if self._waste_disposal.disposition is not DisposalDisposition.SHIELDED_STORAGE:
                raise ValueError(
                    "decontamination waste transfer requires in-scene shielded storage"
                )

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
        colors = host_color + fraction[:, None] * (self._initial_display_colors - host_color)
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
        qualified_triangles: list[int] = []
        pending_triangles: list[int] = []
        effective_exposure_s = effective_contact_exposure_s(
            dt_s,
            speed,
            self.config.max_surface_speed_m_s,
        )
        for triangle_index in hit_counts:
            row = self._row_by_triangle[triangle_index]
            previous_dwell_s = float(self.verified_contact_dwell_s[row])
            current_dwell_s = previous_dwell_s + dt_s
            self.verified_contact_dwell_s[row] = current_dwell_s
            eligible_contact_s = max(
                current_dwell_s - self.minimum_tool_dwell_s,
                0.0,
            ) - max(previous_dwell_s - self.minimum_tool_dwell_s, 0.0)
            # Ray count is spatial sampling density, not elapsed time.  Apply
            # one contact tick per hit face so densifying the pad footprint
            # cannot dilute treatment, matching SurfaceSourceGrid.apply_tool().
            incremental_exposure = effective_exposure_s * eligible_contact_s / dt_s
            if incremental_exposure <= 0.0:
                pending_triangles.append(triangle_index)
                continue
            qualified_triangles.append(triangle_index)
            removal_fraction = 1.0 - math.exp(
                -self.treatment_model.dry_contact.rate_constant_s_inv
                * incremental_exposure
                * self.truth_efficiency[row]
            )
            removed = float(self.activity_bq[row] * removal_fraction)
            self.activity_bq[row] -= removed
            self.exposure[row] += incremental_exposure
            self.last_treated_step[row] = simulation_step
            removed_total += removed
        if hit_counts:
            self._dirty = True
        if removed_total > 0:
            self._cumulative_removed += removed_total
            if self.config.transfer_mode == "transfer_to_waste":
                self._transfer_to_waste(removed_total)
            if simulation_step % 10 == 0:
                self._update_surface_visuals()
        recontaminated = self.inject_recontamination(self.config.recontamination_rate_bq_s * dt_s)
        return TreatmentTickResult(
            accepted_contacts=accepted,
            rejected_contacts=rejected,
            treated_triangle_indices=tuple(sorted(qualified_triangles)),
            removed_activity_bq=removed_total,
            cumulative_removed_activity_bq=self._cumulative_removed,
            recontaminated_activity_bq=recontaminated,
            dwell_qualified_triangle_indices=tuple(sorted(qualified_triangles)),
            dwell_pending_triangle_indices=tuple(sorted(pending_triangles)),
            minimum_tool_dwell_s=self.minimum_tool_dwell_s,
            treatment_model_id=self.treatment_model.model_id,
            treatment_data_status=self.treatment_model.status,
            treatment_numeric_sha256=self.treatment_model.numeric_sha256,
            activity_balance_error_bq=self.activity_balance_error_bq,
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
        self._cumulative_recontaminated += activity_bq
        self._dirty = True
        return float(activity_bq)

    @property
    def activity_balance_error_bq(self) -> float:
        """Return source-plus-waste conservation error for this treatment run."""

        return float(
            self._initial_total_activity_bq
            + self._cumulative_recontaminated
            - np.sum(self.activity_bq)
            - self._cumulative_removed
        )

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
            zone = self.stage.GetPrimAtPath(self._waste_disposal.zone_path)
            if zone and zone.IsValid():
                world = UsdGeom.XformCache().GetLocalToWorldTransform(zone).Transform(Gf.Vec3d())
                UsdGeom.Xformable(waste).AddTranslateOp().Set(Gf.Vec3d(*world))
            waste.CreateAttribute("rad:source:contained", Sdf.ValueTypeNames.Bool, custom=True).Set(
                True
            )
            waste.CreateAttribute(
                "rad:source:containmentPrimPath", Sdf.ValueTypeNames.String, custom=True
            ).Set(str(self._waste_disposal.storage_prim_path))
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
        digest = self._activity_map.save(self.activity_path)
        surface = self.stage.GetPrimAtPath(self.surface_path)
        for name in ("rad:source:activityMapSha256", "rad:decon:activityMapSha256"):
            attribute = surface.GetAttribute(name)
            if attribute:
                attribute.Set(digest)
        self._dirty = False
        return digest
