"""Robot-executed water treatment on the visible activity-bearing USD mesh."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from radcounter.core.scene import SurfaceActivityMap, resolve_asset_uri
from radcounter.core.treatment import TreatmentMaterialModel, load_treatment_material_model
from radcounter.core.water_decontamination import (
    TriangleSurfaceGeometry,
    TriangleWaterSurfaceDecontaminator,
    WaterDecontaminationState,
    WaterDecontaminationStep,
    WaterJetSpec,
)

from .disposal import DisposalDisposition, disposal_configuration


@dataclass(frozen=True, slots=True)
class MeshWaterDecontaminationConfig:
    nozzle_origin_local_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    jet_axis_local: tuple[float, float, float] = (0.0, 0.0, -1.0)
    runoff_direction_world: tuple[float, float, float] = (1.0, 0.0, 0.0)
    captured_waste_source_path: str = "/World/DecontaminationWastewater"
    discharge_source_path: str = "/World/DecontaminationRunoff"
    random_seed: int = 31
    footprint_points_local_m: tuple[tuple[float, float, float], ...] = (
        (-0.16, -0.16, 0.0),
        (0.16, -0.16, 0.0),
        (-0.16, 0.16, 0.0),
        (0.16, 0.16, 0.0),
    )


@dataclass(frozen=True, slots=True)
class MeshWaterTreatmentResult:
    step: WaterDecontaminationStep
    activity_balance_error_bq: float
    water_balance_error_l: float
    treatment_model_id: str
    treatment_data_status: str
    treatment_numeric_sha256: str

    @property
    def accepted_contacts(self) -> int:
        return len(self.step.contacted_cells)

    @property
    def rejected_contacts(self) -> int:
        return int(self.step.blocked_reason is not None)

    @property
    def removed_activity_bq(self) -> float:
        return self.step.removed_activity_bq

    @property
    def treated_triangle_indices(self) -> tuple[int, ...]:
        return self.step.contacted_cells

    @property
    def dwell_qualified_triangle_indices(self) -> tuple[int, ...]:
        return self.step.contacted_cells

    @property
    def dwell_pending_triangle_indices(self) -> tuple[int, ...]:
        return ()

    @property
    def minimum_tool_dwell_s(self) -> float:
        return 0.0


class MeshWaterDecontaminator:
    """Read robot nozzle pose and update the exact visible surface activity map."""

    treatment_method = "water_jet"

    def __init__(
        self,
        stage: Any,
        tool_path: str,
        surface_path: str,
        spec: WaterJetSpec,
        state: WaterDecontaminationState,
        config: MeshWaterDecontaminationConfig | None = None,
    ) -> None:
        from pxr import UsdGeom

        self.stage = stage
        self.tool_path = tool_path
        self.surface_path = surface_path
        self.spec = spec
        self.state = state
        self.config = config or MeshWaterDecontaminationConfig()
        self._previous_nozzle: np.ndarray | None = None
        self._dirty = False
        surface = stage.GetPrimAtPath(surface_path)
        if (
            not surface
            or not surface.IsValid()
            or not bool(self._attribute(surface, "rad:decon:enabled", False))
        ):
            raise ValueError(f"{surface_path} is not an enabled treatment surface")
        tool = stage.GetPrimAtPath(tool_path)
        if not tool or not tool.IsValid():
            raise ValueError(f"water-treatment tool does not exist: {tool_path}")
        root_layer_path = str(stage.GetRootLayer().realPath or "")
        base_directory = Path(root_layer_path).resolve().parent if root_layer_path else Path.cwd()
        uri_value = self._attribute(
            surface,
            "rad:decon:activityMapUri",
            self._attribute(surface, "rad:source:activityMapUri", ""),
        )
        uri = str(getattr(uri_value, "path", uri_value))
        map_sha256 = str(self._attribute(surface, "rad:decon:activityMapSha256", ""))
        if not uri or not map_sha256:
            raise ValueError("water treatment requires an activity-map URI and SHA256")
        self.activity_path = resolve_asset_uri(uri, base_directory)
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
                    "water-treatment activity map is missing arrays: " + ", ".join(sorted(missing))
                )
        self.activity_map = SurfaceActivityMap.load(
            uri,
            base_directory=base_directory,
            expected_sha256=map_sha256,
        )
        substrate = str(self._attribute(surface, "rad:decon:substrateMaterialId", ""))
        treatment_uri_value = self._attribute(surface, "rad:decon:treatmentModelUri", "")
        treatment_uri = str(getattr(treatment_uri_value, "path", treatment_uri_value))
        treatment_sha256 = str(self._attribute(surface, "rad:decon:treatmentModelSha256", ""))
        treatment_path = resolve_asset_uri(treatment_uri, base_directory)
        self.treatment_model: TreatmentMaterialModel = load_treatment_material_model(
            treatment_path,
            expected_file_sha256=treatment_sha256,
            expected_substrate_material_id=substrate,
        )
        mesh = UsdGeom.Mesh(surface)
        if not mesh:
            raise ValueError("water treatment requires a visible USD Mesh")
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        if len(counts) == 0 or np.any(counts != 3):
            raise ValueError("water treatment requires triangulated visible source geometry")
        points_local = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        transform = UsdGeom.XformCache().GetLocalToWorldTransform(surface)
        from pxr import Gf

        points_world = np.asarray(
            [transform.Transform(Gf.Vec3d(*point)) for point in points_local],
            dtype=np.float64,
        )
        faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
        geometry = TriangleSurfaceGeometry(
            points_world,
            faces,
            self.activity_map.triangle_indices,
        )
        self.process = TriangleWaterSurfaceDecontaminator(
            self.activity_map,
            geometry,
            state,
            self.treatment_model.water_jet,
            runoff_direction_world=self.config.runoff_direction_world,
            random_seed=self.config.random_seed,
        )
        colors = np.asarray(UsdGeom.Gprim(surface).GetDisplayColorAttr().Get() or ())
        self._initial_colors = (
            colors.copy() if colors.shape == (len(self.activity_map.activity_bq), 3) else None
        )
        self._initial_activity = self.activity_map.activity_bq.copy()
        self._waste_disposal = disposal_configuration(stage, "/World/DisposalZone")
        if self._waste_disposal.disposition is not DisposalDisposition.SHIELDED_STORAGE:
            raise ValueError("captured wastewater requires visible shielded storage")

    @property
    def activity_bq(self) -> np.ndarray:
        return self.activity_map.activity_bq

    @property
    def triangle_indices(self) -> np.ndarray:
        return self.activity_map.triangle_indices

    @property
    def minimum_tool_dwell_s(self) -> float:
        return 0.0

    @staticmethod
    def _attribute(prim: Any, name: str, default: object = None) -> object:
        attribute = prim.GetAttribute(name)
        if not attribute or not attribute.HasAuthoredValueOpinion():
            return default
        value = attribute.Get()
        return default if value is None else value

    def _nozzle_pose(self) -> tuple[np.ndarray, np.ndarray]:
        from pxr import Gf, UsdGeom

        tool = self.stage.GetPrimAtPath(self.tool_path)
        transform = UsdGeom.XformCache().GetLocalToWorldTransform(tool)
        nozzle = np.asarray(
            transform.Transform(Gf.Vec3d(*self.config.nozzle_origin_local_m)),
            dtype=np.float64,
        )
        endpoint = np.asarray(
            transform.Transform(Gf.Vec3d(*self.config.jet_axis_local)),
            dtype=np.float64,
        )
        direction = endpoint - np.asarray(transform.Transform(Gf.Vec3d()), dtype=np.float64)
        if np.linalg.norm(direction) <= 1e-12:
            raise ValueError("water-treatment jet axis cannot be zero")
        return nozzle, direction / np.linalg.norm(direction)

    def tick(self, dt_s: float, simulation_step: int) -> MeshWaterTreatmentResult:
        nozzle, direction = self._nozzle_pose()
        speed = (
            0.0
            if self._previous_nozzle is None
            else float(np.linalg.norm(nozzle - self._previous_nozzle) / dt_s)
        )
        self._previous_nozzle = nozzle
        step = self.process.apply(
            self.spec,
            nozzle_world_m=tuple(float(value) for value in nozzle),
            jet_direction_world=tuple(float(value) for value in direction),
            surface_speed_m_s=speed,
            dt_s=dt_s,
            simulation_step=simulation_step,
        )
        if step.removed_activity_bq > 0.0:
            self._dirty = True
            isotope_id = str(
                self._attribute(
                    self.stage.GetPrimAtPath(self.surface_path),
                    "rad:source:isotopeId",
                    "",
                )
            )
            if step.captured_activity_bq > 0.0:
                self._add_waste_source(
                    self.config.captured_waste_source_path,
                    step.captured_activity_bq,
                    isotope_id,
                    contained=True,
                    position_world_m=None,
                    stream="captured_wastewater",
                )
            if step.discharged_activity_bq > 0.0:
                self._add_waste_source(
                    self.config.discharge_source_path,
                    step.discharged_activity_bq,
                    isotope_id,
                    contained=False,
                    position_world_m=step.impact_world_m,
                    stream="in_scene_runoff",
                )
            self._update_surface_visuals()
        return MeshWaterTreatmentResult(
            step=step,
            activity_balance_error_bq=self.process.mass_balance_error_bq,
            water_balance_error_l=self.process.water_balance_error_l,
            treatment_model_id=self.treatment_model.model_id,
            treatment_data_status=self.treatment_model.status,
            treatment_numeric_sha256=self.treatment_model.numeric_sha256,
        )

    def _add_waste_source(
        self,
        path: str,
        activity_bq: float,
        isotope_id: str,
        *,
        contained: bool,
        position_world_m: tuple[float, float, float] | None,
        stream: str,
    ) -> None:
        from pxr import Gf, Sdf, UsdGeom

        prim = self.stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            prim = UsdGeom.Xform.Define(self.stage, path).GetPrim()
            prim.CreateAttribute("rad:role", Sdf.ValueTypeNames.String, custom=True).Set("source")
            prim.CreateAttribute("rad:source:type", Sdf.ValueTypeNames.String, custom=True).Set(
                "point"
            )
            prim.CreateAttribute(
                "rad:source:isotopeId", Sdf.ValueTypeNames.String, custom=True
            ).Set(isotope_id)
            prim.CreateAttribute(
                "rad:source:hiddenFromEstimator", Sdf.ValueTypeNames.Bool, custom=True
            ).Set(False)
            prim.CreateAttribute("rad:source:enabled", Sdf.ValueTypeNames.Bool, custom=True).Set(
                True
            )
            prim.CreateAttribute("rad:waste:stream", Sdf.ValueTypeNames.String, custom=True).Set(
                stream
            )
            if contained:
                zone = self.stage.GetPrimAtPath(self._waste_disposal.zone_path)
                location = UsdGeom.XformCache().GetLocalToWorldTransform(zone).Transform(Gf.Vec3d())
                prim.CreateAttribute(
                    "rad:source:contained", Sdf.ValueTypeNames.Bool, custom=True
                ).Set(True)
                prim.CreateAttribute(
                    "rad:source:containmentPrimPath",
                    Sdf.ValueTypeNames.String,
                    custom=True,
                ).Set(str(self._waste_disposal.storage_prim_path))
            else:
                if position_world_m is None:
                    raise ValueError("in-scene runoff requires an explicit world position")
                location = Gf.Vec3d(*position_world_m)
                prim.CreateAttribute(
                    "rad:source:contained", Sdf.ValueTypeNames.Bool, custom=True
                ).Set(False)
            UsdGeom.Xformable(prim).AddTranslateOp().Set(location)
        activity = prim.GetAttribute("rad:source:activityBq")
        if not activity:
            activity = prim.CreateAttribute(
                "rad:source:activityBq", Sdf.ValueTypeNames.Double, custom=True
            )
        activity.Set(float(activity.Get() or 0.0) + activity_bq)

    def _update_surface_visuals(self) -> None:
        if self._initial_colors is None:
            return
        from pxr import Gf, UsdGeom

        fraction = np.divide(
            self.activity_map.activity_bq,
            self._initial_activity,
            out=np.zeros_like(self.activity_map.activity_bq),
            where=self._initial_activity > 0.0,
        )
        fraction = np.clip(fraction, 0.0, 1.0)
        wetness = self.state.surface_water_l_by_cell
        wet_fraction = np.clip(wetness / max(float(np.max(wetness)), 1e-12), 0.0, 1.0)
        host = np.asarray((0.20, 0.27, 0.32))
        water_blue = np.asarray((0.05, 0.38, 0.92))
        colors = host + fraction[:, None] * (self._initial_colors - host)
        colors = colors * (1.0 - 0.25 * wet_fraction[:, None]) + water_blue * (
            0.25 * wet_fraction[:, None]
        )
        prim = self.stage.GetPrimAtPath(self.surface_path)
        attribute = UsdGeom.Gprim(prim).GetDisplayColorAttr()
        attribute.Set([Gf.Vec3f(*color) for color in colors])
        attribute.SetMetadata("interpolation", UsdGeom.Tokens.uniform)

    def flush(self) -> str:
        if not self._dirty:
            return hashlib.sha256(self.activity_path.read_bytes()).hexdigest()
        digest = self.activity_map.save(self.activity_path)
        surface = self.stage.GetPrimAtPath(self.surface_path)
        for name in ("rad:source:activityMapSha256", "rad:decon:activityMapSha256"):
            attribute = surface.GetAttribute(name)
            if attribute:
                attribute.Set(digest)
        self._dirty = False
        return digest
