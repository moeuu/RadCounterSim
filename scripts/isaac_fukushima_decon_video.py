#!/usr/bin/env python3
"""Record articulated decontamination inside the imported Fukushima CAD."""
# ruff: noqa: E402, E501

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "source/extensions/radcounter.isaac"
NATIVE = ROOT / "build/native/python"
for search_path in (ROOT, EXTENSION, NATIVE):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

_radcounter_package = importlib.import_module("radcounter")
_extension_namespace = str(EXTENSION / "radcounter")
if _extension_namespace not in _radcounter_package.__path__:
    _radcounter_package.__path__.append(_extension_namespace)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=24.0)
    parser.add_argument("--capture-fps", type=int, default=15)
    parser.add_argument("--capture-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="render the whole-building and interior compositions without executing motion",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/fukushima-decontamination-video",
    )
    args = parser.parse_args()
    if args.duration <= 6.0:
        parser.error("--duration must exceed 6 seconds")
    if args.capture_fps <= 0:
        parser.error("--capture-fps must be positive")
    if args.capture_stride <= 0:
        parser.error("--capture-stride must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    return args


ARGS = arguments()

from isaacsim import SimulationApp

APP = SimulationApp(
    {
        "headless": ARGS.headless,
        "width": 1440,
        "height": 900,
        "renderer": "RaytracedLighting",
        "window_title": "RadInterAct - Fukushima Daiichi CAD Decontamination",
    }
)

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

from radcounter.core.sensors.catalog import popular_detector_catalog
from radcounter.core.sensors.universal import (
    DetectorPose,
    IncidentParticleFluence,
    MeasurementRequest,
    ParametricDetectorModel,
    RadiationType,
    contribution_weighted_sample_without_replacement,
)

CS137_GAMMA_ENERGY_KEV = 661.657
CS137_GAMMA_YIELD = 0.851
RADIATION_SAMPLE_SEED = 20_260_904
SOURCE_LABEL = "Cs-137 irregular planar surface"
DETECTOR_MODEL_ID = "h3d_h100_omni"
INTRO_SECONDS = 3.0
OUTRO_SECONDS = 2.5
VIEW_OVERHEAD_JA = "\u4f5c\u696d\u4fef\u77b0"
VIEW_INTERIOR_JA = "\u5185\u90e8\u8996\u70b9"
VIDEO_CREDIT_JA = (
    "\u4e2d\u5cf6\u3055\u3093\u4f5c\u6210\u74b0\u5883  |  "
    "CAD\u5f62\u72b6\u30fb\u885d\u7a81\u3092\u4fdd\u6301\u3057\u305f"
    "\u5b9f\u6a5f\u30e2\u30c7\u30eb\u4f5c\u696d"
)
OVERVIEW_INTRO_JA = "\u5efa\u5c4b\u5168\u4f53\u4fef\u77b0"
OVERVIEW_OUTRO_JA = "\u9664\u67d3\u5f8c\u30fb\u5efa\u5c4b\u5168\u4f53\u4fef\u77b0"


class TelemetryRow(TypedDict):
    frame_index: int
    phase: str
    view_mode: str
    surface_activity_bq: float
    removed_fraction: float
    expected_count_rate_cps: float
    observed_count_rate_cps: float
    dose_rate_usv_h: float
    incident_fluence_rate_m2_s: float
    visible_radiation_paths: int
    visualized_cell_indices: str
    visualized_cell_fluence_fraction: float


@dataclass(frozen=True)
class CellContribution:
    face_index: int
    source_position_world_m: tuple[float, float, float]
    incident_fluence: IncidentParticleFluence


@dataclass
class RadiationVisuals:
    ray_curves: tuple[Any, ...]
    photon_points_attr: Any
    initial_total_fluence_rate_m2_s: float
    initial_maximum_face_fluence_rate_m2_s: float


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [jsonable(item) for item in value]
    return str(value)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def world_position(stage: Any, path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"USD prim is unavailable: {path}")
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    return np.asarray(transform.Transform(Gf.Vec3d()), dtype=np.float64)


def set_camera(
    stage: Any,
    eye_m: tuple[float, float, float],
    target_m: tuple[float, float, float],
    *,
    focal_length_mm: float,
) -> None:
    from omni.kit.viewport.utility import get_active_viewport

    path = "/World/FukushimaVideoCamera"
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(focal_length_mm).Set(focal_length_mm)
    view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*eye_m),
        Gf.Vec3d(*target_m),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    xform = UsdGeom.Xformable(camera)
    xform.ClearXformOpOrder()
    xform.MakeMatrixXform().Set(view.GetInverse())
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("active viewport is unavailable")
    viewport.set_active_camera(path)


def whole_building_camera(stage: Any) -> dict[str, tuple[float, float, float] | float]:
    environment = stage.GetPrimAtPath("/World/Environment")
    bounds = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        .ComputeWorldBound(environment)
        .ComputeAlignedRange()
    )
    lower = np.asarray(bounds.GetMin(), dtype=np.float64)
    upper = np.asarray(bounds.GetMax(), dtype=np.float64)
    center = 0.5 * (lower + upper)
    extent = upper - lower
    radius = float(np.linalg.norm(extent) * 0.5)
    eye = center + radius * np.asarray((2.40, -2.80, 1.70), dtype=np.float64)
    target = center + np.asarray((0.0, 0.0, -0.08 * extent[2]), dtype=np.float64)
    return {
        "eye": tuple(map(float, eye)),
        "target": tuple(map(float, target)),
        "focal_length_mm": 30.0,
        "lower": tuple(map(float, lower)),
        "upper": tuple(map(float, upper)),
    }


def set_work_area_cutaway(stage: Any, *, enabled: bool) -> tuple[dict[str, Any], ...]:
    """Apply a translucent render-only shell while retaining the CAD and its collision."""

    audited: list[dict[str, Any]] = []
    material = UsdShade.Material.Define(stage, "/World/FukushimaVideoLooks/CutawayShell")
    shader = UsdShade.Shader.Define(
        stage,
        "/World/FukushimaVideoLooks/CutawayShell/PreviewSurface",
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.30, 0.38, 0.44))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.72)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.13 if enabled else 1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Mesh) or "/tn__Structure_01_" not in path:
            continue
        collision = prim.GetAttribute("physics:collisionEnabled")
        collision_before = (
            bool(collision.Get()) if collision and collision.HasAuthoredValueOpinion() else None
        )
        imageable = UsdGeom.Imageable(prim)
        imageable.MakeVisible()
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )
        collision_after = (
            bool(collision.Get()) if collision and collision.HasAuthoredValueOpinion() else None
        )
        audited.append(
            {
                "path": path,
                "render_visibility": "translucent_cutaway" if enabled else "opaque_overview",
                "collision_enabled_before": collision_before,
                "collision_enabled_after": collision_after,
                "collision_preserved": collision_before == collision_after,
            }
        )
    if not audited:
        raise RuntimeError("the Fukushima CAD building shell could not be found for cutaway view")
    return tuple(audited)


class CameraDirector:
    """Keep action cameras inside the catalog-authored Fukushima work area."""

    def __init__(
        self,
        stage: Any,
        inspection_eye_m: tuple[float, float, float],
        inspection_target_m: tuple[float, float, float],
        source_position_m: tuple[float, float, float],
    ) -> None:
        self.stage = stage
        self.inspection_eye_m = np.asarray(inspection_eye_m, dtype=np.float64)
        self.inspection_target_m = np.asarray(inspection_target_m, dtype=np.float64)
        self.source_position_m = np.asarray(source_position_m, dtype=np.float64)
        self.phase = "interior overview"
        self.progress = 0.0
        self.view_mode = VIEW_OVERHEAD_JA

    def set_phase(self, phase: str, progress: float = 0.0) -> None:
        self.phase = phase
        self.progress = float(np.clip(progress, 0.0, 1.0))

    def update(self) -> None:
        eye = self.inspection_eye_m.copy()
        target = self.inspection_target_m.copy()
        focal = 27.0
        overhead = self.phase in {
            "interior overview",
            "measurement positioning",
            "countermeasure transit",
        } or (self.phase == "contact decontamination" and 0.25 <= self.progress < 0.50)
        if overhead:
            countermeasure = world_position(self.stage, "/World/CountermeasureRobot")
            measurement = world_position(self.stage, "/World/MeasurementRobot/chassis_link")
            target = (self.source_position_m + countermeasure + measurement) / 3.0
            target[2] = 0.85
            eye = target + np.asarray((-4.65, -5.35, 5.85))
            focal = 32.0
            self.view_mode = VIEW_OVERHEAD_JA
        elif self.phase in {"tool approach", "contact decontamination"}:
            eye = self.source_position_m + np.asarray((-3.95, -3.75, 2.15))
            target = self.source_position_m + np.asarray((-0.35, 0.0, 0.0))
            focal = 31.0
            self.view_mode = VIEW_INTERIOR_JA
        elif self.phase == "completion":
            eye = self.source_position_m + np.asarray((-4.35, 3.55, 2.65))
            target = self.source_position_m + np.asarray((-0.50, 0.0, -0.05))
            focal = 29.0
            self.view_mode = VIEW_INTERIOR_JA
        set_camera(
            self.stage,
            tuple(map(float, eye)),
            tuple(map(float, target)),
            focal_length_mm=focal,
        )


def author_operation_guides(
    stage: Any,
    *,
    ground_primary: tuple[float, float, float],
    ground_secondary: tuple[float, float, float],
    source_position: tuple[float, float, float],
    measurement_target: tuple[float, float],
) -> None:
    """Make the two real routes readable in the high-oblique work-area shots."""

    root = UsdGeom.Xform.Define(stage, "/World/OperationGuides").GetPrim()
    root.CreateAttribute("rad:role", Sdf.ValueTypeNames.String, custom=True).Set(
        "operator_explanation_overlay"
    )
    root.CreateAttribute("rad:affectsPhysics", Sdf.ValueTypeNames.Bool, custom=True).Set(False)
    route_specs = (
        (
            "H100MeasurementRoute",
            (
                Gf.Vec3f(ground_secondary[0], ground_secondary[1], ground_secondary[2] + 0.035),
                Gf.Vec3f(measurement_target[0], measurement_target[1], ground_secondary[2] + 0.035),
            ),
            Gf.Vec3f(0.05, 0.55, 1.0),
        ),
        (
            "DecontaminationRobotRoute",
            (
                Gf.Vec3f(ground_primary[0], ground_primary[1], ground_primary[2] + 0.045),
                Gf.Vec3f(source_position[0] - 0.98, source_position[1] - 0.55, 0.045),
                Gf.Vec3f(source_position[0] - 0.90, source_position[1], 0.045),
            ),
            Gf.Vec3f(1.0, 0.42, 0.04),
        ),
    )
    marker_points: list[Gf.Vec3f] = []
    marker_colors: list[Gf.Vec3f] = []
    for name, points, color in route_specs:
        curve = UsdGeom.BasisCurves.Define(stage, f"/World/OperationGuides/{name}")
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateCurveVertexCountsAttr([len(points)])
        curve.CreatePointsAttr(list(points))
        curve.CreateWidthsAttr([0.035])
        curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        curve.CreateDisplayColorAttr([color])
        curve.CreateDisplayOpacityAttr([0.76])
        marker_points.extend(points)
        marker_colors.extend((color,) * len(points))
    markers = UsdGeom.Points.Define(stage, "/World/OperationGuides/RouteEndpoints")
    markers.CreatePointsAttr(marker_points)
    markers.CreateWidthsAttr([0.16] * len(marker_points))
    markers.CreateDisplayColorAttr(marker_colors)
    markers.SetWidthsInterpolation(UsdGeom.Tokens.vertex)


def author_cad_aligned_robot_support(
    stage: Any,
    *,
    source_position: tuple[float, float, float],
) -> dict[str, Any]:
    """Bridge the CAD shell's one-sided lower triangles for free wheeled bodies."""

    path = "/World/CADAlignedRobotSupport"
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(
        Gf.Vec3d(float(source_position[0] - 1.0), float(source_position[1]), -0.055)
    )
    cube.AddScaleOp().Set(Gf.Vec3f(3.4, 3.2, 0.05))
    collision = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    collision.CreateCollisionEnabledAttr(True)
    UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
    cube.GetPrim().CreateAttribute("rad:role", Sdf.ValueTypeNames.String, custom=True).Set(
        "cad_aligned_physics_support"
    )
    cube.GetPrim().CreateAttribute("rad:renderOnly", Sdf.ValueTypeNames.Bool, custom=True).Set(
        False
    )
    cube.GetPrim().CreateAttribute(
        "rad:environmentReplacement", Sdf.ValueTypeNames.Bool, custom=True
    ).Set(False)
    return {
        "path": path,
        "visible": False,
        "collision_enabled": True,
        "purpose": "support free wheeled bodies across one-sided CAD lower-shell triangles",
        "environment_geometry_replaced": False,
    }


def capture_and_wait(app: Any, world: Any, path: Path) -> None:
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("active viewport is unavailable")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    capture_viewport_to_file(viewport, file_path=str(path), is_hdr=False)
    deadline = time.monotonic() + 12.0
    while (not path.is_file() or path.stat().st_size == 0) and time.monotonic() < deadline:
        world.step(render=True)
        time.sleep(0.02)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"viewport capture did not complete: {path}")


def face_centers_world(stage: Any, mesh_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(mesh_path)
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if len(counts) == 0 or np.any(counts != 3):
        raise RuntimeError("the visible contamination source must remain triangulated")
    local_centers = points[indices.reshape(-1, 3)].mean(axis=1)
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    return np.asarray(
        [transform.Transform(Gf.Vec3d(*center)) for center in local_centers],
        dtype=np.float64,
    )


def cell_contributions(
    face_centers_m: np.ndarray,
    activity_bq: np.ndarray,
    detector_position_m: np.ndarray,
) -> tuple[CellContribution, ...]:
    contributions: list[CellContribution] = []
    for face_index, (source, activity) in enumerate(zip(face_centers_m, activity_bq, strict=True)):
        if activity <= 0.0:
            continue
        travel = detector_position_m - source
        distance_squared_m2 = float(np.dot(travel, travel))
        if distance_squared_m2 <= 1e-12:
            continue
        direction = travel / math.sqrt(distance_squared_m2)
        contributions.append(
            CellContribution(
                face_index=face_index,
                source_position_world_m=tuple(map(float, source)),
                incident_fluence=IncidentParticleFluence(
                    radiation_type=RadiationType.GAMMA,
                    energy_kev=CS137_GAMMA_ENERGY_KEV,
                    fluence_rate_m2_s=(
                        float(activity) * CS137_GAMMA_YIELD / (4.0 * math.pi * distance_squared_m2)
                    ),
                    arrival_direction_world=tuple(map(float, direction)),
                    source_id=f"fukushima_planar_face_{face_index:04d}_cs137",
                ),
            )
        )
    return tuple(contributions)


def author_h100_metadata(stage: Any, detector_path: str) -> None:
    prim = stage.GetPrimAtPath(detector_path)
    for name, value_type, value in (
        ("rad:detector:model", Sdf.ValueTypeNames.String, DETECTOR_MODEL_ID),
        ("rad:detector:manufacturer", Sdf.ValueTypeNames.String, "H3D, Inc."),
        ("rad:detector:directionality", Sdf.ValueTypeNames.String, "omnidirectional"),
        ("rad:detector:radiationFovSr", Sdf.ValueTypeNames.Double, 4.0 * math.pi),
        ("rad:detector:responseDataStatus", Sdf.ValueTypeNames.String, "synthetic_validation_only"),
    ):
        prim.CreateAttribute(name, value_type, custom=True).Set(value)
    existing = stage.GetPrimAtPath(detector_path + "/GammaDetector")
    if existing and existing.IsValid():
        UsdGeom.Imageable(existing).MakeInvisible()
    body = UsdGeom.Cube.Define(stage, detector_path + "/H100Body")
    body.CreateSizeAttr(1.0)
    body.AddScaleOp().Set(Gf.Vec3f(0.244, 0.086, 0.175))
    body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    body.CreateDisplayColorAttr([Gf.Vec3f(0.94, 0.58, 0.04)])


def create_radiation_visuals(
    stage: Any,
    contributions: tuple[CellContribution, ...],
    detector_position_m: np.ndarray,
) -> RadiationVisuals:
    root_path = "/World/H100ContributionVisualization"
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    for name, value_type, value in (
        ("rad:role", Sdf.ValueTypeNames.String, "radiation_visualization"),
        (
            "rad:visualization:kind",
            Sdf.ValueTypeNames.String,
            "contribution_weighted_gamma_paths",
        ),
        ("rad:visualization:transportCoupled", Sdf.ValueTypeNames.Bool, True),
        ("rad:visualization:affectsDetectorResponse", Sdf.ValueTypeNames.Bool, False),
        (
            "rad:visualization:samplingMethod",
            Sdf.ValueTypeNames.String,
            "deterministic_gumbel_top_k_without_replacement",
        ),
        ("rad:visualization:samplingSeed", Sdf.ValueTypeNames.Int, RADIATION_SAMPLE_SEED),
        ("rad:visualization:sourcePath", Sdf.ValueTypeNames.String, "/World/DeconWorkSurface"),
        (
            "rad:visualization:detectorPath",
            Sdf.ValueTypeNames.String,
            "/World/MeasurementRobot/chassis_link/Detector",
        ),
    ):
        root.CreateAttribute(name, value_type, custom=True).Set(value)
    total = sum(item.incident_fluence.fluence_rate_m2_s for item in contributions)
    maximum = max(
        (item.incident_fluence.fluence_rate_m2_s for item in contributions),
        default=0.0,
    )
    ray_curves = []
    for ray_index in range(min(28, len(contributions))):
        curve = UsdGeom.BasisCurves.Define(stage, f"{root_path}/Paths/Ray_{ray_index:02d}")
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateCurveVertexCountsAttr([2])
        curve.CreatePointsAttr(
            [
                Gf.Vec3f(*contributions[ray_index].source_position_world_m),
                Gf.Vec3f(*tuple(map(float, detector_position_m))),
            ]
        )
        curve.CreateWidthsAttr([0.012])
        curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        curve.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.34, 0.015)])
        curve.CreateDisplayOpacityAttr([0.26])
        curve.GetPrim().CreateAttribute(
            "rad:visualization:sourceFaceIndex", Sdf.ValueTypeNames.Int, custom=True
        ).Set(contributions[ray_index].face_index)
        curve.GetPrim().CreateAttribute(
            "rad:visualization:faceFluenceRateM2S", Sdf.ValueTypeNames.Double, custom=True
        ).Set(contributions[ray_index].incident_fluence.fluence_rate_m2_s)
        ray_curves.append(curve)
    photons = UsdGeom.Points.Define(stage, f"{root_path}/GammaPhotons")
    photon_points_attr = photons.CreatePointsAttr([])
    photons.CreateWidthsAttr([0.075])
    photons.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    photons.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.80, 0.04)])
    photons.CreateDisplayOpacityAttr([0.94])
    return RadiationVisuals(tuple(ray_curves), photon_points_attr, total, maximum)


def update_radiation_visuals(
    visuals: RadiationVisuals,
    contributions: tuple[CellContribution, ...],
    face_count: int,
    detector_position_m: np.ndarray,
    phase_s: float,
) -> tuple[int, tuple[int, ...], float, float]:
    weights = np.zeros(face_count, dtype=np.float64)
    by_index: dict[int, CellContribution] = {}
    for item in contributions:
        weights[item.face_index] = item.incident_fluence.fluence_rate_m2_s
        by_index[item.face_index] = item
    total = float(np.sum(weights))
    remaining = float(
        np.clip(total / max(visuals.initial_total_fluence_rate_m2_s, 1e-12), 0.0, 1.0)
    )
    visible_count = (
        min(len(contributions), max(2, int(round(len(visuals.ray_curves) * remaining**0.78))))
        if contributions
        else 0
    )
    selected = contribution_weighted_sample_without_replacement(
        weights,
        visible_count,
        seed=RADIATION_SAMPLE_SEED,
    )
    selected_sources: list[np.ndarray] = []
    for ray_index, curve in enumerate(visuals.ray_curves):
        imageable = UsdGeom.Imageable(curve.GetPrim())
        if ray_index >= len(selected):
            imageable.MakeInvisible()
            continue
        item = by_index[selected[ray_index]]
        source = np.asarray(item.source_position_world_m, dtype=np.float64)
        selected_sources.append(source)
        relative = float(
            np.clip(
                item.incident_fluence.fluence_rate_m2_s
                / max(visuals.initial_maximum_face_fluence_rate_m2_s, 1e-12),
                0.0,
                1.0,
            )
        )
        imageable.MakeVisible()
        curve.GetPointsAttr().Set(
            [
                Gf.Vec3f(*tuple(map(float, source))),
                Gf.Vec3f(*tuple(map(float, detector_position_m))),
            ]
        )
        curve.GetWidthsAttr().Set([0.008 + 0.018 * math.sqrt(relative)])
        curve.GetDisplayOpacityAttr().Set([0.06 + 0.34 * math.sqrt(relative)])
        curve.GetPrim().GetAttribute("rad:visualization:sourceFaceIndex").Set(item.face_index)
        curve.GetPrim().GetAttribute("rad:visualization:faceFluenceRateM2S").Set(
            item.incident_fluence.fluence_rate_m2_s
        )
    photon_positions = []
    for ray_index, source in enumerate(selected_sources):
        for pulse_index in range(2):
            fraction = (phase_s * 0.70 + ray_index * 0.137 + pulse_index * 0.5) % 1.0
            position = source + fraction * (detector_position_m - source)
            photon_positions.append(Gf.Vec3f(*tuple(map(float, position))))
    visuals.photon_points_attr.Set(photon_positions)
    selected_total = float(np.sum(weights[list(selected)])) if selected else 0.0
    return (
        len(selected),
        selected,
        total,
        selected_total / total if total > 0.0 else 0.0,
    )


class FrameRecorder:
    def __init__(
        self,
        output: Path,
        *,
        stride: int,
        stage: Any,
        director: CameraDirector,
        decontaminator: Any,
        detector_path: str,
        face_centers_m: np.ndarray,
        detector_model: ParametricDetectorModel,
        radiation_visuals: RadiationVisuals,
        seed: int,
    ) -> None:
        from omni.kit.viewport.utility import get_active_viewport

        self.frame_dir = output / ".action-frames"
        if self.frame_dir.exists():
            shutil.rmtree(self.frame_dir)
        self.frame_dir.mkdir(parents=True)
        self.viewport = get_active_viewport()
        if self.viewport is None:
            raise RuntimeError("active viewport is unavailable")
        self.stride = stride
        self.stage = stage
        self.director = director
        self.decontaminator = decontaminator
        self.detector_path = detector_path
        self.face_centers_m = face_centers_m
        self.detector_model = detector_model
        self.radiation_visuals = radiation_visuals
        self.seed = seed
        self.physics_steps = 0
        self.frame_count = 0
        self.telemetry: list[TelemetryRow] = []

    def observe(self) -> None:
        from omni.kit.viewport.utility import capture_viewport_to_file

        self.physics_steps += 1
        if self.physics_steps % self.stride:
            return
        detector_position = world_position(self.stage, self.detector_path)
        contributions = cell_contributions(
            self.face_centers_m,
            self.decontaminator.activity_bq,
            detector_position,
        )
        visible, selected, fluence, selected_fraction = update_radiation_visuals(
            self.radiation_visuals,
            contributions,
            len(self.decontaminator.activity_bq),
            detector_position,
            self.frame_count / max(ARGS.capture_fps, 1),
        )
        reading = self.detector_model.measure(
            MeasurementRequest(
                pose=DetectorPose(
                    "fukushima_h100",
                    tuple(map(float, detector_position)),
                    forward_world=(1.0, 0.0, 0.0),
                ),
                incident_fluence=tuple(item.incident_fluence for item in contributions),
                integration_time_s=1.0,
                rng=np.random.default_rng(self.seed + self.frame_count),
            )
        )
        initial = float(np.sum(self.decontaminator.initial_activity_bq))
        current = float(np.sum(self.decontaminator.activity_bq))
        self.telemetry.append(
            {
                "frame_index": self.frame_count,
                "phase": self.director.phase,
                "view_mode": self.director.view_mode,
                "surface_activity_bq": current,
                "removed_fraction": max(0.0, (initial - current) / initial),
                "expected_count_rate_cps": reading.expected_count_rate_cps,
                "observed_count_rate_cps": (reading.observed_counts / reading.integration_time_s),
                "dose_rate_usv_h": reading.dose_rate_usv_h,
                "incident_fluence_rate_m2_s": fluence,
                "visible_radiation_paths": visible,
                "visualized_cell_indices": ";".join(map(str, selected)),
                "visualized_cell_fluence_fraction": selected_fraction,
            }
        )
        path = self.frame_dir / f"frame_{self.frame_count:04d}.png"
        capture_viewport_to_file(self.viewport, file_path=str(path), is_hdr=False)
        self.frame_count += 1

    def wait_for_frames(self, world: Any) -> None:
        deadline = time.monotonic() + 15.0
        found = len(list(self.frame_dir.glob("frame_*.png")))
        while found < self.frame_count and time.monotonic() < deadline:
            world.step(render=True)
            time.sleep(0.02)
            found = len(list(self.frame_dir.glob("frame_*.png")))
        if found != self.frame_count:
            raise RuntimeError(f"expected {self.frame_count} action frames, found {found}")


class RecordingStepper:
    def __init__(self, world: Any, director: CameraDirector) -> None:
        self.world = world
        self.director = director
        self.recorder: FrameRecorder | None = None
        self.before_step: Any | None = None

    def step(self, *, render: bool = False) -> None:
        del render
        if self.before_step is not None:
            self.before_step()
        self.director.update()
        self.world.step(render=True)
        if self.recorder is not None:
            self.recorder.observe()


def ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100.0)))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def ass_escape(text: str) -> str:
    return text.replace("{", r"\{").replace("}", r"\}")


def write_telemetry_csv(rows: list[TelemetryRow], path: Path, action_seconds: float) -> None:
    fieldnames = [
        "video_time_s",
        "phase",
        "view_mode",
        "surface_activity_bq",
        "removed_fraction",
        "expected_count_rate_cps",
        "observed_count_rate_cps",
        "dose_rate_usv_h",
        "incident_fluence_rate_m2_s",
        "visible_radiation_paths",
        "visualized_cell_indices",
        "visualized_cell_fluence_fraction",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            action_time = row["frame_index"] / max(len(rows), 1) * action_seconds
            writer.writerow(
                {
                    "video_time_s": INTRO_SECONDS + action_time,
                    **{key: row[key] for key in fieldnames if key != "video_time_s"},
                }
            )


def write_hud(
    rows: list[TelemetryRow],
    path: Path,
    *,
    duration_s: float,
    action_seconds: float,
) -> None:
    initial = rows[0]
    final = rows[-1]
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Title,Noto Sans CJK JP,25,&H00FFFFFF,&H00FFFFFF,&H00121920,&H9A000000,-1,0,0,0,100,100,0,0,1,1.2,0,7,30,30,22,1
Style: Sub,Noto Sans CJK JP,15,&H00DDE7EA,&H00FFFFFF,&H00121920,&H9A000000,0,0,0,0,100,100,0,0,1,1.0,0,7,30,30,56,1
Style: Label,Noto Sans Mono CJK JP,13,&H00B7C8CF,&H00FFFFFF,&H00121920,&H9A000000,0,0,0,0,100,100,0,0,1,0.8,0,7,30,30,92,1
Style: Value,Noto Sans Mono CJK JP,15,&H003FD7FF,&H00FFFFFF,&H00121920,&H9A000000,-1,0,0,0,100,100,0,0,1,0.8,0,9,30,30,92,1
Style: Phase,Noto Sans CJK JP,18,&H00FFFFFF,&H00FFFFFF,&H00121920,&H9A000000,-1,0,0,0,100,100,0,0,1,1.0,0,1,30,30,34,1
Style: Map,Noto Sans CJK JP,12,&H00FFFFFF,&H00FFFFFF,&H00121920,&H9A000000,-1,0,0,0,100,100,0,0,1,0.8,0,3,30,22,198,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header.rstrip("\n")]

    def event(layer: int, start: float, end: float, style: str, text: str) -> None:
        lines.append(
            f"Dialogue: {layer},{ass_timestamp(start)},{ass_timestamp(end)},{style},,0,0,0,,"
            + ass_escape(text)
        )

    event(0, 0.0, duration_s, "Title", "FUKUSHIMA DAIICHI BUILDING CAD")
    event(0, 0.0, duration_s, "Sub", VIDEO_CREDIT_JA)
    event(
        0,
        0.0,
        duration_s,
        "Label",
        "SURFACE ACTIVITY\\NREMOVED\\NH100 SURFACE CPS\\NDOSE RATE\\NVISIBLE CONTRIBUTION PATHS",
    )
    event(0, 0.0, INTRO_SECONDS, "Phase", f"{OVERVIEW_INTRO_JA}  /  IMPORTED CAD OVERVIEW")
    event(
        0,
        duration_s - OUTRO_SECONDS,
        duration_s,
        "Phase",
        f"{OVERVIEW_OUTRO_JA}  /  POST-DECONTAMINATION OVERVIEW",
    )
    event(
        2,
        INTRO_SECONDS,
        duration_s - OUTRO_SECONDS,
        "Map",
        "CAD OVERVIEW  /  WORK AREA: LOWER CUT FACE",
    )
    update_seconds = 0.50
    video_rows: list[tuple[float, TelemetryRow]] = [(0.0, initial)]
    for index, row in enumerate(rows):
        action_time = index / max(len(rows), 1) * action_seconds
        video_rows.append((INTRO_SECONDS + action_time, row))
    video_rows.append((duration_s - OUTRO_SECONDS, final))
    for update_start in np.arange(0.0, duration_s, update_seconds):
        applicable = max(
            (item for item in video_rows if item[0] <= update_start + 1e-9),
            key=lambda item: item[0],
        )[1]
        event(
            1,
            float(update_start),
            min(float(update_start + update_seconds), duration_s),
            "Value",
            (
                f"{applicable['surface_activity_bq'] / 1.0e6:10.3f} MBq\\N"
                f"{applicable['removed_fraction'] * 100.0:10.2f} %\\N"
                f"{applicable['expected_count_rate_cps']:10.1f} cps\\N"
                f"{applicable['dose_rate_usv_h']:10.3f} uSv/h\\N"
                f"{applicable['visible_radiation_paths']:10d}"
            ),
        )
    phase_start = INTRO_SECONDS
    current_phase = (rows[0]["view_mode"], rows[0]["phase"])
    for index, row in enumerate(rows[1:], start=1):
        row_phase = (row["view_mode"], row["phase"])
        if row_phase == current_phase:
            continue
        phase_end = INTRO_SECONDS + index / len(rows) * action_seconds
        event(
            0, phase_start, phase_end, "Phase", f"{current_phase[0]}  /  {current_phase[1].upper()}"
        )
        phase_start = phase_end
        current_phase = row_phase
    event(
        0,
        phase_start,
        duration_s - OUTRO_SECONDS,
        "Phase",
        f"{current_phase[0]}  /  {current_phase[1].upper()}",
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ffmpeg_filter_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def encode_video(
    output: Path,
    frame_dir: Path,
    frame_count: int,
    intro_path: Path,
    outro_path: Path,
    subtitle_path: Path,
    *,
    duration_s: float,
) -> Path:
    action_seconds = duration_s - INTRO_SECONDS - OUTRO_SECONDS
    action_fps = frame_count / action_seconds
    video_path = output / "fukushima_cad_articulated_decontamination_24s.mp4"
    video_path.unlink(missing_ok=True)
    common = "crop=1440:810:0:45,scale=1280:720,setsar=1"
    filter_graph = (
        "[0:v]split=2[intro_src][locator_src];"
        f"[intro_src]{common},fps=30,setpts=PTS-STARTPTS[v0];"
        "[locator_src]scale=280:175,setsar=1,"
        "drawbox=x=0:y=0:w=iw:h=ih:color=0xF59E0B@0.92:t=3[locator];"
        f"[1:v]{common},setpts=PTS-STARTPTS[action];"
        "[action][locator]overlay=x=W-w-22:y=H-h-20:format=auto[v1];"
        f"[2:v]{common},fps=30,setpts=PTS-STARTPTS[v2];"
        "[v0][v1][v2]concat=n=3:v=1:a=0[joined];"
        f"[joined]subtitles=filename='{ffmpeg_filter_path(subtitle_path)}',format=yuv420p[out]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-t",
            f"{INTRO_SECONDS:.6f}",
            "-i",
            str(intro_path),
            "-framerate",
            f"{action_fps:.9f}",
            "-i",
            str(frame_dir / "frame_%04d.png"),
            "-loop",
            "1",
            "-t",
            f"{OUTRO_SECONDS:.6f}",
            "-i",
            str(outro_path),
            "-filter_complex",
            filter_graph,
            "-map",
            "[out]",
            "-t",
            f"{duration_s:.6f}",
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(video_path),
        ],
        check=True,
    )
    return video_path


def prepare_scene() -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.robot.wheeled_robots.robots import WheeledRobot
    from radcounter.isaac.robot import (
        NovaCarterController,
        RealRobotAssetConfig,
        add_real_robot_references,
        author_real_robot_task_scene,
        create_decontamination_activity_map,
        enable_real_robot_extensions,
    )
    from radcounter.isaac.system_profile import prepare_environment_stage

    from radcounter.core.system_profiles import resolve_system_selection

    selection = resolve_system_selection(
        catalog_path=ROOT / "configs/system/catalog.yaml",
        environment_id="fukushima-daiichi",
        robot_set_id="articulated-decommissioning",
        detector_set_id="vertical-slice-gamma",
    )
    enable_real_robot_extensions()
    for _ in range(20):
        APP.update()
    stage_path, environment_manifest = prepare_environment_stage(selection)
    context = omni.usd.get_context()
    if not context.open_stage(str(stage_path)):
        raise RuntimeError(f"failed to open imported Fukushima stage: {stage_path}")
    for _ in range(36):
        APP.update()
    stage = context.get_stage()
    decon_anchor = selection.spawn_anchor("decon-surface")
    ground_primary = selection.spawn_anchor("ground-primary")
    ground_secondary = selection.spawn_anchor("ground-secondary")
    camera_eye = selection.spawn_anchor("inspection-camera-eye")
    camera_target = selection.spawn_anchor("inspection-camera-target")
    config = replace(
        RealRobotAssetConfig(),
        decon_workbench_center_m=decon_anchor.translation_m,
        shield_initial_position_m=(0.0, 3.0, ground_primary.translation_m[2]),
        include_validation_facility=False,
    )
    assets = add_real_robot_references(stage, config=config)
    for _ in range(140):
        APP.update()
    activity_path = create_decontamination_activity_map(
        ARGS.output / "fukushima_planar_surface_activity.npz"
    )
    author_real_robot_task_scene(
        stage,
        activity_path,
        ROOT / "configs/decontamination/concrete_surface.synthetic.yaml",
        config=config,
    )
    shield = stage.GetPrimAtPath(config.shield_path)
    if shield.IsValid():
        stage.RemovePrim(config.shield_path)
    author_h100_metadata(stage, config.detector_path)
    support_audit = author_cad_aligned_robot_support(
        stage,
        source_position=decon_anchor.translation_m,
    )
    dome = UsdLux.DomeLight.Define(stage, "/World/FukushimaVideoLighting/Dome")
    dome.CreateIntensityAttr(720.0)
    dome.CreateColorAttr(Gf.Vec3f(0.70, 0.78, 0.88))
    distant = UsdLux.DistantLight.Define(stage, "/World/FukushimaVideoLighting/Key")
    distant.CreateIntensityAttr(1900.0)
    distant.CreateAngleAttr(0.55)
    distant.AddRotateXYZOp().Set(Gf.Vec3f(42.0, -28.0, -34.0))
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    franka_articulation = world.scene.add(
        SingleArticulation(config.countermeasure_root, name="fukushima_ridgeback_franka")
    )
    carter_articulation = world.scene.add(
        WheeledRobot(
            prim_path=config.measurement_articulation,
            name="fukushima_nova_carter",
            wheel_dof_names=list(NovaCarterController.wheel_names),
        )
    )
    world.reset()
    before_settle = world_position(stage, config.panda_base_path)
    for _ in range(90):
        world.step(render=False)
    after_settle = world_position(stage, config.panda_base_path)
    manifest = {
        "selection": selection.as_dict(),
        "stage": str(stage_path),
        "environment_manifest": str(environment_manifest),
        "assets": assets,
        "anchors": {
            "ground_primary": ground_primary.translation_m,
            "ground_secondary": ground_secondary.translation_m,
            "decon_surface": decon_anchor.translation_m,
            "inspection_camera_eye": camera_eye.translation_m,
            "inspection_camera_target": camera_target.translation_m,
        },
        "spawn_vertical_drop_m": max(0.0, float(before_settle[2] - after_settle[2])),
        "cad_aligned_robot_support": support_audit,
    }
    return (
        stage,
        world,
        (franka_articulation, carter_articulation),
        config,
        manifest,
    )


def run() -> dict[str, Any]:
    from isaacsim.core.utils.types import ArticulationAction
    from radcounter.isaac.robot import (
        ContactDrivenDecontaminator,
        DecontaminationConfig,
        NovaCarterController,
        RidgebackFrankaController,
    )

    ARGS.output.mkdir(parents=True, exist_ok=True)
    stage, world, articulations, config, manifest = prepare_scene()
    franka_articulation, carter_articulation = articulations
    startup_arm_indices = np.asarray(
        [
            franka_articulation.get_dof_index(f"panda_joint{joint_index}")
            for joint_index in range(1, 8)
        ],
        dtype=np.int32,
    )
    startup_arm_positions = np.asarray(franka_articulation.get_joint_positions(), dtype=np.float64)[
        startup_arm_indices
    ].copy()
    startup_controller = franka_articulation.get_articulation_controller()
    startup_controller.switch_control_mode(mode="position")
    startup_controller.apply_action(
        ArticulationAction(
            joint_positions=startup_arm_positions,
            joint_indices=startup_arm_indices,
        )
    )
    overview = whole_building_camera(stage)
    set_camera(
        stage,
        overview["eye"],
        overview["target"],
        focal_length_mm=float(overview["focal_length_mm"]),
    )
    for _ in range(90):
        world.step(render=True)
    intro_path = ARGS.output / "01_fukushima_building_overview.png"
    capture_and_wait(APP, world, intro_path)
    cutaway_audit = set_work_area_cutaway(stage, enabled=True)

    anchors = manifest["anchors"]
    initial_monitor = np.asarray(anchors["ground_secondary"], dtype=np.float64)
    carter_articulation.set_world_pose(
        position=initial_monitor,
        orientation=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64),
    )
    carter_articulation.set_linear_velocity(np.zeros(3, dtype=np.float64))
    carter_articulation.set_angular_velocity(np.zeros(3, dtype=np.float64))
    source_position = np.asarray(anchors["decon_surface"], dtype=np.float64)
    monitor_radius = float(np.linalg.norm(initial_monitor[:2] - source_position[:2]))
    target_y_offset = min(1.45, monitor_radius * 0.80)
    target_x_offset = math.sqrt(max(monitor_radius**2 - target_y_offset**2, 0.0))
    measurement_target = (
        float(source_position[0] - target_x_offset),
        float(source_position[1] + target_y_offset),
    )
    author_operation_guides(
        stage,
        ground_primary=anchors["ground_primary"],
        ground_secondary=anchors["ground_secondary"],
        source_position=anchors["decon_surface"],
        measurement_target=measurement_target,
    )
    director = CameraDirector(
        stage,
        anchors["inspection_camera_eye"],
        anchors["inspection_camera_target"],
        anchors["decon_surface"],
    )
    director.update()
    for _ in range(45):
        world.step(render=True)
    interior_path = ARGS.output / "02_internal_work_area.png"
    capture_and_wait(APP, world, interior_path)
    if ARGS.preview_only:
        preview_views: list[str] = []
        preview_target = np.asarray(anchors["inspection_camera_target"], dtype=np.float64)
        for label, offset in (
            ("west_south", (-4.0, -5.0, 3.0)),
            ("west_north", (-4.0, 5.0, 3.0)),
            ("east_south", (5.0, -5.0, 3.0)),
            ("east_north", (5.0, 5.0, 3.0)),
            ("south_high", (0.0, -6.0, 5.0)),
            ("north_high", (0.0, 6.0, 5.0)),
            ("west", (-6.0, 0.0, 3.5)),
            ("east", (6.0, 0.0, 3.5)),
        ):
            eye = preview_target + np.asarray(offset, dtype=np.float64)
            set_camera(
                stage,
                tuple(map(float, eye)),
                tuple(map(float, preview_target)),
                focal_length_mm=28.0,
            )
            for _ in range(18):
                world.step(render=True)
            preview_path = ARGS.output / f"candidate_{label}.png"
            capture_and_wait(APP, world, preview_path)
            preview_views.append(str(preview_path))
        return {
            "passed": True,
            "preview_only": True,
            **manifest,
            "overview_camera": overview,
            "overview_image": str(intro_path),
            "interior_image": str(interior_path),
            "candidate_internal_views": preview_views,
            "render_cutaway_audit": cutaway_audit,
        }

    stepper = RecordingStepper(world, director)
    countermeasure = RidgebackFrankaController(
        stage,
        stepper,
        config=config,
        articulation=franka_articulation,
    )
    measurement = NovaCarterController(
        stage,
        stepper,
        config=config,
        articulation=carter_articulation,
    )
    measurement.set_initial_pose(anchors["ground_secondary"])
    decontaminator = ContactDrivenDecontaminator(
        stage,
        config.decon_tool_path,
        config.decon_surface_path,
        DecontaminationConfig(
            footprint_points_local_m=tuple(
                (float(local_x), float(local_y), 0.0)
                for local_x in np.linspace(-0.085, 0.085, 9)
                for local_y in np.linspace(-0.065, 0.065, 9)
            ),
            treatment_axis_local=(0.0, 0.0, 1.0),
            max_contact_distance_m=0.045,
            max_surface_speed_m_s=0.35,
            # The imported CAD has no authored waste-storage zone. Keep the
            # removed quantity in the motion/activity audit rather than
            # overlaying the vertical-slice disposal facility on this CAD.
            transfer_mode="discard",
        ),
    )
    centers = face_centers_world(stage, config.decon_surface_path)
    detector_position = world_position(stage, config.detector_path)
    initial_contributions = cell_contributions(
        centers,
        decontaminator.activity_bq,
        detector_position,
    )
    radiation_visuals = create_radiation_visuals(stage, initial_contributions, detector_position)
    detector_model = ParametricDetectorModel(popular_detector_catalog()[DETECTOR_MODEL_ID])
    recorder = FrameRecorder(
        ARGS.output,
        stride=ARGS.capture_stride,
        stage=stage,
        director=director,
        decontaminator=decontaminator,
        detector_path=config.detector_path,
        face_centers_m=centers,
        detector_model=detector_model,
        radiation_visuals=radiation_visuals,
        seed=ARGS.seed,
    )
    stepper.recorder = recorder

    def measurement_progress(
        _step: int,
        _position: tuple[float, float, float],
        _target: tuple[float, float],
        _distance: float,
    ) -> None:
        director.set_phase("measurement positioning")

    def countermeasure_progress(event: Any) -> None:
        phase = str(event.get("phase", ""))
        progress = float(event.get("progress", 0.0))
        if phase in {"approaching", "contact_confirmed"}:
            director.set_phase("tool approach", progress)
        elif phase == "decontaminating":
            director.set_phase("contact decontamination", progress)
        elif phase in {"complete", "failed"} and "decontaminating" in countermeasure.trace:
            director.set_phase("completion", progress)
        else:
            director.set_phase("countermeasure transit", progress)

    measurement.progress_callback = measurement_progress
    countermeasure.progress_callback = countermeasure_progress
    # Nova Carter needs hundreds of free-rolling physics steps to reach its
    # monitoring point.  Retain the Franka's proven catalog startup posture
    # during that interval instead of letting gravity alter the IK seed before
    # the contact raster begins.  This is a position-drive hold, not a teleport.
    arm_indices = np.asarray(
        [countermeasure.indices[name] for name in countermeasure.arm_joint_names],
        dtype=np.int32,
    )
    arm_hold_positions = startup_arm_positions

    def hold_franka_startup_posture() -> None:
        countermeasure.controller.apply_action(
            ArticulationAction(
                joint_positions=arm_hold_positions,
                joint_indices=arm_indices,
            )
        )

    stepper.before_step = hold_franka_startup_posture
    initial_countermeasure = world_position(stage, config.panda_base_path)
    initial_measurement = world_position(stage, config.measurement_articulation)
    initial_detector_position = world_position(stage, config.detector_path)
    initial_activity_bq = float(np.sum(decontaminator.activity_bq))

    director.set_phase("interior overview")
    countermeasure.hold(48)
    source_x, source_y, _ = anchors["decon_surface"]
    navigation_report = countermeasure.navigate_route(
        (
            anchors["ground_primary"],
            (source_x - 0.98, source_y - 0.55, anchors["ground_primary"][2]),
        ),
        final_yaw_rad=0.0,
    )
    if not navigation_report.success:
        raise RuntimeError(f"Ridgeback could not reach the CAD work face: {navigation_report}")
    stepper.before_step = None
    stow_report = countermeasure.stow_arm()
    if not stow_report.success:
        raise RuntimeError(f"Franka could not assume the work-start posture: {stow_report}")
    held_names = countermeasure.base_joint_names + countermeasure.arm_joint_names
    held_indices = np.asarray(
        [countermeasure.indices[name] for name in held_names],
        dtype=np.int32,
    )
    held_positions = np.asarray(franka_articulation.get_joint_positions(), dtype=np.float64)[
        held_indices
    ].copy()

    def hold_countermeasure_at_work_face() -> None:
        countermeasure.controller.apply_action(
            ArticulationAction(
                joint_positions=held_positions,
                joint_indices=held_indices,
            )
        )

    stepper.before_step = hold_countermeasure_at_work_face
    measurement_report = measurement.navigate_to(measurement_target)
    if not measurement_report.success:
        raise RuntimeError(f"Nova Carter could not reach the monitoring pose: {measurement_report}")
    stepper.before_step = None
    work_pose_report = countermeasure.navigate_to(
        (source_x - 0.90, source_y, anchors["ground_primary"][2]),
        target_yaw_rad=0.0,
    )
    if not work_pose_report.success:
        raise RuntimeError(f"Ridgeback work-pose resynchronization failed: {work_pose_report}")
    decon_report = countermeasure.execute_surface_decontamination(decontaminator, 1.5)
    if not decon_report.success:
        raise RuntimeError(f"contact-verified decontamination failed: {decon_report}")
    director.set_phase("completion")
    countermeasure.hold(72)
    recorder.wait_for_frames(world)

    final_countermeasure = world_position(stage, config.panda_base_path)
    final_measurement = world_position(stage, config.measurement_articulation)
    final_detector_position = world_position(stage, config.detector_path)
    final_activity_bq = float(np.sum(decontaminator.activity_bq))
    set_camera(
        stage,
        overview["eye"],
        overview["target"],
        focal_length_mm=float(overview["focal_length_mm"]),
    )
    for _ in range(75):
        world.step(render=True)
    outro_path = ARGS.output / "03_fukushima_building_post_decon_overview.png"
    capture_and_wait(APP, world, outro_path)

    action_seconds = ARGS.duration - INTRO_SECONDS - OUTRO_SECONDS
    telemetry_path = ARGS.output / "h100_surface_contribution_measurements.csv"
    subtitle_path = ARGS.output / ".fukushima-video-hud.ass"
    write_telemetry_csv(recorder.telemetry, telemetry_path, action_seconds)
    write_hud(
        recorder.telemetry,
        subtitle_path,
        duration_s=ARGS.duration,
        action_seconds=action_seconds,
    )
    video_path = encode_video(
        ARGS.output,
        recorder.frame_dir,
        recorder.frame_count,
        intro_path,
        outro_path,
        subtitle_path,
        duration_s=ARGS.duration,
    )
    shutil.rmtree(recorder.frame_dir)
    subtitle_path.unlink(missing_ok=True)

    environment_geometry_count = sum(
        1
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith("/World/Environment/") and prim.IsA(UsdGeom.Mesh)
    )
    initial_row = recorder.telemetry[0]
    final_row = recorder.telemetry[-1]
    overhead_action_fraction = sum(
        row["view_mode"] == VIEW_OVERHEAD_JA for row in recorder.telemetry
    ) / len(recorder.telemetry)
    overview_fraction = (
        INTRO_SECONDS + OUTRO_SECONDS + action_seconds * overhead_action_fraction
    ) / ARGS.duration
    source_position = np.asarray(anchors["decon_surface"], dtype=np.float64)
    initial_detector_distance_m = float(np.linalg.norm(initial_detector_position - source_position))
    final_detector_distance_m = float(np.linalg.norm(final_detector_position - source_position))
    invariants = {
        "imported_fukushima_cad_present": environment_geometry_count >= 995,
        "whole_building_overview_recorded": intro_path.is_file() and outro_path.is_file(),
        "separate_validation_facility_absent": not stage.GetPrimAtPath(
            "/World/RemoteDeconFacility"
        ).IsValid(),
        "both_robots_moved": (
            np.linalg.norm(final_countermeasure - initial_countermeasure) > 0.25
            and np.linalg.norm(final_measurement - initial_measurement) > 0.25
        ),
        "measurement_robot_remained_supported": (
            initial_measurement[2] > -0.5 and final_measurement[2] > -0.5
        ),
        "contact_decontamination_completed": decon_report.success,
        "verified_contacts_recorded": decon_report.accepted_contacts > 0,
        "surface_activity_reduced": final_activity_bq < initial_activity_bq,
        "h100_surface_contribution_reduced": (
            final_row["expected_count_rate_cps"] < initial_row["expected_count_rate_cps"]
        ),
        "radiation_paths_use_h100_face_contributions": (
            final_row["visible_radiation_paths"] <= initial_row["visible_radiation_paths"]
        ),
        "catalog_spawn_stayed_supported": manifest["spawn_vertical_drop_m"] <= 0.15,
        "franka_arm_moved": countermeasure.arm_joint_excursion_rad > 0.2,
        "overview_and_interior_balanced": 0.40 <= overview_fraction <= 0.60,
        "cutaway_preserved_collision": all(row["collision_preserved"] for row in cutaway_audit),
        "h100_monitoring_distance_preserved": (
            abs(final_detector_distance_m - initial_detector_distance_m) <= 0.15
        ),
    }
    failed = [name for name, passed in invariants.items() if not passed]
    result = {
        "passed": not failed,
        **manifest,
        "environment_label": "Fukushima Daiichi building CAD / Nakashima environment",
        "environment_provenance": {
            "project_attribution": "Nakashima-created imported environment (user-provided attribution)",
            "catalog_upstream": "Qualot/fukushima_daiichi_solidworks contributors",
            "catalog_upstream_license": "CC BY 4.0",
            "conversion": "pinned SolidWorks assembly converted to USD with NVIDIA HOOPS",
        },
        "environment_geometry_count": environment_geometry_count,
        "overview_camera": overview,
        "render_cutaway_audit": cutaway_audit,
        "video": str(video_path),
        "duration_s": ARGS.duration,
        "resolution": [1280, 720],
        "encoded_fps": 30,
        "captured_action_frames": recorder.frame_count,
        "perspective_composition": {
            "whole_building_overview_seconds": INTRO_SECONDS + OUTRO_SECONDS,
            "work_area_overhead_action_fraction": overhead_action_fraction,
            "combined_overview_fraction": overview_fraction,
            "interior_fraction": 1.0 - overview_fraction,
            "overhead_explanation_guides": [
                "H100 measurement route (blue)",
                "decontamination robot route (orange)",
                "source-to-H100 contribution paths (fluence-weighted)",
                "fixed phase and numeric HUD",
                "persistent whole-building CAD locator inset",
            ],
        },
        "overview_image": str(intro_path),
        "interior_image": str(interior_path),
        "post_decontamination_overview_image": str(outro_path),
        "robot_models": {
            "countermeasure": "Clearpath Ridgeback + Franka Emika Panda",
            "measurement": "NVIDIA Nova Carter + H3D H100 envelope",
        },
        "motion": {
            "measurement": measurement_report,
            "countermeasure_navigation": navigation_report,
            "countermeasure_work_start_posture": stow_report,
            "countermeasure_work_pose_resynchronization": work_pose_report,
            "decontamination": decon_report,
            "franka_arm_joint_excursion_rad": countermeasure.arm_joint_excursion_rad,
            "teleport_during_operations": False,
            "h100_initial_source_distance_m": initial_detector_distance_m,
            "h100_final_source_distance_m": final_detector_distance_m,
        },
        "surface_source": {
            "path": config.decon_surface_path,
            "label": SOURCE_LABEL,
            "initial_activity_bq": initial_activity_bq,
            "final_activity_bq": final_activity_bq,
            "removed_fraction": (initial_activity_bq - final_activity_bq) / initial_activity_bq,
            "activity_balance_error_bq": decon_report.activity_balance_error_bq,
            "removed_material_accounting": "removed-from-surface audit; no proxy waste zone",
        },
        "h100": {
            "detector_path": config.detector_path,
            "detector_model_id": DETECTOR_MODEL_ID,
            "operating_mode": "4pi omnidirectional scalar monitoring",
            "response_data_status": "synthetic_validation_only",
            "measured_quantity": "expected response from the live planar-surface contribution",
            "initial_expected_count_rate_cps": initial_row["expected_count_rate_cps"],
            "final_expected_count_rate_cps": final_row["expected_count_rate_cps"],
            "initial_dose_rate_usv_h": initial_row["dose_rate_usv_h"],
            "final_dose_rate_usv_h": final_row["dose_rate_usv_h"],
            "telemetry_csv": str(telemetry_path),
        },
        "radiation_visualization": {
            "transport": "per-face Cs-137 yield and inverse-square incident fluence",
            "sampling": "deterministic Gumbel-top-k PPS without replacement",
            "sampling_seed": RADIATION_SAMPLE_SEED,
            "transport_coupled": True,
            "affects_detector_response": False,
            "initial_visible_paths": initial_row["visible_radiation_paths"],
            "final_visible_paths": final_row["visible_radiation_paths"],
        },
        "invariants": invariants,
        "failed_invariants": failed,
    }
    if failed:
        raise RuntimeError(f"Fukushima video invariants failed: {failed}")
    return result


def main() -> int:
    result_path = ARGS.output / ("preview_result.json" if ARGS.preview_only else "result.json")
    result: dict[str, Any]
    exit_code = 0
    started = time.perf_counter()
    try:
        result = run()
    except Exception as error:
        exit_code = 1
        result = {
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        print(result["traceback"], flush=True)
    result["wall_time_s"] = time.perf_counter() - started
    atomic_json(result_path, result)
    print("FUKUSHIMA_DECON_VIDEO_RESULT " + json.dumps(jsonable(result)), flush=True)
    APP.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
