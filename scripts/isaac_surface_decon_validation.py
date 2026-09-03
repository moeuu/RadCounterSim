#!/usr/bin/env python3
"""Run an Arounder-type high-pressure-water decontamination demonstration."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "source/extensions/radcounter.isaac"))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=170.0)
    parser.add_argument("--method", choices=("water", "dry"), default="water")
    parser.add_argument(
        "--operation-mode",
        choices=("surface-wash", "coating-strip", "scarify"),
        default="surface-wash",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=120.0,
        help="treatment-time acceleration; robot treatment speed remains 2 m^2/h physically",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".cache/arounder-water-decon",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="render the decommissioning overview and process-head detail without running a sweep",
    )
    parser.add_argument(
        "--render-video",
        action="store_true",
        help="render a high-reach wall-decontamination frame sequence and encode an MP4",
    )
    parser.add_argument("--video-seconds", type=float, default=20.0)
    parser.add_argument("--video-fps", type=int, default=15)
    parser.add_argument(
        "--render-scene",
        choices=("decontamination", "shield-manipulation"),
        default="decontamination",
        help="select the isolated research scene produced by --render-only",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="render off-screen without creating an X11 application window",
    )
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.time_scale <= 0.0:
        parser.error("--time-scale must be positive")
    if args.video_seconds <= 0.0:
        parser.error("--video-seconds must be positive")
    if args.video_fps <= 0:
        parser.error("--video-fps must be positive")
    if args.render_only and args.render_video:
        parser.error("--render-only and --render-video are mutually exclusive")
    return args


ARGS = parse_arguments()

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": ARGS.headless,
        "width": 1440,
        "height": 900,
        "renderer": "RaytracedLighting",
        "window_title": "RadCounterSim - Arounder-Type Remote Water Decontamination",
    }
)

import omni.kit.viewport.utility as viewport_utility
import omni.ui as ui
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade
from radcounter.isaac.robot.input_router import IsaacRobotInputRouter

from radcounter.core.robots.control import JointCommand, TwistCommand
from radcounter.core.sensors.catalog import popular_detector_catalog
from radcounter.core.sensors.universal import (
    DetectorPose,
    IncidentParticleFluence,
    MeasurementRequest,
    ParametricDetectorModel,
    RadiationType,
    contribution_weighted_sample_without_replacement,
)
from radcounter.core.surface_decontamination import (
    DecontaminationTool,
    SurfaceSourceGrid,
    irregular_deposition_field,
)
from radcounter.core.treatment import WaterJetTreatment
from radcounter.core.water_decontamination import (
    WaterDecontaminationState,
    WaterJetSpec,
    WaterSurfaceDecontaminator,
)

ROBOT_ID = "arounder_research_replica"
ROBOT_PATH = "/World/Arounder"
SOURCE_PATH = "/World/ReactorBuilding/ContaminatedFloor"
HIGH_WALL_SOURCE_PATH = "/World/ReactorBuilding/ContaminatedHighWall"
HIGH_REACH_ROBOT_PATH = "/World/HighReach10"
MEASUREMENT_ROBOT_PATH = "/World/H100MeasurementRover"
H100_DETECTOR_PATH = f"{MEASUREMENT_ROBOT_PATH}/SensorMast/H100"
H100_DETECTOR_POSITION_WORLD_M = (3.80, -2.30, 2.65)
CS137_GAMMA_ENERGY_KEV = 661.657
CS137_GAMMA_YIELD = 0.851
RADIATION_VISUALIZATION_SAMPLE_SEED = 20_260_903
DT_S = 1.0 / 60.0
HEAD_X_M = 0.95
HEAD_NOZZLE_Z_M = -0.17
HEAD_BRUSH_Z_M = -0.34
HEAD_STROKE_M = 0.45
HEAD_WIDTH_M = 0.48
TARGET_TREATMENT_RATE_M2_H = 2.0
PHYSICAL_WASH_SPEED_M_S = TARGET_TREATMENT_RATE_M2_H / 3600.0 / HEAD_WIDTH_M

HITACHI_REFERENCE = "https://www.hitachi-hgne.co.jp/news/2013/20130308.html"
IRID_REFERENCE = "https://irid.or.jp/_pdf/20150714_5.pdf"
MHI_SUPER_GIRAFFE_REFERENCE = (
    "https://www.mhi.com/jp/business/products-services/energy-environment/"
    "nuclear-power-generation/applied-products/robot-mechatronics/super-giraffe"
)
IRID_HIGH_PLACE_REFERENCE = (
    "https://irid.or.jp/topics/"
    "%E9%AB%98%E6%89%80%E7%94%A8%E3%83%89%E3%83%A9%E3%82%A4%E3%82%A2%E3%82%A4%E3%82%B9"
    "%E3%83%96%E3%83%A9%E3%82%B9%E3%83%88%E9%99%A4%E6%9F%93%E8%A3%85%E7%BD%AE%E3%81%AE"
    "%E9%96%8B%E7%99%BA%E3%83%BB%E6%B4%BB/"
)
H3D_H100_REFERENCE = "https://h3dgamma.com/h100.php"
H3D_H100_SPECIFICATION = "https://h3dgamma.com/H100Specs.pdf"
RIDGEBACK_FRANKA_REFERENCE = (
    "https://docs.isaacsim.omniverse.nvidia.com/6.0.0/assets/usd_assets_robots.html"
)
RIDGEBACK_FRANKA_ASSET = "/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd"
MANIPULATOR_PATH = "/World/CountermeasureRobot"
SHIELD_PATH = "/World/ShieldManipulation/ShieldCassette"


@dataclass(frozen=True)
class RouteSegment:
    target_xy_m: tuple[float, float]
    wash: bool
    label: str


@dataclass
class RobotVisuals:
    nozzle_translate_op: object
    spray_prims: tuple[object, ...]
    seal_color_attr: object
    tether_points_attrs: tuple[object, ...]


@dataclass(frozen=True)
class RadiationVisuals:
    ray_curves: tuple[object, ...]
    photon_points_attr: object
    maximum_ray_count: int
    initial_total_fluence_rate_m2_s: float
    initial_maximum_cell_fluence_rate_m2_s: float


@dataclass(frozen=True)
class H100CellContribution:
    cell_index: int
    source_position_world_m: tuple[float, float, float]
    incident_fluence: IncidentParticleFluence


@dataclass(frozen=True)
class RadiationVisualizationFrame:
    visible_ray_count: int
    selected_cell_indices: tuple[int, ...]
    total_fluence_rate_m2_s: float
    selected_fluence_fraction: float


class H100TelemetryRow(TypedDict):
    video_time_s: float
    surface_activity_bq: float
    removed_fraction: float
    expected_count_rate_cps: float
    observed_count_rate_cps: float
    dose_rate_usv_h: float
    incident_fluence_rate_m2_s: float
    visible_radiation_paths: float
    visualized_cell_indices: str
    visualized_cell_fluence_fraction: float


def define_material(
    stage,
    name: str,
    color: tuple[float, float, float],
    *,
    metallic: float = 0.0,
    roughness: float = 0.55,
    opacity: float = 1.0,
):
    material = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
    shader = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def create_materials(stage) -> dict[str, object]:
    UsdGeom.Scope.Define(stage, "/World/Looks")
    return {
        "epoxy": define_material(stage, "EpoxyFloor", (0.19, 0.27, 0.25), roughness=0.38),
        "concrete": define_material(stage, "AgedConcrete", (0.38, 0.40, 0.39), roughness=0.88),
        "yellow": define_material(stage, "IndustrialYellow", (0.91, 0.56, 0.035), roughness=0.34),
        "yellow_dark": define_material(stage, "IndustrialYellowDark", (0.54, 0.30, 0.02)),
        "track": define_material(stage, "CrawlerRubber", (0.018, 0.022, 0.024), roughness=0.92),
        "steel": define_material(
            stage, "StainlessSteel", (0.52, 0.56, 0.57), metallic=0.82, roughness=0.24
        ),
        "dark_steel": define_material(stage, "DarkSteel", (0.09, 0.11, 0.12), metallic=0.72),
        "pipe": define_material(stage, "PipeSteel", (0.31, 0.35, 0.36), metallic=0.65),
        "blue": define_material(stage, "WaterLineBlue", (0.03, 0.24, 0.62), metallic=0.18),
        "red": define_material(stage, "ProcessRed", (0.60, 0.055, 0.035), metallic=0.12),
        "green": define_material(stage, "RecoveryGreen", (0.04, 0.38, 0.17), metallic=0.10),
        "glass": define_material(
            stage, "LensGlass", (0.025, 0.14, 0.19), metallic=0.15, roughness=0.08
        ),
        "brush": define_material(stage, "ContainmentBrush", (0.018, 0.018, 0.014), roughness=0.98),
        "water": define_material(
            stage, "WaterJet", (0.05, 0.40, 0.95), roughness=0.08, opacity=0.42
        ),
        "wet": define_material(
            stage, "WetFloor", (0.025, 0.12, 0.20), roughness=0.10, opacity=0.58
        ),
        "white": define_material(stage, "PaintedWhite", (0.72, 0.74, 0.70), roughness=0.54),
        "orange": define_material(stage, "SafetyOrange", (0.94, 0.24, 0.025), roughness=0.45),
        "lead": define_material(
            stage, "ShieldLeadCore", (0.13, 0.15, 0.16), metallic=0.78, roughness=0.42
        ),
        "shield_skin": define_material(
            stage, "ShieldStainlessJacket", (0.28, 0.31, 0.32), metallic=0.86, roughness=0.25
        ),
        "corrosion": define_material(
            stage, "ContaminatedCorrosion", (0.38, 0.12, 0.035), metallic=0.20, roughness=0.88
        ),
    }


def bind(geometry, material) -> None:
    UsdShade.MaterialBindingAPI.Apply(geometry.GetPrim()).Bind(material)


def add_cube(
    stage,
    path: str,
    size_xyz: tuple[float, float, float],
    position_xyz: tuple[float, float, float],
    material,
    *,
    rotation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    collision: bool = False,
):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*position_xyz))
    if any(abs(value) > 1e-9 for value in rotation_xyz):
        cube.AddRotateXYZOp().Set(Gf.Vec3f(*rotation_xyz))
    cube.AddScaleOp().Set(Gf.Vec3f(*size_xyz))
    bind(cube, material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def add_cylinder(
    stage,
    path: str,
    radius_m: float,
    height_m: float,
    position_xyz: tuple[float, float, float],
    material,
    *,
    axis: str = "Z",
    collision: bool = False,
):
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr(axis)
    cylinder.CreateRadiusAttr(radius_m)
    cylinder.CreateHeightAttr(height_m)
    cylinder.AddTranslateOp().Set(Gf.Vec3d(*position_xyz))
    bind(cylinder, material)
    if collision:
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    return cylinder


def add_sphere(stage, path: str, radius_m: float, position_xyz, material):
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius_m)
    sphere.AddTranslateOp().Set(Gf.Vec3d(*position_xyz))
    bind(sphere, material)
    return sphere


def add_link(stage, path: str, start_xz, end_xz, y_m: float, width_m: float, material):
    dx = end_xz[0] - start_xz[0]
    dz = end_xz[1] - start_xz[1]
    length = math.hypot(dx, dz)
    angle_y = -math.degrees(math.atan2(dz, dx))
    return add_cube(
        stage,
        path,
        (length, width_m, width_m),
        ((start_xz[0] + end_xz[0]) * 0.5, y_m, (start_xz[1] + end_xz[1]) * 0.5),
        material,
        rotation_xyz=(0.0, angle_y, 0.0),
    )


def add_link_3d(stage, path: str, start_xyz, end_xyz, width_m: float, material):
    start = np.asarray(start_xyz, dtype=np.float64)
    end = np.asarray(end_xyz, dtype=np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        raise ValueError("3-D link endpoints must be distinct")
    horizontal = math.hypot(float(delta[0]), float(delta[1]))
    yaw_z = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
    pitch_y = -math.degrees(math.atan2(float(delta[2]), horizontal))
    return add_cube(
        stage,
        path,
        (length, width_m, width_m),
        tuple(float(value) for value in 0.5 * (start + end)),
        material,
        rotation_xyz=(0.0, pitch_y, yaw_z),
    )


def add_curve(stage, path: str, points, width_m: float, color) -> object:
    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateCurveVertexCountsAttr([len(points)])
    points_attr = curve.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    curve.CreateWidthsAttr([width_m])
    curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return points_attr


def create_environment(
    stage, materials, *, high_bay: bool = False
) -> tuple[tuple[float, float, float], ...]:
    UsdGeom.Xform.Define(stage, "/World/ReactorBuilding")
    wall_height_m = 13.0 if high_bay else 5.0
    column_height_m = 12.6 if high_bay else 4.8
    add_cube(
        stage,
        "/World/ReactorBuilding/Floor",
        (16.0, 12.0, 0.16),
        (0.0, 0.0, -0.08),
        materials["epoxy"],
        collision=True,
    )
    add_cube(
        stage,
        "/World/ReactorBuilding/WallNorth",
        (16.0, 0.26, wall_height_m),
        (0.0, 5.86, wall_height_m * 0.5),
        materials["concrete"],
        collision=True,
    )
    add_cube(
        stage,
        "/World/ReactorBuilding/WallWest",
        (0.26, 12.0, wall_height_m),
        (-7.86, 0.0, wall_height_m * 0.5),
        materials["concrete"],
        collision=True,
    )
    add_cube(
        stage,
        "/World/ReactorBuilding/WallEast",
        (0.26, 8.0, wall_height_m),
        (7.86, 2.0, wall_height_m * 0.5),
        materials["concrete"],
        collision=True,
    )

    column_positions = (
        ((-6.4, 4.2), (6.4, 4.2), (-6.4, -4.1), (6.4, -4.1))
        if high_bay
        else ((-5.4, 4.2), (0.0, 4.2), (5.4, 4.2), (-5.4, -4.1), (5.4, -4.1))
    )
    for index, (x_m, y_m) in enumerate(column_positions):
        add_cube(
            stage,
            f"/World/ReactorBuilding/Columns/C{index}",
            (0.52, 0.52, column_height_m),
            (x_m, y_m, column_height_m * 0.5),
            materials["concrete"],
            collision=True,
        )
        add_cube(
            stage,
            f"/World/ReactorBuilding/Columns/C{index}_Foot",
            (0.82, 0.82, 0.16),
            (x_m, y_m, 0.08),
            materials["concrete"],
            collision=True,
        )

    add_cylinder(
        stage,
        "/World/ReactorBuilding/BiologicalShield",
        1.62,
        8.8 if high_bay else 4.3,
        (4.8, 2.25, 4.4 if high_bay else 2.15),
        materials["concrete"],
        collision=True,
    )
    add_cylinder(
        stage,
        "/World/ReactorBuilding/BiologicalShieldRing",
        1.82,
        0.26,
        (4.8, 2.25, 0.18),
        materials["dark_steel"],
        collision=True,
    )

    pipe_specs = (
        ("Condensate", 0.16, 1.25, materials["blue"]),
        ("FireMain", 0.13, 1.85, materials["red"]),
        ("Vent", 0.22, 2.55, materials["pipe"]),
    )
    if high_bay:
        pipe_specs += (
            ("HighSteam", 0.25, 7.15, materials["pipe"]),
            ("HighService", 0.18, 11.15, materials["blue"]),
        )
    for name, radius, z_m, material in pipe_specs:
        add_cylinder(
            stage,
            f"/World/ReactorBuilding/PipeRack/{name}",
            radius,
            13.8,
            (0.0, 5.25, z_m),
            material,
            axis="X",
        )
    rack_support_height_m = 11.8 if high_bay else 3.1
    for index, x_m in enumerate((-6.2, -3.0, 0.2, 3.4, 6.3)):
        add_cube(
            stage,
            f"/World/ReactorBuilding/PipeRack/Support{index}",
            (0.12, 0.55, rack_support_height_m),
            (x_m, 5.22, rack_support_height_m * 0.5),
            materials["dark_steel"],
            collision=True,
        )

    for index, (x_m, radius, height) in enumerate(((5.8, 0.48, 1.8), (6.65, 0.36, 1.45))):
        add_cylinder(
            stage,
            f"/World/ReactorBuilding/ProcessSkid/Tank{index}",
            radius,
            height,
            (x_m, -3.7, height * 0.5 + 0.15),
            materials["white"],
            collision=True,
        )
        add_cylinder(
            stage,
            f"/World/ReactorBuilding/ProcessSkid/Tank{index}Cap",
            radius * 0.82,
            0.08,
            (x_m, -3.7, height + 0.18),
            materials["dark_steel"],
        )
    add_cube(
        stage,
        "/World/ReactorBuilding/ProcessSkid/Base",
        (2.25, 1.55, 0.18),
        (5.9, -3.7, 0.09),
        materials["dark_steel"],
        collision=True,
    )
    for step in range(6):
        add_cube(
            stage,
            f"/World/ReactorBuilding/Stairs/Step{step}",
            (0.75, 0.32, 0.10),
            (6.7, -1.65 + step * 0.28, 0.05 + step * 0.11),
            materials["steel"],
            collision=True,
        )

    drain_y = -1.42
    add_cube(
        stage,
        "/World/ReactorBuilding/Drain/Channel",
        (4.2, 0.30, 0.025),
        (0.35, drain_y, 0.012),
        materials["dark_steel"],
    )
    for index, x_m in enumerate(np.linspace(-1.65, 2.35, 24)):
        add_cube(
            stage,
            f"/World/ReactorBuilding/Drain/Bar{index:02d}",
            (0.035, 0.29, 0.035),
            (float(x_m), drain_y, 0.028),
            materials["steel"],
        )

    add_cube(
        stage,
        "/World/ReactorBuilding/UtilitySkid/Base",
        (2.8, 1.75, 0.16),
        (-5.55, -3.65, 0.08),
        materials["dark_steel"],
        collision=True,
    )
    add_cylinder(
        stage,
        "/World/ReactorBuilding/UtilitySkid/RecoveryTank",
        0.53,
        1.55,
        (-6.15, -3.65, 0.92),
        materials["white"],
        collision=True,
    )
    add_cylinder(
        stage,
        "/World/ReactorBuilding/UtilitySkid/Pump",
        0.25,
        0.64,
        (-4.65, -3.80, 0.42),
        materials["blue"],
        axis="X",
    )
    add_cube(
        stage,
        "/World/ReactorBuilding/UtilitySkid/ControlCabinet",
        (0.58, 0.42, 1.05),
        (-4.75, -3.10, 0.61),
        materials["yellow_dark"],
        collision=True,
    )
    add_cylinder(
        stage,
        "/World/ReactorBuilding/HoseReel/Drum",
        0.43,
        0.68,
        (-5.15, -2.55, 0.78),
        materials["yellow"],
        axis="Y",
    )
    add_cylinder(
        stage,
        "/World/ReactorBuilding/HoseReel/FlangeL",
        0.55,
        0.07,
        (-5.15, -2.93, 0.78),
        materials["yellow_dark"],
        axis="Y",
    )
    add_cylinder(
        stage,
        "/World/ReactorBuilding/HoseReel/FlangeR",
        0.55,
        0.07,
        (-5.15, -2.17, 0.78),
        materials["yellow_dark"],
        axis="Y",
    )
    add_cube(
        stage,
        "/World/ReactorBuilding/HoseReel/StandL",
        (0.10, 0.10, 1.25),
        (-5.58, -2.85, 0.63),
        materials["yellow_dark"],
    )
    add_cube(
        stage,
        "/World/ReactorBuilding/HoseReel/StandR",
        (0.10, 0.10, 1.25),
        (-4.72, -2.85, 0.63),
        materials["yellow_dark"],
    )

    roller_points = ((-3.75, -2.25, 0.20), (-2.55, -1.85, 0.20))
    for index, point in enumerate(roller_points):
        add_cylinder(
            stage,
            f"/World/ReactorBuilding/CornerRollers/R{index}",
            0.12,
            0.32,
            point,
            materials["orange"],
            axis="Z",
        )
        add_cube(
            stage,
            f"/World/ReactorBuilding/CornerRollers/R{index}Base",
            (0.38, 0.38, 0.04),
            (point[0], point[1], 0.02),
            materials["dark_steel"],
        )

    for index in range(5):
        x_m = -6.6 + index * 0.55
        add_cube(
            stage,
            f"/World/ReactorBuilding/Barrier/Post{index}",
            (0.06, 0.06, 0.85),
            (x_m, -2.65, 0.43),
            materials["orange"],
        )
    add_cube(
        stage,
        "/World/ReactorBuilding/Barrier/Rail",
        (2.25, 0.05, 0.06),
        (-5.5, -2.65, 0.72),
        materials["orange"],
    )

    for index, (x_m, y_m, yaw) in enumerate(
        ((-6.5, 1.0, 17.0), (3.4, -4.3, -8.0), (6.7, 0.2, 25.0))
    ):
        add_cube(
            stage,
            f"/World/ReactorBuilding/Debris/Chunk{index}",
            (0.34, 0.22, 0.14),
            (x_m, y_m, 0.08),
            materials["concrete"],
            rotation_xyz=(0.0, 0.0, yaw),
            collision=True,
        )

    light_height_m = 12.35 if high_bay else 4.65
    for index, x_m in enumerate((-4.8, 0.0, 4.8)):
        light = UsdLux.RectLight.Define(stage, f"/World/ReactorBuilding/Lights/L{index}")
        light.CreateIntensityAttr(18_000.0)
        light.CreateWidthAttr(2.2)
        light.CreateHeightAttr(0.25)
        light.CreateColorAttr(Gf.Vec3f(0.78, 0.88, 1.0))
        light.AddTranslateOp().Set(Gf.Vec3d(x_m, 0.0, light_height_m))
        light.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    return roller_points


def create_robot(stage, materials, start_xyz) -> RobotVisuals:
    root = UsdGeom.Xform.Define(stage, ROBOT_PATH)
    root.AddTranslateOp().Set(Gf.Vec3d(*start_xyz))
    prim = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(840.0)
    prim.CreateAttribute("rad:robot:id", Sdf.ValueTypeNames.String).Set(ROBOT_ID)
    prim.CreateAttribute("rad:robot:role", Sdf.ValueTypeNames.String).Set("mitigation")
    prim.CreateAttribute("rad:robot:reference", Sdf.ValueTypeNames.String).Set(
        "Hitachi-GE Arounder"
    )
    prim.CreateAttribute("rad:stream:focus", Sdf.ValueTypeNames.Bool).Set(True)

    add_cube(
        stage,
        f"{ROBOT_PATH}/LowerChassis",
        (1.28, 0.52, 0.30),
        (-0.05, 0.0, -0.10),
        materials["yellow_dark"],
        collision=True,
    )
    add_cube(
        stage,
        f"{ROBOT_PATH}/MainBody",
        (1.12, 0.54, 0.42),
        (-0.13, 0.0, 0.16),
        materials["yellow"],
        collision=True,
    )
    add_cube(
        stage,
        f"{ROBOT_PATH}/RearPowerPack",
        (0.46, 0.50, 0.55),
        (-0.42, 0.0, 0.52),
        materials["yellow"],
    )
    add_cube(
        stage,
        f"{ROBOT_PATH}/TopServicePanel",
        (0.62, 0.45, 0.06),
        (-0.12, 0.0, 0.48),
        materials["dark_steel"],
    )

    for side_name, y_m in (("Left", 0.36), ("Right", -0.36)):
        add_cube(
            stage,
            f"{ROBOT_PATH}/Tracks/{side_name}/Belt",
            (1.50, 0.18, 0.34),
            (-0.08, y_m, -0.18),
            materials["track"],
            collision=True,
        )
        for wheel_index, x_m in enumerate((-0.57, -0.18, 0.22, 0.57)):
            add_cylinder(
                stage,
                f"{ROBOT_PATH}/Tracks/{side_name}/RoadWheel{wheel_index}",
                0.135,
                0.19,
                (x_m, y_m, -0.18),
                materials["steel"],
                axis="Y",
            )
            add_cylinder(
                stage,
                f"{ROBOT_PATH}/Tracks/{side_name}/Hub{wheel_index}",
                0.055,
                0.205,
                (x_m, y_m, -0.18),
                materials["yellow_dark"],
                axis="Y",
            )
        for tread_index, x_m in enumerate(np.linspace(-0.70, 0.54, 10)):
            for surface, z_m in (("Top", -0.005), ("Bottom", -0.355)):
                add_cube(
                    stage,
                    f"{ROBOT_PATH}/Tracks/{side_name}/{surface}Tread{tread_index:02d}",
                    (0.105, 0.205, 0.030),
                    (float(x_m), y_m, z_m),
                    materials["track"],
                )

    shoulder = (-0.25, 0.46)
    elbow = (0.16, 0.73)
    wrist = (0.52, 0.39)
    tool_mount = (0.79, -0.10)
    add_cylinder(
        stage,
        f"{ROBOT_PATH}/Arm/BaseYaw",
        0.19,
        0.16,
        (-0.25, 0.0, 0.43),
        materials["yellow_dark"],
        axis="Z",
    )
    add_link(stage, f"{ROBOT_PATH}/Arm/Boom", shoulder, elbow, 0.0, 0.13, materials["yellow"])
    add_link(stage, f"{ROBOT_PATH}/Arm/Forearm", elbow, wrist, 0.0, 0.12, materials["steel"])
    add_link(
        stage, f"{ROBOT_PATH}/Arm/WristLink", wrist, tool_mount, 0.0, 0.105, materials["steel"]
    )
    add_link(
        stage,
        f"{ROBOT_PATH}/Arm/HydraulicCylinderA",
        (-0.30, 0.34),
        (0.09, 0.68),
        0.10,
        0.052,
        materials["steel"],
    )
    add_link(
        stage,
        f"{ROBOT_PATH}/Arm/HydraulicCylinderB",
        (0.05, 0.65),
        (0.47, 0.33),
        -0.10,
        0.045,
        materials["yellow_dark"],
    )
    for index, point in enumerate((shoulder, elbow, wrist, tool_mount)):
        add_cylinder(
            stage,
            f"{ROBOT_PATH}/Arm/Joint{index}",
            0.105,
            0.22,
            (point[0], 0.0, point[1]),
            materials["dark_steel"],
            axis="Y",
        )

    head = UsdGeom.Xform.Define(stage, f"{ROBOT_PATH}/Arm/DeconHead")
    head.AddTranslateOp().Set(Gf.Vec3d(HEAD_X_M, 0.0, -0.26))
    add_cube(
        stage,
        f"{ROBOT_PATH}/Arm/DeconHead/Top",
        (0.42, 0.58, 0.075),
        (0.0, 0.0, 0.09),
        materials["steel"],
    )
    add_cube(
        stage,
        f"{ROBOT_PATH}/Arm/DeconHead/FrontFrame",
        (0.075, 0.58, 0.18),
        (0.17, 0.0, 0.015),
        materials["steel"],
    )
    add_cube(
        stage,
        f"{ROBOT_PATH}/Arm/DeconHead/RearFrame",
        (0.075, 0.58, 0.18),
        (-0.17, 0.0, 0.015),
        materials["steel"],
    )
    add_cube(
        stage,
        f"{ROBOT_PATH}/Arm/DeconHead/BrushFront",
        (0.055, 0.58, 0.10),
        (0.19, 0.0, -0.075),
        materials["brush"],
    )
    add_cube(
        stage,
        f"{ROBOT_PATH}/Arm/DeconHead/BrushRear",
        (0.055, 0.58, 0.10),
        (-0.19, 0.0, -0.075),
        materials["brush"],
    )
    add_cube(
        stage,
        f"{ROBOT_PATH}/Arm/DeconHead/BrushLeft",
        (0.34, 0.045, 0.10),
        (0.0, 0.29, -0.075),
        materials["brush"],
    )
    add_cube(
        stage,
        f"{ROBOT_PATH}/Arm/DeconHead/BrushRight",
        (0.34, 0.045, 0.10),
        (0.0, -0.29, -0.075),
        materials["brush"],
    )
    add_cylinder(
        stage,
        f"{ROBOT_PATH}/Arm/DeconHead/SuctionPort",
        0.075,
        0.13,
        (-0.10, 0.0, 0.17),
        materials["green"],
        axis="Z",
    )

    carriage = UsdGeom.Xform.Define(stage, f"{ROBOT_PATH}/Arm/DeconHead/NozzleCarriage")
    nozzle_translate_op = carriage.AddTranslateOp()
    nozzle_translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    add_cube(
        stage,
        f"{ROBOT_PATH}/Arm/DeconHead/NozzleCarriage/RailCar",
        (0.12, 0.10, 0.055),
        (0.0, 0.0, 0.035),
        materials["yellow_dark"],
    )
    add_cylinder(
        stage,
        f"{ROBOT_PATH}/Arm/DeconHead/NozzleCarriage/Nozzle",
        0.028,
        0.105,
        (0.04, 0.0, -0.02),
        materials["steel"],
        axis="Z",
    )
    spray_prims = []
    for index, offset_y in enumerate((-0.025, 0.0, 0.025)):
        cone = UsdGeom.Cone.Define(
            stage, f"{ROBOT_PATH}/Arm/DeconHead/NozzleCarriage/ContainedJet{index}"
        )
        cone.CreateAxisAttr("Z")
        cone.CreateRadiusAttr(0.035 + 0.008 * index)
        cone.CreateHeightAttr(0.11)
        cone.AddTranslateOp().Set(Gf.Vec3d(0.04, offset_y, -0.095))
        bind(cone, materials["water"])
        cone.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        spray_prims.append(cone)

    for camera_index, y_m in enumerate((-0.18, 0.18)):
        add_cube(
            stage,
            f"{ROBOT_PATH}/Vision/Camera{camera_index}Body",
            (0.14, 0.12, 0.11),
            (0.52, y_m, 0.54),
            materials["dark_steel"],
        )
        add_cylinder(
            stage,
            f"{ROBOT_PATH}/Vision/Camera{camera_index}Lens",
            0.043,
            0.035,
            (0.60, y_m, 0.54),
            materials["glass"],
            axis="X",
        )
        add_sphere(
            stage,
            f"{ROBOT_PATH}/Vision/WorkLight{camera_index}",
            0.065,
            (0.58, y_m * 0.45, 0.38),
            materials["white"],
        )
    seal_lamp = add_sphere(
        stage, f"{ROBOT_PATH}/Vision/SealLamp", 0.045, (0.30, 0.0, 0.66), materials["red"]
    )
    seal_color_attr = seal_lamp.CreateDisplayColorAttr([Gf.Vec3f(0.88, 0.04, 0.02)])

    add_curve(
        stage,
        f"{ROBOT_PATH}/Hoses/HighPressure",
        [(-0.65, -0.16, 0.62), (-0.12, -0.16, 0.78), (0.42, -0.15, 0.44), (0.88, -0.12, -0.08)],
        0.036,
        (0.06, 0.22, 0.72),
    )
    add_curve(
        stage,
        f"{ROBOT_PATH}/Hoses/Recovery",
        [(-0.65, 0.16, 0.58), (-0.12, 0.18, 0.74), (0.45, 0.18, 0.40), (0.86, 0.13, -0.04)],
        0.058,
        (0.05, 0.08, 0.07),
    )
    add_curve(
        stage,
        f"{ROBOT_PATH}/Hoses/Control",
        [(-0.65, 0.05, 0.66), (-0.05, 0.05, 0.82), (0.52, 0.06, 0.38)],
        0.018,
        (0.95, 0.34, 0.02),
    )

    UsdGeom.Xform.Define(stage, "/World/Utilities")
    tether_points_attrs = (
        add_curve(
            stage,
            "/World/Utilities/Tether/Pressure",
            [(0.0, 0.0, 0.0)] * 5,
            0.040,
            (0.04, 0.16, 0.60),
        ),
        add_curve(
            stage,
            "/World/Utilities/Tether/Recovery",
            [(0.0, 0.0, 0.0)] * 5,
            0.064,
            (0.025, 0.035, 0.03),
        ),
        add_curve(
            stage,
            "/World/Utilities/Tether/PowerControl",
            [(0.0, 0.0, 0.0)] * 5,
            0.020,
            (0.94, 0.28, 0.02),
        ),
    )
    return RobotVisuals(
        nozzle_translate_op, tuple(spray_prims), seal_color_attr, tether_points_attrs
    )


def activity_field(cells_x: int, cells_y: int) -> np.ndarray:
    return irregular_deposition_field(cells_x, cells_y)


def create_surface_source(stage) -> tuple[SurfaceSourceGrid, list, list]:
    cells_x, cells_y = 48, 28
    efficiency = np.ones(cells_x * cells_y, dtype=np.float64)
    field = efficiency.reshape(cells_y, cells_x)
    field[3:8, 17:23] = 0.52
    field[9:13, 6:11] = 0.22
    grid = SurfaceSourceGrid(
        cells_x=cells_x,
        cells_y=cells_y,
        size_x_m=3.0,
        size_y_m=1.6,
        center_world_m=(0.35, 0.0, 0.014),
        activity_bq_per_cell=activity_field(cells_x, cells_y),
        efficiency_field=efficiency,
    )
    root = UsdGeom.Xform.Define(stage, SOURCE_PATH)
    root_prim = root.GetPrim()
    root_prim.CreateAttribute("rad:role", Sdf.ValueTypeNames.String).Set("contaminated_surface")
    root_prim.CreateAttribute("rad:source:type", Sdf.ValueTypeNames.String).Set("surface")
    root_prim.CreateAttribute("rad:surface:coating", Sdf.ValueTypeNames.String).Set(
        "radiation-resistant epoxy"
    )
    root_prim.CreateAttribute("rad:source:initialActivityBq", Sdf.ValueTypeNames.Double).Set(
        grid.initial_total_activity_bq
    )
    color_attributes = []
    activity_attributes = []
    raw_colors = grid.color_rgb()
    epoxy = np.asarray([0.19, 0.27, 0.25])
    # Radiation contamination is not a physical red mat. Keep this as a muted
    # scientific overlay on the epoxy floor, with inactive grid cells hidden.
    colors = 0.34 * raw_colors + 0.66 * epoxy
    for index, center in enumerate(grid.centers_world_m):
        cell = UsdGeom.Cube.Define(stage, f"{SOURCE_PATH}/Cell_{index:03d}")
        cell.CreateSizeAttr(1.0)
        cell.AddTranslateOp().Set(Gf.Vec3d(*center))
        cell.AddScaleOp().Set(Gf.Vec3f(grid.cell_size_x_m * 1.02, grid.cell_size_y_m * 1.02, 0.004))
        color_attribute = cell.CreateDisplayColorAttr([Gf.Vec3f(*colors[index])])
        if grid.activity_bq[index] <= 0.0:
            cell.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        color_attributes.append(color_attribute)
        prim = cell.GetPrim()
        prim.CreateAttribute("rad:role", Sdf.ValueTypeNames.String).Set("source")
        prim.CreateAttribute("rad:source:type", Sdf.ValueTypeNames.String).Set("surface_cell")
        activity_attr = prim.CreateAttribute("rad:source:activityBq", Sdf.ValueTypeNames.Double)
        activity_attr.Set(float(grid.activity_bq[index]))
        activity_attributes.append(activity_attr)
        prim.CreateAttribute("rad:decon:enabled", Sdf.ValueTypeNames.Bool).Set(True)
    return grid, color_attributes, activity_attributes


def create_high_wall_surface_source(stage) -> SurfaceSourceGrid:
    cells_x, cells_y = 48, 28
    efficiency = np.ones(cells_x * cells_y, dtype=np.float64)
    efficiency.reshape(cells_y, cells_x)[8:15, 20:29] = 0.48
    grid = SurfaceSourceGrid(
        cells_x=cells_x,
        cells_y=cells_y,
        size_x_m=3.2,
        size_y_m=2.2,
        center_world_m=(0.0, 5.724, 9.35),
        activity_bq_per_cell=activity_field(cells_x, cells_y),
        efficiency_field=efficiency,
        surface_u_world=(1.0, 0.0, 0.0),
        surface_v_world=(0.0, 0.0, 1.0),
    )
    root = UsdGeom.Xform.Define(stage, HIGH_WALL_SOURCE_PATH)
    root_prim = root.GetPrim()
    for name, value_type, value in (
        ("rad:role", Sdf.ValueTypeNames.String, "contaminated_surface"),
        ("rad:source:type", Sdf.ValueTypeNames.String, "surface"),
        ("rad:source:isotopeId", Sdf.ValueTypeNames.String, "Cs-137"),
        ("rad:source:surfaceKind", Sdf.ValueTypeNames.String, "high_wall"),
        ("rad:decon:enabled", Sdf.ValueTypeNames.Bool, True),
        ("rad:source:initialActivityBq", Sdf.ValueTypeNames.Double, grid.initial_total_activity_bq),
        ("rad:source:maximumHeightM", Sdf.ValueTypeNames.Double, 10.45),
    ):
        root_prim.CreateAttribute(name, value_type).Set(value)

    raw_colors = grid.color_rgb()
    concrete = np.asarray([0.38, 0.40, 0.39])
    colors = 0.62 * raw_colors + 0.38 * concrete
    for index, center in enumerate(grid.centers_world_m):
        if grid.activity_bq[index] <= 0.0:
            continue
        cell = UsdGeom.Cube.Define(stage, f"{HIGH_WALL_SOURCE_PATH}/Cell_{index:04d}")
        cell.CreateSizeAttr(1.0)
        cell.AddTranslateOp().Set(Gf.Vec3d(*center))
        cell.AddScaleOp().Set(
            Gf.Vec3f(grid.cell_size_x_m * 1.025, 0.004, grid.cell_size_y_m * 1.025)
        )
        cell.CreateDisplayColorAttr([Gf.Vec3f(*colors[index])])
        prim = cell.GetPrim()
        prim.CreateAttribute("rad:role", Sdf.ValueTypeNames.String).Set("source")
        prim.CreateAttribute("rad:source:type", Sdf.ValueTypeNames.String).Set("surface_cell")
        prim.CreateAttribute("rad:source:activityBq", Sdf.ValueTypeNames.Double).Set(
            float(grid.activity_bq[index])
        )
        prim.CreateAttribute("rad:decon:enabled", Sdf.ValueTypeNames.Bool).Set(True)
    return grid


def create_radiation_visualization(stage, grid: SurfaceSourceGrid) -> RadiationVisuals:
    """Author a detector-contribution-derived view of gamma paths to the H100.

    The visible subset is sampled from the same per-cell incident-fluence values
    submitted to the H100 response. The curves remain a one-way monitor: they
    never participate in transport or alter the detector calculation.
    """

    root_path = "/World/RadiationVisualization"
    root = UsdGeom.Xform.Define(stage, root_path)
    root_prim = root.GetPrim()
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
        (
            "rad:visualization:samplingSeed",
            Sdf.ValueTypeNames.Int,
            RADIATION_VISUALIZATION_SAMPLE_SEED,
        ),
        ("rad:visualization:sourcePath", Sdf.ValueTypeNames.String, HIGH_WALL_SOURCE_PATH),
        ("rad:visualization:detectorPath", Sdf.ValueTypeNames.String, H100_DETECTOR_PATH),
    ):
        root_prim.CreateAttribute(name, value_type).Set(value)

    initial_contributions = _h100_cell_contributions(grid)
    maximum_ray_count = min(32, len(initial_contributions))
    initial_total_fluence_rate = sum(
        item.incident_fluence.fluence_rate_m2_s for item in initial_contributions
    )
    initial_maximum_cell_fluence_rate = max(
        (item.incident_fluence.fluence_rate_m2_s for item in initial_contributions),
        default=0.0,
    )
    initial_weights = np.zeros(len(grid.activity_bq), dtype=np.float64)
    initial_contributions_by_index = {}
    for item in initial_contributions:
        initial_weights[item.cell_index] = item.incident_fluence.fluence_rate_m2_s
        initial_contributions_by_index[item.cell_index] = item
    initial_selected_indices = contribution_weighted_sample_without_replacement(
        initial_weights,
        maximum_ray_count,
        seed=RADIATION_VISUALIZATION_SAMPLE_SEED,
    )
    detector_position = tuple(float(value) for value in H100_DETECTOR_POSITION_WORLD_M)
    ray_curves = []
    for ray_index in range(maximum_ray_count):
        contribution = initial_contributions_by_index[initial_selected_indices[ray_index]]
        source_position = contribution.source_position_world_m
        curve = UsdGeom.BasisCurves.Define(stage, f"{root_path}/Paths/Ray_{ray_index:02d}")
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateCurveVertexCountsAttr([2])
        curve.CreatePointsAttr([Gf.Vec3f(*source_position), Gf.Vec3f(*detector_position)])
        curve.CreateWidthsAttr([0.015])
        curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        curve.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.42, 0.025)])
        curve.CreateDisplayOpacityAttr([0.32])
        curve.GetPrim().CreateAttribute(
            "rad:visualization:sourceCellIndex", Sdf.ValueTypeNames.Int
        ).Set(contribution.cell_index)
        curve.GetPrim().CreateAttribute(
            "rad:visualization:cellFluenceRateM2S", Sdf.ValueTypeNames.Double
        ).Set(contribution.incident_fluence.fluence_rate_m2_s)
        ray_curves.append(curve)

    photons = UsdGeom.Points.Define(stage, f"{root_path}/GammaPhotons")
    photon_points_attr = photons.CreatePointsAttr([])
    photons.CreateWidthsAttr([0.18])
    photons.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    photons.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.78, 0.04)])
    photons.CreateDisplayOpacityAttr([0.94])
    photons.GetPrim().CreateAttribute("rad:visualization:particle", Sdf.ValueTypeNames.String).Set(
        "Cs-137 661.657 keV gamma"
    )
    return RadiationVisuals(
        ray_curves=tuple(ray_curves),
        photon_points_attr=photon_points_attr,
        maximum_ray_count=maximum_ray_count,
        initial_total_fluence_rate_m2_s=initial_total_fluence_rate,
        initial_maximum_cell_fluence_rate_m2_s=initial_maximum_cell_fluence_rate,
    )


def create_high_reach_robot(stage, materials) -> dict[str, object]:
    """Build a 10 m research high-reach decontamination mechanism.

    The architecture is derived from SUPER-Giraffe's telescopic-ladder,
    outrigger, and distal-manipulator layout, but the sixth ladder stage and
    10 m reach envelope are RadCounterSim research extensions rather than a
    claim about the manufacturer's 8 m machine.
    """

    root = UsdGeom.Xform.Define(stage, HIGH_REACH_ROBOT_PATH)
    root.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.70, 0.0))
    prim = root.GetPrim()
    for name, value_type, value in (
        ("rad:role", Sdf.ValueTypeNames.String, "countermeasure_robot"),
        ("rad:robot:model", Sdf.ValueTypeNames.String, "RadCounter HighReach-10"),
        ("rad:robot:referenceArchitecture", Sdf.ValueTypeNames.String, "MHI SUPER-Giraffe"),
        ("rad:robot:geometryFidelity", Sdf.ValueTypeNames.String, "reference_procedural"),
        (
            "rad:robot:driveArchitecture",
            Sdf.ValueTypeNames.String,
            "four_wheel_drive_four_wheel_steering",
        ),
        ("rad:robot:maximumReachHeightM", Sdf.ValueTypeNames.Double, 10.0),
        ("rad:robot:telescopicStages", Sdf.ValueTypeNames.Int, 6),
        ("rad:robot:distalManipulatorDofs", Sdf.ValueTypeNames.Int, 7),
        ("rad:robot:stabilityInterlock", Sdf.ValueTypeNames.Bool, True),
        ("rad:robot:transportRequiresBoomStowed", Sdf.ValueTypeNames.Bool, True),
        ("rad:robot:driveInterlockedWithOutriggers", Sdf.ValueTypeNames.Bool, True),
        ("rad:decon:method", Sdf.ValueTypeNames.String, "dry_ice_blast_with_suction"),
    ):
        prim.CreateAttribute(name, value_type).Set(value)

    # A 10 m ladder cannot credibly sit on the compact cart used by the first
    # concept render.  This is a road-mobile 4WD/4WS carrier: the ladder folds
    # into the rear cradle for transport, and the four jacks must be retracted
    # before the wheel drives can be enabled.
    add_cube(
        stage,
        f"{HIGH_REACH_ROBOT_PATH}/Underbody",
        (2.18, 3.18, 0.22),
        (0.0, -0.10, 0.31),
        materials["dark_steel"],
        collision=True,
    )
    add_cube(
        stage,
        f"{HIGH_REACH_ROBOT_PATH}/Chassis",
        (2.50, 3.60, 0.62),
        (0.0, -0.10, 0.62),
        materials["yellow_dark"],
        collision=True,
    )
    add_cube(
        stage,
        f"{HIGH_REACH_ROBOT_PATH}/EquipmentDeck",
        (2.14, 2.72, 0.34),
        (0.0, -0.30, 1.02),
        materials["yellow"],
        collision=True,
    )
    for y_m, axle_name in ((-1.27, "Rear"), (1.20, "Front")):
        for x_m, side_name in ((-1.27, "Left"), (1.27, "Right")):
            name = f"{side_name}_{axle_name}"
            wheel = add_cylinder(
                stage,
                f"{HIGH_REACH_ROBOT_PATH}/SteeredWheels/{name}/Tire",
                0.46,
                0.34,
                (x_m, y_m, 0.47),
                materials["track"],
                axis="X",
            )
            wheel.GetPrim().CreateAttribute("rad:drive:powered", Sdf.ValueTypeNames.Bool).Set(True)
            wheel.GetPrim().CreateAttribute("rad:drive:steered", Sdf.ValueTypeNames.Bool).Set(True)
            add_cylinder(
                stage,
                f"{HIGH_REACH_ROBOT_PATH}/SteeredWheels/{name}/Hub",
                0.18,
                0.38,
                (x_m, y_m, 0.47),
                materials["steel"],
                axis="X",
            )
            add_cube(
                stage,
                f"{HIGH_REACH_ROBOT_PATH}/SteeredWheels/{name}/SteeringKnuckle",
                (0.18, 0.32, 0.25),
                (math.copysign(1.05, x_m), y_m, 0.62),
                materials["dark_steel"],
            )

    for y_m, name in ((-1.94, "Rear"), (1.74, "Front")):
        add_cube(
            stage,
            f"{HIGH_REACH_ROBOT_PATH}/Bumpers/{name}",
            (2.40, 0.16, 0.28),
            (0.0, y_m, 0.48),
            materials["steel"],
            collision=True,
        )

    for x_m in (-1.78, 1.78):
        for y_m in (-1.48, 1.48):
            name = f"{'L' if x_m < 0.0 else 'R'}_{'Rear' if y_m < 0.0 else 'Front'}"
            beam_start = (math.copysign(0.98, x_m), y_m, 0.62)
            beam_end = (x_m, y_m, 0.36)
            add_link_3d(
                stage,
                f"{HIGH_REACH_ROBOT_PATH}/Outriggers/{name}Beam",
                beam_start,
                beam_end,
                0.14,
                materials["steel"],
            )
            add_cylinder(
                stage,
                f"{HIGH_REACH_ROBOT_PATH}/Outriggers/{name}Jack",
                0.11,
                0.52,
                (x_m, y_m, 0.30),
                materials["dark_steel"],
                axis="Z",
            )
            add_cube(
                stage,
                f"{HIGH_REACH_ROBOT_PATH}/Outriggers/{name}Pad",
                (0.44, 0.44, 0.10),
                (x_m, y_m, 0.06),
                materials["dark_steel"],
                collision=True,
            )

    add_cylinder(
        stage,
        f"{HIGH_REACH_ROBOT_PATH}/Turntable",
        0.62,
        0.30,
        (0.0, 0.35, 1.31),
        materials["dark_steel"],
    )
    add_cube(
        stage,
        f"{HIGH_REACH_ROBOT_PATH}/Counterweight",
        (1.55, 0.90, 0.78),
        (0.0, -0.78, 1.47),
        materials["yellow"],
        collision=True,
    )

    for x_m in (-0.53, 0.53):
        add_cube(
            stage,
            f"{HIGH_REACH_ROBOT_PATH}/TransportCradle/Side{'L' if x_m < 0 else 'R'}",
            (0.16, 0.22, 0.72),
            (x_m, -1.25, 1.48),
            materials["dark_steel"],
        )
    add_cube(
        stage,
        f"{HIGH_REACH_ROBOT_PATH}/TransportCradle/Saddle",
        (1.22, 0.25, 0.18),
        (0.0, -1.25, 1.80),
        materials["steel"],
    )

    boom_start = np.asarray((0.0, 0.35, 1.57), dtype=np.float64)
    boom_tip = np.asarray((0.0, 4.12, 9.40), dtype=np.float64)
    fractions = (0.0, 0.23, 0.43, 0.61, 0.76, 0.89, 1.0)
    widths = (0.48, 0.43, 0.38, 0.33, 0.28, 0.23)
    for index, (low, high, width_m) in enumerate(
        zip(fractions[:-1], fractions[1:], widths, strict=True), start=1
    ):
        start = boom_start + low * (boom_tip - boom_start)
        end = boom_start + high * (boom_tip - boom_start)
        add_link_3d(
            stage,
            f"{HIGH_REACH_ROBOT_PATH}/TelescopicLadder/Stage{index}",
            start,
            end,
            width_m,
            materials["yellow"] if index % 2 else materials["steel"],
        )
        add_cube(
            stage,
            f"{HIGH_REACH_ROBOT_PATH}/TelescopicLadder/Collar{index}",
            (width_m + 0.09, width_m + 0.09, width_m + 0.09),
            tuple(float(value) for value in end),
            materials["dark_steel"],
        )

    arm_points = (
        tuple(boom_tip),
        (0.18, 4.22, 9.55),
        (0.33, 4.34, 9.61),
        (0.28, 4.46, 9.49),
        (0.15, 4.58, 9.39),
        (0.06, 4.69, 9.38),
        (0.01, 4.79, 9.38),
        (0.00, 4.88, 9.38),
    )
    for index, (start, end) in enumerate(zip(arm_points[:-1], arm_points[1:], strict=True)):
        add_link_3d(
            stage,
            f"{HIGH_REACH_ROBOT_PATH}/DistalManipulator/Link{index + 1}",
            start,
            end,
            0.14 - 0.008 * index,
            materials["steel"] if index % 2 else materials["orange"],
        )
        add_sphere(
            stage,
            f"{HIGH_REACH_ROBOT_PATH}/DistalManipulator/Joint{index + 1}",
            0.10 - 0.005 * index,
            end,
            materials["dark_steel"],
        )

    head_center = (0.0, 4.96, 9.38)
    head = UsdGeom.Xform.Define(stage, f"{HIGH_REACH_ROBOT_PATH}/DeconHead")
    head.GetPrim().CreateAttribute("rad:role", Sdf.ValueTypeNames.String).Set("decon_tool")
    head.GetPrim().CreateAttribute("rad:decon:contactHeightM", Sdf.ValueTypeNames.Double).Set(9.38)
    add_cube(
        stage,
        f"{HIGH_REACH_ROBOT_PATH}/DeconHead/RecoveryHousing",
        (0.64, 0.16, 0.54),
        head_center,
        materials["steel"],
    )
    add_cube(
        stage,
        f"{HIGH_REACH_ROBOT_PATH}/DeconHead/ContainmentBrush",
        (0.60, 0.06, 0.50),
        (0.0, 5.07, 9.38),
        materials["brush"],
    )
    add_cylinder(
        stage,
        f"{HIGH_REACH_ROBOT_PATH}/DeconHead/SuctionPort",
        0.085,
        0.18,
        (0.0, 4.84, 9.38),
        materials["green"],
        axis="Y",
    )
    add_curve(
        stage,
        f"{HIGH_REACH_ROBOT_PATH}/Hoses/BlastSupply",
        [
            (0.38, -0.35, 0.85),
            (0.28, 1.15, 3.0),
            (0.20, 2.35, 5.7),
            (0.12, 3.55, 8.0),
            (0.08, 4.82, 9.30),
        ],
        0.045,
        (0.10, 0.28, 0.72),
    )
    add_curve(
        stage,
        f"{HIGH_REACH_ROBOT_PATH}/Hoses/DustRecovery",
        [
            (-0.38, -0.30, 0.82),
            (-0.30, 1.10, 2.9),
            (-0.22, 2.30, 5.6),
            (-0.15, 3.50, 7.9),
            (-0.10, 4.82, 9.32),
        ],
        0.075,
        (0.04, 0.05, 0.05),
    )

    return {
        "robot_path": HIGH_REACH_ROBOT_PATH,
        "model": "RadCounter HighReach-10",
        "drive_architecture": "four-wheel drive / four-wheel steering",
        "wheel_diameter_m": 0.92,
        "chassis_dimensions_m": [2.50, 3.60, 0.62],
        "rendered_configuration": "working: outriggers deployed, wheel drive interlocked",
        "transport_transition": [
            "retract distal manipulator",
            "retract and lower telescopic ladder into transport cradle",
            "raise four outrigger jacks",
            "enable four wheel steering and drive",
        ],
        "maximum_reach_height_m": 10.0,
        "telescopic_stages": 6,
        "distal_manipulator_dofs": 7,
        "reference_architecture": "MHI SUPER-Giraffe (8 m); extended research design",
        "reference_urls": [MHI_SUPER_GIRAFFE_REFERENCE, IRID_HIGH_PLACE_REFERENCE],
    }


def create_h100_measurement_robot(stage, materials) -> dict[str, object]:
    """Build a separate rover carrying an H3D H100-sized detector body."""

    root = UsdGeom.Xform.Define(stage, MEASUREMENT_ROBOT_PATH)
    root.AddTranslateOp().Set(Gf.Vec3d(3.80, -2.30, 0.0))
    root_prim = root.GetPrim()
    for name, value_type, value in (
        ("rad:role", Sdf.ValueTypeNames.String, "measurement_robot"),
        ("rad:robot:model", Sdf.ValueTypeNames.String, "RadCounter H100 Survey Rover"),
        ("rad:robot:geometryFidelity", Sdf.ValueTypeNames.String, "reference_procedural"),
        ("rad:robot:controller", Sdf.ValueTypeNames.String, "stationary_monitoring_pose"),
        ("rad:measurement:independentPlatform", Sdf.ValueTypeNames.Bool, True),
    ):
        root_prim.CreateAttribute(name, value_type).Set(value)

    add_cube(
        stage,
        f"{MEASUREMENT_ROBOT_PATH}/Chassis",
        (1.10, 1.42, 0.30),
        (0.0, 0.0, 0.38),
        materials["blue"],
        collision=True,
    )
    add_cube(
        stage,
        f"{MEASUREMENT_ROBOT_PATH}/EquipmentDeck",
        (0.86, 0.92, 0.34),
        (0.0, -0.04, 0.66),
        materials["white"],
    )
    for x_m, side in ((-0.64, "Left"), (0.64, "Right")):
        add_cube(
            stage,
            f"{MEASUREMENT_ROBOT_PATH}/Tracks/{side}",
            (0.24, 1.58, 0.34),
            (x_m, 0.0, 0.31),
            materials["track"],
            collision=True,
        )
        for y_m, axle in ((-0.52, "Rear"), (0.52, "Front")):
            add_cylinder(
                stage,
                f"{MEASUREMENT_ROBOT_PATH}/Tracks/{side}/{axle}Wheel",
                0.20,
                0.27,
                (x_m, y_m, 0.31),
                materials["dark_steel"],
                axis="X",
            )

    add_cylinder(
        stage,
        f"{MEASUREMENT_ROBOT_PATH}/SensorMast/Lower",
        0.055,
        1.35,
        (0.0, 0.0, 1.40),
        materials["steel"],
    )
    add_cylinder(
        stage,
        f"{MEASUREMENT_ROBOT_PATH}/SensorMast/Upper",
        0.037,
        0.78,
        (0.0, 0.0, 2.31),
        materials["dark_steel"],
    )
    add_cube(
        stage,
        f"{MEASUREMENT_ROBOT_PATH}/SensorMast/Cradle",
        (0.36, 0.26, 0.055),
        (0.0, 0.0, 2.54),
        materials["dark_steel"],
    )
    # H3D publishes a 9.6 x 3.4 x 6.9 inch envelope.  This body keeps that
    # physical scale; the larger cradle and mast make the small instrument
    # legible in the wide reactor-building shot.
    detector = add_cube(
        stage,
        H100_DETECTOR_PATH,
        (0.244, 0.086, 0.175),
        (0.0, 0.0, 2.65),
        materials["yellow"],
    )
    detector_prim = detector.GetPrim()
    for name, value_type, value in (
        ("rad:role", Sdf.ValueTypeNames.String, "detector"),
        ("rad:detector:id", Sdf.ValueTypeNames.String, "h100_rover"),
        ("rad:detector:model", Sdf.ValueTypeNames.String, "h3d_h100_omni"),
        ("rad:detector:manufacturer", Sdf.ValueTypeNames.String, "H3D, Inc."),
        ("rad:detector:directionality", Sdf.ValueTypeNames.String, "omnidirectional"),
        ("rad:detector:radiationFovSr", Sdf.ValueTypeNames.Double, 4.0 * math.pi),
        ("rad:detector:energyMinKeV", Sdf.ValueTypeNames.Double, 50.0),
        ("rad:detector:energyMaxKeV", Sdf.ValueTypeNames.Double, 3000.0),
        ("rad:detector:cztVolumeCm3", Sdf.ValueTypeNames.Double, 6.0),
        ("rad:detector:responseDataStatus", Sdf.ValueTypeNames.String, "synthetic_validation_only"),
        ("rad:detector:productUrl", Sdf.ValueTypeNames.String, H3D_H100_REFERENCE),
        ("rad:detector:specificationUrl", Sdf.ValueTypeNames.String, H3D_H100_SPECIFICATION),
    ):
        detector_prim.CreateAttribute(name, value_type).Set(value)
    add_cube(
        stage,
        f"{MEASUREMENT_ROBOT_PATH}/SensorMast/H100FrontPanel",
        (0.176, 0.006, 0.108),
        (0.0, -0.046, 2.65),
        materials["dark_steel"],
    )
    add_sphere(
        stage,
        f"{MEASUREMENT_ROBOT_PATH}/StatusBeacon",
        0.075,
        (0.0, -0.34, 0.91),
        materials["green"],
    )
    return {
        "measurement_robot_path": MEASUREMENT_ROBOT_PATH,
        "measurement_robot_model": "RadCounter H100 Survey Rover",
        "detector_path": H100_DETECTOR_PATH,
        "detector_model": "H3D H100 Gamma-Ray Imaging Spectrometer",
        "detector_operating_mode": "omnidirectional scalar monitoring",
        "detector_response_data_status": "synthetic_validation_only",
        "detector_reference_urls": [H3D_H100_REFERENCE, H3D_H100_SPECIFICATION],
    }


def create_work_zone(stage, materials, grid: SurfaceSourceGrid) -> None:
    half_x = grid.size_x_m * 0.5
    half_y = grid.size_y_m * 0.5
    cx, cy = grid.center_world_m[:2]
    for name, size, position in (
        ("North", (grid.size_x_m + 0.14, 0.055, 0.025), (cx, cy + half_y + 0.06, 0.026)),
        ("South", (grid.size_x_m + 0.14, 0.055, 0.025), (cx, cy - half_y - 0.06, 0.026)),
        ("West", (0.055, grid.size_y_m + 0.14, 0.025), (cx - half_x - 0.06, cy, 0.026)),
        ("East", (0.055, grid.size_y_m + 0.14, 0.025), (cx + half_x + 0.06, cy, 0.026)),
    ):
        add_cube(
            stage, f"/World/ReactorBuilding/WorkZone/{name}", size, position, materials["orange"]
        )
    for index, x_m in enumerate(np.linspace(cx - half_x, cx + half_x, 8)):
        add_cube(
            stage,
            f"/World/ReactorBuilding/WorkZone/SurveyMark{index}",
            (0.025, 0.16, 0.02),
            (float(x_m), cy + half_y + 0.18, 0.022),
            materials["white"],
        )


def create_irregular_cylindrical_surface_source(
    stage,
    path: str,
    *,
    center_world_m: tuple[float, float, float],
    radius_m: float,
    height_m: float,
    total_activity_bq: float,
) -> dict[str, object]:
    """Create a sparse face-activity field conforming to a cylindrical object."""

    angular_cells = 48
    vertical_cells = 24
    angles = np.linspace(-math.pi, math.pi, angular_cells, endpoint=False)
    heights = np.linspace(-1.0, 1.0, vertical_cells)
    theta, zz = np.meshgrid(angles, heights, indexing="xy")

    def wrapped_delta(value, center):
        return np.angle(np.exp(1j * (value - center)))

    field = (
        2.5 * np.exp(-(wrapped_delta(theta, -2.45) ** 2 / 0.48 + (zz - 0.18) ** 2 / 0.22))
        + 1.7 * np.exp(-(wrapped_delta(theta, -1.75) ** 2 / 0.24 + (zz + 0.38) ** 2 / 0.12))
        + 1.1 * np.exp(-(wrapped_delta(theta, 2.85) ** 2 / 0.14 + (zz - 0.62) ** 2 / 0.08))
    )
    roughness = 0.22 * np.sin(7.0 * theta + 4.0 * zz) + 0.17 * np.cos(13.0 * zz - 2.0 * theta)
    active = field + roughness > 0.62
    active &= ~((wrapped_delta(theta, -2.30) / 0.22) ** 2 + ((zz + 0.02) / 0.16) ** 2 < 1.0)
    active |= (wrapped_delta(theta, -0.95) ** 2 + (zz - 0.52) ** 2 < 0.025) | (
        wrapped_delta(theta, 2.15) ** 2 + (zz + 0.55) ** 2 < 0.018
    )

    weights = np.where(active, np.maximum(0.1, field + 0.45 * roughness), 0.0)
    weights /= max(float(weights.sum()), 1e-12)
    face_activity = total_activity_bq * weights
    points: list[Gf.Vec3f] = []
    counts: list[int] = []
    indices: list[int] = []
    activities: list[float] = []
    colors: list[Gf.Vec3f] = []
    center_x, center_y, center_z = center_world_m
    delta_theta = 2.0 * math.pi / angular_cells
    delta_z = height_m / (vertical_cells - 1)
    visual_radius = radius_m + 0.004
    maximum = max(float(face_activity.max()), 1e-12)
    for row, column in np.argwhere(active):
        theta_center = float(theta[row, column])
        z_center = center_z + 0.5 * height_m * float(zz[row, column])
        theta_low = theta_center - 0.52 * delta_theta
        theta_high = theta_center + 0.52 * delta_theta
        z_low = z_center - 0.52 * delta_z
        z_high = z_center + 0.52 * delta_z
        base_index = len(points)
        for angle, z_m in (
            (theta_low, z_low),
            (theta_high, z_low),
            (theta_high, z_high),
            (theta_low, z_high),
        ):
            points.append(
                Gf.Vec3f(
                    center_x + visual_radius * math.cos(angle),
                    center_y + visual_radius * math.sin(angle),
                    z_m,
                )
            )
        counts.append(4)
        indices.extend((base_index, base_index + 1, base_index + 2, base_index + 3))
        activity = float(face_activity[row, column])
        activities.append(activity)
        fraction = activity / maximum
        colors.append(Gf.Vec3f(0.34 + 0.46 * fraction, 0.07 + 0.10 * fraction, 0.025))

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    color_primvar = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform)
    color_primvar.Set(colors)
    mesh_prim = mesh.GetPrim()
    for name, value_type, value in (
        ("rad:role", Sdf.ValueTypeNames.String, "source"),
        ("rad:source:type", Sdf.ValueTypeNames.String, "surface"),
        ("rad:source:isotopeId", Sdf.ValueTypeNames.String, "Cs-137"),
        ("rad:source:activityBq", Sdf.ValueTypeNames.Double, total_activity_bq),
        ("rad:source:surfaceKind", Sdf.ValueTypeNames.String, "adhered_cylindrical_patch"),
        ("rad:source:faceActivityBq", Sdf.ValueTypeNames.DoubleArray, activities),
        ("rad:source:irregularMask", Sdf.ValueTypeNames.Bool, True),
        ("rad:decon:enabled", Sdf.ValueTypeNames.Bool, True),
    ):
        mesh_prim.CreateAttribute(name, value_type).Set(value)
    return {
        "path": path,
        "active_faces": len(activities),
        "candidate_faces": angular_cells * vertical_cells,
        "total_activity_bq": total_activity_bq,
    }


def create_shield_manipulation_scene(stage, materials) -> dict[str, object]:
    """Author an isolated, reactor-building shield-handling research scene."""

    from isaacsim.storage.native import get_assets_root_path

    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError("Isaac Sim assets root is unavailable")

    UsdGeom.Xform.Define(stage, "/World/ShieldManipulation")
    robot = UsdGeom.Xform.Define(stage, MANIPULATOR_PATH)
    robot.GetPrim().GetReferences().AddReference(assets_root + RIDGEBACK_FRANKA_ASSET)
    robot_prim = robot.GetPrim()
    robot_prim.CreateAttribute("rad:role", Sdf.ValueTypeNames.String).Set("countermeasure_robot")
    robot_prim.CreateAttribute("rad:robot:model", Sdf.ValueTypeNames.String).Set(
        "Clearpath Ridgeback + Franka Emika Panda"
    )
    robot_prim.CreateAttribute("rad:robot:geometryFidelity", Sdf.ValueTypeNames.String).Set(
        "manufacturer_asset"
    )
    robot_prim.CreateAttribute("rad:robot:controller", Sdf.ValueTypeNames.String).Set(
        "holonomic base + seven-axis arm IK + parallel gripper"
    )

    shield = UsdGeom.Xform.Define(stage, SHIELD_PATH)
    shield.AddTranslateOp().Set(Gf.Vec3d(0.78, -0.62, 0.0))
    shield_prim = shield.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(shield_prim).CreateRigidBodyEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(shield_prim).CreateMassAttr(2.8)
    for name, value_type, value in (
        ("rad:role", Sdf.ValueTypeNames.String, "shield"),
        ("rad:material:id", Sdf.ValueTypeNames.String, "tungsten_composite"),
        ("rad:material:mode", Sdf.ValueTypeNames.String, "solid"),
        ("rad:shield:movable", Sdf.ValueTypeNames.Bool, True),
        ("rad:shield:massKg", Sdf.ValueTypeNames.Double, 2.8),
        ("rad:manipulation:movable", Sdf.ValueTypeNames.Bool, True),
        ("rad:manipulation:graspFrame", Sdf.ValueTypeNames.String, "GraspFrame"),
    ):
        shield_prim.CreateAttribute(name, value_type).Set(value)

    add_cube(
        stage,
        f"{SHIELD_PATH}/BaseFoot",
        (0.30, 0.52, 0.065),
        (0.0, 0.0, 0.035),
        materials["dark_steel"],
        collision=True,
    )
    core = add_cube(
        stage,
        f"{SHIELD_PATH}/TungstenCore",
        (0.025, 0.42, 0.62),
        (0.0, 0.0, 0.37),
        materials["lead"],
        collision=True,
    )
    core.GetPrim().CreateAttribute("rad:material:id", Sdf.ValueTypeNames.String).Set(
        "tungsten_composite"
    )
    for side_name, x_m in (("RobotSide", -0.020), ("SourceSide", 0.020)):
        add_cube(
            stage,
            f"{SHIELD_PATH}/Jacket{side_name}",
            (0.012, 0.46, 0.66),
            (x_m, 0.0, 0.37),
            materials["shield_skin"],
            collision=True,
        )
    for y_m in (-0.235, 0.235):
        add_cube(
            stage,
            f"{SHIELD_PATH}/Edge_{'L' if y_m > 0.0 else 'R'}",
            (0.060, 0.025, 0.68),
            (0.0, y_m, 0.37),
            materials["yellow_dark"],
        )
    for y_m in (-0.10, 0.10):
        add_cube(
            stage,
            f"{SHIELD_PATH}/HandleStand_{'L' if y_m > 0.0 else 'R'}",
            (0.12, 0.025, 0.025),
            (-0.075, y_m, 0.54),
            materials["orange"],
        )
    add_cube(
        stage,
        f"{SHIELD_PATH}/Handle",
        (0.025, 0.23, 0.030),
        (-0.14, 0.0, 0.54),
        materials["orange"],
    )
    grasp = UsdGeom.Xform.Define(stage, f"{SHIELD_PATH}/GraspFrame")
    grasp.AddTranslateOp().Set(Gf.Vec3d(-0.14, 0.0, 0.54))

    # A corroded process-pipe spool carries an adhered, spatially varying
    # surface source. The object itself is not a homogeneous volume source.
    source_path = "/World/ShieldManipulation/ContaminatedValveSpool"
    source = UsdGeom.Xform.Define(stage, source_path)
    source_prim = source.GetPrim()
    source_prim.CreateAttribute("rad:role", Sdf.ValueTypeNames.String).Set("contaminated_object")
    source_prim.CreateAttribute("rad:manipulation:movable", Sdf.ValueTypeNames.Bool).Set(True)
    add_cylinder(
        stage,
        f"{source_path}/VerticalPipe",
        0.20,
        0.88,
        (1.95, -0.62, 0.46),
        materials["pipe"],
        collision=True,
    )
    for z_m in (0.10, 0.80):
        add_cylinder(
            stage,
            f"{source_path}/Flange_{int(z_m * 100):02d}",
            0.31,
            0.10,
            (1.95, -0.62, z_m),
            materials["dark_steel"],
            collision=True,
        )
    add_cylinder(
        stage,
        f"{source_path}/BranchPipe",
        0.13,
        0.72,
        (1.62, -0.62, 0.56),
        materials["pipe"],
        axis="X",
        collision=True,
    )
    add_cylinder(
        stage,
        f"{source_path}/ValveBody",
        0.23,
        0.28,
        (1.30, -0.62, 0.56),
        materials["corrosion"],
        axis="X",
        collision=True,
    )
    surface_source = create_irregular_cylindrical_surface_source(
        stage,
        f"{source_path}/AdheredSurfaceContamination",
        center_world_m=(1.95, -0.62, 0.46),
        radius_m=0.20,
        height_m=0.76,
        total_activity_bq=8.5e8,
    )

    return {
        "robot_asset": assets_root + RIDGEBACK_FRANKA_ASSET,
        "robot_path": MANIPULATOR_PATH,
        "shield_path": SHIELD_PATH,
        "contaminated_object_path": source_path,
        "surface_source": surface_source,
        "shield_mass_kg": 2.8,
    }


def build_route(grid: SurfaceSourceGrid) -> tuple[tuple[float, float, float], list[RouteSegment]]:
    low_x = float(grid.center_world_m[0] - grid.size_x_m * 0.5)
    high_x = float(grid.center_world_m[0] + grid.size_x_m * 0.5)
    rows = np.linspace(-0.60, 0.60, 4)
    forward_start_x = low_x - HEAD_X_M
    forward_end_x = high_x - HEAD_X_M
    reverse_start_x = high_x + HEAD_X_M
    reverse_end_x = low_x + HEAD_X_M
    start = (forward_start_x, float(rows[0]), 0.36)
    segments: list[RouteSegment] = []
    for row_index, row in enumerate(rows):
        if row_index % 2 == 0:
            segments.append(
                RouteSegment((forward_end_x, float(row)), True, f"sealed wash lane {row_index + 1}")
            )
            if row_index + 1 < len(rows):
                segments.append(
                    RouteSegment(
                        (reverse_start_x, float(rows[row_index + 1])),
                        False,
                        "head closed / reposition",
                    )
                )
        else:
            segments.append(
                RouteSegment((reverse_end_x, float(row)), True, f"sealed wash lane {row_index + 1}")
            )
            if row_index + 1 < len(rows):
                segments.append(
                    RouteSegment(
                        (forward_start_x, float(rows[row_index + 1])),
                        False,
                        "head closed / reposition",
                    )
                )
    return start, segments


def transform_and_pose(stage) -> tuple[Gf.Matrix4d, tuple[float, float, float], float]:
    transform = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT_PATH)).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    position = transform.ExtractTranslation()
    x_axis = transform.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0)).GetNormalized()
    yaw = math.atan2(x_axis[1], x_axis[0])
    return transform, (float(position[0]), float(position[1]), float(position[2])), yaw


class RigidArounderController:
    def __init__(self, stage) -> None:
        self.stage = stage
        self.prim = RigidPrim(ROBOT_PATH)
        self.prim.initialize_cpp_data_view()
        self.last_command = TwistCommand()

    def command_twist(self, command: TwistCommand) -> None:
        self.last_command = command
        transform, _position, _yaw = transform_and_pose(self.stage)
        x_axis = transform.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0)).GetNormalized()
        y_axis = transform.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0)).GetNormalized()
        velocity = x_axis * command.linear_x_m_s + y_axis * command.linear_y_m_s
        self.prim.set_velocities(
            linear_velocities=np.asarray([[velocity[0], velocity[1], 0.0]], dtype=np.float32),
            angular_velocities=np.asarray([[0.0, 0.0, command.angular_z_rad_s]], dtype=np.float32),
        )

    def command_joint(self, command: JointCommand) -> None:
        raise RuntimeError(f"Arounder base has no exposed joint group: {command.names}")

    def command_gripper(self, name: str, fraction_closed: float) -> None:
        raise RuntimeError(f"Arounder has a process head, not a gripper: {name}")


class ArounderAutoController:
    def __init__(self, stage, route: list[RouteSegment], time_scale: float) -> None:
        self.stage = stage
        self.route = route
        self.time_scale = time_scale
        self.index = 0
        self.completed = False
        self.water_requested = False
        self.phase = "initial alignment"

    def __call__(self, _dt_s: float) -> TwistCommand:
        self.water_requested = False
        if self.completed:
            self.phase = "post-treatment verification"
            return TwistCommand()
        segment = self.route[self.index]
        _transform, position, yaw = transform_and_pose(self.stage)
        dx = segment.target_xy_m[0] - position[0]
        dy = segment.target_xy_m[1] - position[1]
        distance_m = math.hypot(dx, dy)
        if distance_m < 0.075:
            self.index += 1
            if self.index >= len(self.route):
                self.completed = True
                self.phase = "post-treatment verification"
                return TwistCommand()
            segment = self.route[self.index]
            dx = segment.target_xy_m[0] - position[0]
            dy = segment.target_xy_m[1] - position[1]
            distance_m = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)
        yaw_error = (desired_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
        angular = float(np.clip(1.45 * yaw_error, -0.62, 0.62))
        if abs(yaw_error) > 0.16:
            self.phase = "aligning crawler / water isolated"
            return TwistCommand(angular_z_rad_s=angular)
        if segment.wash:
            treatment_speed = PHYSICAL_WASH_SPEED_M_S * self.time_scale
            linear = min(treatment_speed, 1.1 * distance_m)
            self.water_requested = abs(yaw_error) < 0.065 and linear > 0.015
            self.phase = segment.label if self.water_requested else "head sealing"
        else:
            linear = min(0.34, 1.25 * distance_m)
            self.phase = segment.label
        return TwistCommand(linear_x_m_s=linear, angular_z_rad_s=angular)


class ArounderWindow:
    def __init__(self, initial_activity_bq: float, pressure_mpa: float) -> None:
        self.initial_activity_bq = initial_activity_bq
        self.window = ui.Window("Arounder-Type Decontamination", width=500, height=590)
        self.progress_model = ui.SimpleFloatModel(0.0)
        with self.window.frame, ui.VStack(spacing=7):
            ui.Label("IRID / Hitachi-GE AROUNDER-TYPE RESEARCH REPLICA")
            ui.Label("Crawler + 4-DOF hydraulic arm + sealed recovery head")
            ui.Label("W600 x D1529 x H1255 mm folded | 840 kg robot/arm")
            ui.Separator()
            self.status = ui.Label("Phase: initializing")
            self.seal = ui.Label("Head seal: OPEN | jet interlocked")
            self.process = ui.Label(f"Process: {pressure_mpa:.0f} MPa | 450 mm nozzle stroke")
            self.speed = ui.Label("Treatment rate: 2.0 m2/h | accelerated view")
            self.activity = ui.Label(f"Surface activity: {initial_activity_bq:,.0f} Bq")
            self.df = ui.Label("Decontamination factor: 1.00")
            self.treated = ui.Label("Treated area: 0.0%")
            self.route = ui.Label("Route segment: 0 / 0")
            self.water = ui.Label("Water: 0.0 L supplied / 0.0 L recovered")
            self.recovery = ui.Label("Recovery: 0.0% | discharge: 0.0 L")
            self.waste = ui.Label("Captured activity: 0 Bq")
            self.clock = ui.Label("Plant operation time: 00:00:00")
            ui.ProgressBar(model=self.progress_model, height=22)
            ui.Separator()
            ui.Label("Water jets remain inside the brush-sealed head.")
            ui.Label("Rear hoses connect to the remote pump, reel, and 1000 L tank.")
            ui.Label("WASD/QE or gamepad can override autonomous motion.")

    def update(self, grid, auto, sealed, state, physical_elapsed_s, pressure_mpa) -> None:
        self.status.text = f"Phase: {auto.phase}"
        self.seal.text = (
            "Head seal: CLOSED | jet + vacuum ON" if sealed else "Head seal: OPEN | jet interlocked"
        )
        self.process.text = f"Process: {pressure_mpa:.0f} MPa | 450 mm internal nozzle scan"
        self.activity.text = f"Surface activity: {grid.total_activity_bq:,.0f} Bq"
        df = grid.initial_total_activity_bq / max(grid.total_activity_bq, 1e-12)
        self.df.text = f"Decontamination factor: {df:.2f}"
        self.treated.text = f"Treated area: {grid.treated_fraction * 100.0:.1f}%"
        self.route.text = (
            f"Route segment: {min(auto.index + 1, len(auto.route))} / {len(auto.route)}"
        )
        self.water.text = (
            f"Water: {state.applied_water_l:.1f} L supplied / "
            f"{state.recovered_water_l:.1f} L recovered"
        )
        recovery = state.recovered_water_l / max(state.applied_water_l, 1e-12) * 100.0
        self.recovery.text = (
            f"Recovery: {recovery:.1f}% | discharge: {state.discharged_water_l:.1f} L"
        )
        self.waste.text = f"Captured activity: {state.captured_activity_bq:,.0f} Bq"
        hours = int(physical_elapsed_s // 3600)
        minutes = int((physical_elapsed_s % 3600) // 60)
        seconds = int(physical_elapsed_s % 60)
        self.clock.text = (
            f"Plant operation time: {hours:02d}:{minutes:02d}:{seconds:02d} "
            f"({ARGS.time_scale:.0f}x view)"
        )
        self.progress_model.set_value(float(min(grid.removed_fraction, 1.0)))


def operation_parameters(mode: str) -> tuple[float, float]:
    return {
        "surface-wash": (50.0, 0.035),
        "coating-strip": (120.0, 0.060),
        "scarify": (200.0, 0.085),
    }[mode]


def create_water_spec(mode: str) -> WaterJetSpec:
    pressure_mpa, _coefficient = operation_parameters(mode)
    return WaterJetSpec(
        flow_rate_l_min=6.0,
        pressure_mpa=pressure_mpa,
        reference_pressure_mpa=50.0,
        spray_cone_angle_deg=14.0,
        minimum_footprint_radius_m=0.080,
        min_standoff_m=0.10,
        max_standoff_m=0.26,
        max_incidence_angle_deg=12.0,
        max_surface_speed_m_s=0.006,
        water_recovery_fraction=0.97,
        surface_water_retention_fraction=0.01,
        require_wastewater_collection=True,
    )


def create_water_treatment(mode: str) -> WaterJetTreatment:
    _pressure_mpa, coefficient = operation_parameters(mode)
    return WaterJetTreatment(
        removal_coefficient_m2_per_l=coefficient,
        washability_mean=1.0,
        washability_std=0.0,
        activity_capture_fraction=0.98,
        runoff_redeposition_fraction=0.10,
    )


def update_tether(visuals: RobotVisuals, transform, roller_points) -> None:
    starts = (
        transform.Transform(Gf.Vec3d(-0.70, -0.15, 0.48)),
        transform.Transform(Gf.Vec3d(-0.70, 0.02, 0.44)),
        transform.Transform(Gf.Vec3d(-0.70, 0.15, 0.52)),
    )
    reel = (-5.15, -2.55, 0.78)
    for index, (attr, start) in enumerate(zip(visuals.tether_points_attrs, starts, strict=True)):
        lateral = (index - 1) * 0.045
        points = (
            (float(start[0]), float(start[1]), float(start[2])),
            (float(start[0] - 0.55), float(start[1] + lateral), 0.18),
            (roller_points[1][0], roller_points[1][1] + lateral, roller_points[1][2]),
            (roller_points[0][0], roller_points[0][1] + lateral, roller_points[0][2]),
            (reel[0], reel[1] + lateral, reel[2]),
        )
        attr.Set([Gf.Vec3f(*point) for point in points])


def update_process_visuals(visuals: RobotVisuals, scan_y_m: float, sealed: bool) -> None:
    visuals.nozzle_translate_op.Set(Gf.Vec3d(0.0, scan_y_m, 0.0))
    visibility = UsdGeom.Tokens.inherited if sealed else UsdGeom.Tokens.invisible
    for cone in visuals.spray_prims:
        cone.GetVisibilityAttr().Set(visibility)
    visuals.seal_color_attr.Set(
        [Gf.Vec3f(0.04, 0.86, 0.18) if sealed else Gf.Vec3f(0.88, 0.04, 0.02)]
    )


def update_surface_visuals(grid, water_process, color_attributes, activity_attributes) -> None:
    colors = water_process.visual_color_rgb()
    epoxy = np.asarray([0.19, 0.27, 0.25])
    colors = 0.40 * colors + 0.60 * epoxy
    for index, (color_attr, activity_attr) in enumerate(
        zip(color_attributes, activity_attributes, strict=True)
    ):
        color_attr.Set([Gf.Vec3f(*colors[index])])
        activity_attr.Set(float(grid.activity_bq[index]))


def request_capture(path: Path) -> None:
    path.unlink(missing_ok=True)
    viewport = viewport_utility.get_active_viewport()
    if viewport is None:
        raise RuntimeError("active viewport is unavailable")
    viewport_utility.capture_viewport_to_file(viewport, str(path))


def _set_link_geometry(stage, path: str, start_xyz, end_xyz, width_m: float) -> None:
    start = np.asarray(start_xyz, dtype=np.float64)
    end = np.asarray(end_xyz, dtype=np.float64)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        raise ValueError("animated link endpoints must be distinct")
    horizontal = math.hypot(float(delta[0]), float(delta[1]))
    rotation = Gf.Vec3f(
        0.0,
        -math.degrees(math.atan2(float(delta[2]), horizontal)),
        math.degrees(math.atan2(float(delta[1]), float(delta[0]))),
    )
    xformable = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    operations = xformable.GetOrderedXformOps()
    if len(operations) != 3:
        raise RuntimeError(f"expected translate/rotate/scale operations on {path}")
    operations[0].Set(Gf.Vec3d(*(0.5 * (start + end))))
    operations[1].Set(rotation)
    operations[2].Set(Gf.Vec3f(length, width_m, width_m))


def _solve_fabrik_chain(base_xyz, target_xyz, lengths_m: tuple[float, ...]) -> np.ndarray:
    """Solve a small fixed-length serial chain for the rendered distal arm."""

    base = np.asarray(base_xyz, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)
    reach = float(sum(lengths_m))
    separation = float(np.linalg.norm(target - base))
    if separation >= reach:
        direction = (target - base) / max(separation, 1e-12)
        points = [base]
        for length_m in lengths_m:
            points.append(points[-1] + length_m * direction)
        return np.asarray(points)

    interpolation = np.linspace(0.0, 1.0, len(lengths_m) + 1)[:, None]
    bend = np.asarray((0.20, -0.10, 0.34), dtype=np.float64)
    points = base + interpolation * (target - base)
    points += np.sin(math.pi * interpolation) * bend
    points[0] = base
    points[-1] = target
    for _ in range(24):
        points[-1] = target
        for index in range(len(lengths_m) - 1, -1, -1):
            delta = points[index] - points[index + 1]
            distance = float(np.linalg.norm(delta))
            points[index] = points[index + 1] + lengths_m[index] * delta / max(distance, 1e-12)
        points[0] = base
        for index, length_m in enumerate(lengths_m):
            delta = points[index + 1] - points[index]
            distance = float(np.linalg.norm(delta))
            points[index + 1] = points[index] + length_m * delta / max(distance, 1e-12)
        if float(np.linalg.norm(points[-1] - target)) <= 1e-5:
            break
    return points


def _animate_high_reach_arm(stage, target_local_xyz: tuple[float, float, float]) -> None:
    base = (0.0, 4.12, 9.40)
    lengths = (0.30, 0.28, 0.26, 0.24, 0.22, 0.20, 0.18)
    points = _solve_fabrik_chain(base, target_local_xyz, lengths)
    for index, (start, end) in enumerate(zip(points[:-1], points[1:], strict=True), start=1):
        _set_link_geometry(
            stage,
            f"{HIGH_REACH_ROBOT_PATH}/DistalManipulator/Link{index}",
            start,
            end,
            0.148 - 0.008 * index,
        )
        joint = UsdGeom.Xformable(
            stage.GetPrimAtPath(f"{HIGH_REACH_ROBOT_PATH}/DistalManipulator/Joint{index}")
        )
        joint.GetOrderedXformOps()[0].Set(Gf.Vec3d(*end))

    head = UsdGeom.Xformable(stage.GetPrimAtPath(f"{HIGH_REACH_ROBOT_PATH}/DeconHead"))
    operations = head.GetOrderedXformOps()
    translate = operations[0] if operations else head.AddTranslateOp()
    initial = np.asarray((0.0, 4.96, 9.38), dtype=np.float64)
    translate.Set(Gf.Vec3d(*(np.asarray(target_local_xyz) - initial)))


def _high_reach_scan_target(video_time_s: float, duration_s: float) -> tuple[float, float, float]:
    lead_s = min(1.5, duration_s * 0.10)
    tail_s = min(2.0, duration_s * 0.12)
    scan_duration_s = max(duration_s - lead_s - tail_s, 1e-6)
    if video_time_s < lead_s:
        progress = 0.0
    elif video_time_s >= duration_s - tail_s:
        progress = 1.0
    else:
        progress = (video_time_s - lead_s) / scan_duration_s
    rows = 6
    row_position = min(progress, 1.0 - 1e-9) * rows
    row = min(int(row_position), rows - 1)
    phase = row_position - row
    if row % 2:
        phase = 1.0 - phase
    x_m = -1.05 + 2.10 * phase
    z_m = 8.70 + row * (1.32 / (rows - 1))
    return (x_m, 4.96, z_m)


def _update_high_wall_visuals(stage, grid: SurfaceSourceGrid) -> None:
    concrete = np.asarray([0.38, 0.40, 0.39])
    fraction = np.divide(
        grid.activity_bq,
        grid.initial_activity_bq,
        out=np.zeros_like(grid.activity_bq),
        where=grid.initial_activity_bq > 0.0,
    )
    overlay_colors = 0.62 * grid.color_rgb() + 0.38 * concrete
    colors = concrete + fraction[:, None] * (overlay_colors - concrete)
    for index in np.flatnonzero(grid.initial_activity_bq > 0.0):
        prim = stage.GetPrimAtPath(f"{HIGH_WALL_SOURCE_PATH}/Cell_{index:04d}")
        if not prim.IsValid():
            continue
        UsdGeom.Imageable(prim).GetVisibilityAttr().Set(
            UsdGeom.Tokens.invisible if fraction[index] < 0.10 else UsdGeom.Tokens.inherited
        )
        UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([Gf.Vec3f(*colors[index])])
        prim.GetAttribute("rad:source:activityBq").Set(float(grid.activity_bq[index]))


def _update_radiation_visualization(
    visuals: RadiationVisuals,
    cell_contributions: tuple[H100CellContribution, ...],
    cell_count: int,
    video_time_s: float,
) -> RadiationVisualizationFrame:
    fluence_weights = np.zeros(cell_count, dtype=np.float64)
    contributions_by_index: dict[int, H100CellContribution] = {}
    for item in cell_contributions:
        fluence_weights[item.cell_index] = item.incident_fluence.fluence_rate_m2_s
        contributions_by_index[item.cell_index] = item
    total_fluence_rate = float(np.sum(fluence_weights))
    fluence_fraction = float(
        np.clip(
            total_fluence_rate / max(visuals.initial_total_fluence_rate_m2_s, 1e-12),
            0.0,
            1.0,
        )
    )
    visible_ray_count = (
        min(
            len(cell_contributions),
            max(2, int(round(visuals.maximum_ray_count * fluence_fraction**0.78))),
        )
        if cell_contributions
        else 0
    )
    selected_cell_indices = contribution_weighted_sample_without_replacement(
        fluence_weights,
        visible_ray_count,
        seed=RADIATION_VISUALIZATION_SAMPLE_SEED,
    )
    detector_position = np.asarray(H100_DETECTOR_POSITION_WORLD_M, dtype=np.float64)
    selected_sources: list[np.ndarray] = []
    for ray_index, curve in enumerate(visuals.ray_curves):
        imageable = UsdGeom.Imageable(curve.GetPrim())
        if ray_index < len(selected_cell_indices):
            contribution = contributions_by_index[selected_cell_indices[ray_index]]
            source = np.asarray(contribution.source_position_world_m, dtype=np.float64)
            selected_sources.append(source)
            relative_cell_fluence = float(
                np.clip(
                    contribution.incident_fluence.fluence_rate_m2_s
                    / max(visuals.initial_maximum_cell_fluence_rate_m2_s, 1e-12),
                    0.0,
                    1.0,
                )
            )
            imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
            curve.GetPointsAttr().Set(
                [
                    Gf.Vec3f(*(float(value) for value in source)),
                    Gf.Vec3f(*(float(value) for value in detector_position)),
                ]
            )
            curve.GetDisplayOpacityAttr().Set([0.05 + 0.37 * math.sqrt(relative_cell_fluence)])
            curve.GetWidthsAttr().Set([0.010 + 0.020 * math.sqrt(relative_cell_fluence)])
            curve.GetPrim().GetAttribute("rad:visualization:sourceCellIndex").Set(
                contribution.cell_index
            )
            curve.GetPrim().GetAttribute("rad:visualization:cellFluenceRateM2S").Set(
                contribution.incident_fluence.fluence_rate_m2_s
            )
        else:
            imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    photon_positions = []
    for ray_index, source in enumerate(selected_sources):
        for pulse_index in range(2):
            phase = (video_time_s * 0.72 + ray_index * 0.173 + pulse_index * 0.5) % 1.0
            position = source + phase * (detector_position - source)
            photon_positions.append(Gf.Vec3f(*(float(value) for value in position)))
    visuals.photon_points_attr.Set(photon_positions)
    selected_fluence_rate = float(np.sum(fluence_weights[list(selected_cell_indices)]))
    return RadiationVisualizationFrame(
        visible_ray_count=len(selected_cell_indices),
        selected_cell_indices=selected_cell_indices,
        total_fluence_rate_m2_s=total_fluence_rate,
        selected_fluence_fraction=(
            selected_fluence_rate / total_fluence_rate if total_fluence_rate > 0.0 else 0.0
        ),
    )


def _h100_cell_contributions(grid: SurfaceSourceGrid) -> tuple[H100CellContribution, ...]:
    """Transport each live wall cell to the H100 while retaining its identity."""

    detector_position = np.asarray(H100_DETECTOR_POSITION_WORLD_M, dtype=np.float64)
    contributions: list[H100CellContribution] = []
    for index, (source_position, activity_bq) in enumerate(
        zip(grid.centers_world_m, grid.activity_bq, strict=True)
    ):
        if activity_bq <= 0.0:
            continue
        travel = detector_position - np.asarray(source_position, dtype=np.float64)
        distance_squared_m2 = float(np.dot(travel, travel))
        if distance_squared_m2 <= 1e-12:
            continue
        direction = travel / math.sqrt(distance_squared_m2)
        source_position_world_m = tuple(float(value) for value in source_position)
        contributions.append(
            H100CellContribution(
                cell_index=index,
                source_position_world_m=source_position_world_m,
                incident_fluence=IncidentParticleFluence(
                    radiation_type=RadiationType.GAMMA,
                    energy_kev=CS137_GAMMA_ENERGY_KEV,
                    fluence_rate_m2_s=(
                        float(activity_bq)
                        * CS137_GAMMA_YIELD
                        / (4.0 * math.pi * distance_squared_m2)
                    ),
                    arrival_direction_world=tuple(float(value) for value in direction),
                    source_id=f"high_wall_cell_{index:04d}_cs137",
                ),
            )
        )
    return tuple(contributions)


def _measure_h100(
    model: ParametricDetectorModel,
    cell_contributions: tuple[H100CellContribution, ...],
    frame_index: int,
):
    return model.measure(
        MeasurementRequest(
            pose=DetectorPose(
                "h100_rover",
                H100_DETECTOR_POSITION_WORLD_M,
                forward_world=(0.0, 1.0, 0.0),
            ),
            incident_fluence=tuple(item.incident_fluence for item in cell_contributions),
            integration_time_s=1.0,
            rng=np.random.default_rng(20_260_903 + frame_index),
        )
    )


def _ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100.0)))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def _write_h100_telemetry(
    telemetry: list[H100TelemetryRow],
    *,
    fps: int,
    csv_path: Path,
    subtitle_path: Path,
) -> None:
    fieldnames = (
        "video_time_s",
        "surface_activity_bq",
        "removed_fraction",
        "expected_count_rate_cps",
        "observed_count_rate_cps",
        "dose_rate_usv_h",
        "incident_fluence_rate_m2_s",
        "visible_radiation_paths",
        "visualized_cell_indices",
        "visualized_cell_fluence_fraction",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(telemetry)

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1280\n"
        "PlayResY: 720\n"
        "WrapStyle: 2\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding\n"
        "Style: Header,DejaVu Sans,24,&H0053E8FF,&H0053E8FF,&H00000000,"
        "&H00000000,-1,0,0,0,100,100,0,0,1,1.5,0,7,0,0,0,1\n"
        "Style: Label,DejaVu Sans,22,&H00FFFFFF,&H00FFFFFF,&H00000000,"
        "&H00000000,0,0,0,0,100,100,0,0,1,1.5,0,7,0,0,0,1\n"
        "Style: Number,DejaVu Sans Mono,22,&H0053E8FF,&H0053E8FF,&H00000000,"
        "&H00000000,-1,0,0,0,100,100,0,0,1,1.5,0,9,0,0,0,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    initial_count_rate = telemetry[0]["expected_count_rate_cps"]
    video_end = _ass_timestamp(len(telemetry) / fps)
    static_labels = (
        ("Header", 34, 27, "H3D H100  |  4PI OMNIDIRECTIONAL CS-137 MONITOR"),
        ("Label", 760, 27, "WALL SOURCE -> H100"),
        ("Label", 34, 59, "COUNT RATE"),
        ("Label", 265, 59, "cps"),
        ("Label", 350, 59, "DOSE RATE"),
        ("Label", 575, 59, "uSv/h"),
        ("Label", 760, 59, "VISIBLE GAMMA PATHS"),
        ("Label", 34, 88, "SURFACE ACTIVITY"),
        ("Label", 397, 88, "Bq"),
        ("Label", 470, 88, "REMOVED"),
        ("Label", 662, 88, "%"),
        ("Label", 34, 117, "LIVE RESPONSE DROP"),
        ("Label", 350, 117, "%"),
        ("Label", 400, 117, "RESPONSE: SYNTHETIC / BODY + 4PI FOV: H100 SPEC"),
    )
    events = [
        f"Dialogue: 0,0:00:00.00,{video_end},{style},,0,0,0,,"
        rf"{{\pos({x_position},{y_position})}}{label}"
        for style, x_position, y_position, label in static_labels
    ]
    # Five numeric updates per second remain readable and align exactly with a
    # 30 fps video. Labels are single, full-duration events, so their glyphs
    # are never torn down and re-rasterized while the measurements change.
    hud_update_frames = max(1, int(round(fps / 5.0)))
    for frame_index in range(0, len(telemetry), hud_update_frames):
        row = telemetry[frame_index]
        start = _ass_timestamp(frame_index / fps)
        end = _ass_timestamp(min(frame_index + hud_update_frames, len(telemetry)) / fps)
        count_drop = 1.0 - row["expected_count_rate_cps"] / max(initial_count_rate, 1e-12)
        values = (
            (248, 59, f"{row['expected_count_rate_cps']:.2f}"),
            (558, 59, f"{row['dose_rate_usv_h']:.3f}"),
            (1035, 59, f"{row['visible_radiation_paths']:.0f}"),
            (380, 88, f"{row['surface_activity_bq']:,.0f}"),
            (645, 88, f"{row['removed_fraction'] * 100.0:.1f}"),
            (333, 117, f"{max(0.0, count_drop) * 100.0:.1f}"),
        )
        for x_position, y_position, value in values:
            events.append(
                f"Dialogue: 1,{start},{end},Number,,0,0,0,,"
                rf"{{\pos({x_position},{y_position})}}{value}"
            )
    subtitle_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


def _ffmpeg_filter_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def render_high_reach_decontamination_video(
    world,
    stage,
    grid: SurfaceSourceGrid,
    radiation_visuals: RadiationVisuals,
) -> dict:
    duration_s = float(ARGS.video_seconds)
    fps = int(ARGS.video_fps)
    frame_count = int(round(duration_s * fps))
    frame_dir = ARGS.output / ".high_reach_video_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for stale_frame in frame_dir.glob("frame_*.png"):
        stale_frame.unlink()
    duration_label = f"{duration_s:g}".replace(".", "p")
    video_path = ARGS.output / f"high_reach_decontamination_with_h100_{duration_label}s.mp4"
    telemetry_path = ARGS.output / "h100_measurements.csv"
    subtitle_path = ARGS.output / ".h100_measurement_hud.ass"
    video_path.unlink(missing_ok=True)

    set_camera_view(
        eye=np.asarray([-11.5, -21.0, 8.4]),
        target=np.asarray([0.30, 3.30, 5.20]),
        camera_prim_path="/OmniverseKit_Persp",
    )
    for _ in range(90):
        world.step(render=True)

    tool = DecontaminationTool(
        length_m=0.72,
        width_m=0.58,
        rate_constant_s_inv=0.48,
        max_contact_distance_m=0.09,
        max_surface_speed_m_s=3.0,
    )
    initial_activity_bq = grid.total_activity_bq
    h100_model = ParametricDetectorModel(popular_detector_catalog()["h3d_h100_omni"])
    telemetry: list[H100TelemetryRow] = []
    previous_target = np.asarray(_high_reach_scan_target(0.0, duration_s), dtype=np.float64)
    active_start_s = min(1.5, duration_s * 0.10)
    active_end_s = duration_s - min(2.0, duration_s * 0.12)
    for frame_index in range(frame_count):
        video_time_s = frame_index / fps
        target_local = np.asarray(
            _high_reach_scan_target(video_time_s, duration_s), dtype=np.float64
        )
        _animate_high_reach_arm(stage, tuple(float(value) for value in target_local))
        if active_start_s <= video_time_s < active_end_s:
            speed_m_s = float(np.linalg.norm(target_local - previous_target) * fps)
            grid.apply_tool(
                tool,
                tool_center_world_m=(
                    float(target_local[0]),
                    float(target_local[1] + 0.70),
                    float(target_local[2]),
                ),
                tool_yaw_rad=0.0,
                surface_speed_m_s=speed_m_s,
                dt_s=5.0 / fps,
            )
            _update_high_wall_visuals(stage, grid)
        cell_contributions = _h100_cell_contributions(grid)
        radiation_frame = _update_radiation_visualization(
            radiation_visuals,
            cell_contributions,
            len(grid.activity_bq),
            video_time_s,
        )
        h100_reading = _measure_h100(h100_model, cell_contributions, frame_index)
        telemetry.append(
            {
                "video_time_s": video_time_s,
                "surface_activity_bq": grid.total_activity_bq,
                "removed_fraction": grid.removed_fraction,
                "expected_count_rate_cps": h100_reading.expected_count_rate_cps,
                "observed_count_rate_cps": (
                    h100_reading.observed_counts / h100_reading.integration_time_s
                ),
                "dose_rate_usv_h": h100_reading.dose_rate_usv_h,
                "incident_fluence_rate_m2_s": radiation_frame.total_fluence_rate_m2_s,
                "visible_radiation_paths": float(radiation_frame.visible_ray_count),
                "visualized_cell_indices": ";".join(
                    str(index) for index in radiation_frame.selected_cell_indices
                ),
                "visualized_cell_fluence_fraction": (radiation_frame.selected_fluence_fraction),
            }
        )
        previous_target = target_local
        world.step(render=True)
        request_capture(frame_dir / f"frame_{frame_index:04d}.png")
        world.step(render=True)
        if frame_index % fps == 0:
            print(
                f"HIGH_REACH_VIDEO t={video_time_s:.1f}s "
                f"removed={grid.removed_fraction * 100.0:.1f}%",
                flush=True,
            )

    for _ in range(30):
        world.step(render=True)
    capture_deadline = time.monotonic() + 5.0
    rendered_frames = len(list(frame_dir.glob("frame_*.png")))
    while rendered_frames < frame_count and time.monotonic() < capture_deadline:
        world.step(render=True)
        time.sleep(0.02)
        rendered_frames = len(list(frame_dir.glob("frame_*.png")))
    if rendered_frames != frame_count:
        raise RuntimeError(f"expected {frame_count} video frames, found {rendered_frames}")
    _write_h100_telemetry(
        telemetry,
        fps=fps,
        csv_path=telemetry_path,
        subtitle_path=subtitle_path,
    )
    video_filter = (
        "crop=1440:810:0:45,scale=1280:720,"
        "drawbox=x=18:y=16:w=1244:h=142:color=black@0.62:t=fill,"
        f"subtitles=filename='{_ffmpeg_filter_path(subtitle_path)}',format=yuv420p"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%04d.png"),
            "-vf",
            video_filter,
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
    for rendered_frame in frame_dir.glob("frame_*.png"):
        rendered_frame.unlink()
    frame_dir.rmdir()
    subtitle_path.unlink(missing_ok=True)
    return {
        "passed": True,
        "video": str(video_path),
        "duration_s": duration_s,
        "capture_fps": fps,
        "encoded_fps": 30,
        "resolution": [1280, 720],
        "frames": frame_count,
        "initial_activity_bq": initial_activity_bq,
        "final_activity_bq": grid.total_activity_bq,
        "removed_fraction": grid.removed_fraction,
        "measurement_robot": MEASUREMENT_ROBOT_PATH,
        "detector_model_id": "h3d_h100_omni",
        "detector_operating_mode": "omnidirectional scalar monitoring",
        "detector_response_data_status": "synthetic_validation_only",
        "initial_expected_count_rate_cps": telemetry[0]["expected_count_rate_cps"],
        "final_expected_count_rate_cps": telemetry[-1]["expected_count_rate_cps"],
        "initial_dose_rate_usv_h": telemetry[0]["dose_rate_usv_h"],
        "final_dose_rate_usv_h": telemetry[-1]["dose_rate_usv_h"],
        "measurement_telemetry_csv": str(telemetry_path),
        "transport_model": "per-cell Cs-137 yield and inverse-square fluence",
        "surface_source": HIGH_WALL_SOURCE_PATH,
        "decontamination_model": "contact-footprint cumulative-exposure decay",
        "radiation_visualization": (
            "deterministic contribution-weighted sample of per-cell H100 fluence paths"
        ),
        "radiation_visualization_sampling": (
            "Gumbel-top-k probability-proportional-to-fluence without replacement"
        ),
        "radiation_visualization_sampling_seed": RADIATION_VISUALIZATION_SAMPLE_SEED,
        "radiation_visualization_transport_coupled": True,
        "radiation_visualization_affects_detector_response": False,
        "initial_visualized_cell_fluence_fraction": telemetry[0][
            "visualized_cell_fluence_fraction"
        ],
        "final_visualized_cell_fluence_fraction": telemetry[-1]["visualized_cell_fluence_fraction"],
        "initial_visible_radiation_paths": int(telemetry[0]["visible_radiation_paths"]),
        "final_visible_radiation_paths": int(telemetry[-1]["visible_radiation_paths"]),
    }


def main() -> int:
    ARGS.output.mkdir(parents=True, exist_ok=True)
    result_path = ARGS.output / ("video_result.json" if ARGS.render_video else "result.json")
    before_path = ARGS.output / "before.png"
    tool_detail_path = ARGS.output / "decontamination_head_detail.png"
    high_reach_path = ARGS.output / "high_reach_decontamination.png"
    shield_manipulator_path = ARGS.output / "shield_manipulator.png"
    during_path = ARGS.output / "during.png"
    after_path = ARGS.output / "after.png"
    result: dict[str, object] = {"passed": False}
    world = None
    router = None
    try:
        world = World(stage_units_in_meters=1.0, physics_dt=DT_S, rendering_dt=DT_S)
        stage = omni.usd.get_context().get_stage()
        materials = create_materials(stage)
        high_reach_render = ARGS.render_video or (
            ARGS.render_only and ARGS.render_scene == "decontamination"
        )
        roller_points = create_environment(stage, materials, high_bay=high_reach_render)

        dome = UsdLux.DomeLight.Define(stage, "/World/ReactorBuilding/Ambient")
        dome.CreateIntensityAttr(340.0)
        dome.CreateColorAttr(Gf.Vec3f(0.56, 0.66, 0.76))

        if high_reach_render:
            grid = create_high_wall_surface_source(stage)
            robot_manifest = create_high_reach_robot(stage, materials)
            measurement_manifest = create_h100_measurement_robot(stage, materials)
            radiation_visuals = create_radiation_visualization(stage, grid)
            world.reset()
            if ARGS.render_video:
                video_result = render_high_reach_decontamination_video(
                    world,
                    stage,
                    grid,
                    radiation_visuals,
                )
                result = {
                    **video_result,
                    "scene": "13 m reactor-building high-wall decontamination",
                    "reference_robot": "RadCounter HighReach-10 research design",
                    **robot_manifest,
                    **measurement_manifest,
                }
                print("HIGH_REACH_VIDEO_RESULT " + json.dumps(result), flush=True)
                return 0
            set_camera_view(
                eye=np.asarray([-8.0, -15.2, 7.1]),
                target=np.asarray([0.0, 3.00, 4.85]),
                camera_prim_path="/OmniverseKit_Persp",
            )
            for _ in range(120):
                world.step(render=True)
            request_capture(high_reach_path)
            for _ in range(30):
                world.step(render=True)
            result = {
                "passed": True,
                "render_only": True,
                "scene": "13 m reactor-building high-bay decontamination",
                "reference_robot": "RadCounter HighReach-10 research design",
                "design_disclosure": (
                    "six-stage 10 m research extension of the documented 8 m "
                    "MHI SUPER-Giraffe architecture"
                ),
                **robot_manifest,
                "surface_source": HIGH_WALL_SOURCE_PATH,
                "surface_orientation": "vertical wall",
                "active_surface_cells": int(np.count_nonzero(grid.activity_bq)),
                "maximum_contamination_height_m": 10.45,
                **measurement_manifest,
                "image": str(high_reach_path),
            }
            print("HIGH_REACH_RENDER_RESULT " + json.dumps(result), flush=True)
            return 0

        if ARGS.render_scene == "shield-manipulation":
            if not ARGS.render_only:
                raise ValueError("shield-manipulation is a render-only scene")
            scene_manifest = create_shield_manipulation_scene(stage, materials)
            world.reset()
            set_camera_view(
                eye=np.asarray([4.35, -5.35, 2.65]),
                target=np.asarray([0.70, -0.55, 0.57]),
                camera_prim_path="/OmniverseKit_Persp",
            )
            for _ in range(150):
                world.step(render=True)
            request_capture(shield_manipulator_path)
            for _ in range(30):
                world.step(render=True)
            result = {
                "passed": True,
                "render_only": True,
                "scene": "reactor-building manipulator shield placement",
                "reference_robot": "Clearpath Ridgeback + Franka Emika Panda",
                "reference_urls": [RIDGEBACK_FRANKA_REFERENCE],
                **scene_manifest,
                "image": str(shield_manipulator_path),
            }
            print("SHIELD_MANIPULATOR_RENDER_RESULT " + json.dumps(result), flush=True)
            return 0

        grid, color_attributes, activity_attributes = create_surface_source(stage)
        start_xyz, route = build_route(grid)
        visuals = create_robot(stage, materials, start_xyz)

        world.reset()
        controller = RigidArounderController(stage)
        auto = ArounderAutoController(stage, route, ARGS.time_scale)
        router = IsaacRobotInputRouter({ROBOT_ID: controller})
        router.attach_devices(ROBOT_ID, ROBOT_ID)
        router.register_auto_controller(ROBOT_ID, auto)

        pressure_mpa, _coefficient = operation_parameters(ARGS.operation_mode)
        panel = ArounderWindow(grid.initial_total_activity_bq, pressure_mpa)
        water_state = WaterDecontaminationState(
            supply_remaining_l=1000.0, wastewater_capacity_l=1000.0
        )
        water_process = WaterSurfaceDecontaminator(
            grid,
            water_state,
            create_water_treatment(ARGS.operation_mode),
            washability=grid.efficiency,
            runoff_direction_world_xy=(0.0, -1.0),
        )
        water_spec = create_water_spec(ARGS.operation_mode)
        dry_tool = DecontaminationTool(
            length_m=0.34,
            width_m=HEAD_WIDTH_M,
            rate_constant_s_inv=0.02,
            max_contact_distance_m=0.08,
            max_surface_speed_m_s=0.006,
        )

        set_camera_view(
            eye=np.asarray([6.8, -7.6, 5.15]),
            target=np.asarray([0.15, 0.15, 0.48]),
            camera_prim_path="/OmniverseKit_Persp",
        )

        for _ in range(55):
            world.step(render=True)
        request_capture(before_path)
        for _ in range(20):
            world.step(render=True)

        set_camera_view(
            eye=np.asarray([-0.10, -2.30, 1.55]),
            target=np.asarray([-1.12, -0.58, 0.20]),
            camera_prim_path="/OmniverseKit_Persp",
        )
        for _ in range(35):
            world.step(render=True)
        request_capture(tool_detail_path)
        for _ in range(20):
            world.step(render=True)

        if ARGS.render_only:
            result = {
                "passed": True,
                "render_only": True,
                "reference_robot": (
                    "Hitachi-GE / IRID Arounder low-section high-pressure-water "
                    "decontamination machine"
                ),
                "reference_urls": [HITACHI_REFERENCE, IRID_REFERENCE],
                "scene": "reactor-building floor decontamination",
                "surface_source": SOURCE_PATH,
                "surface_cells": len(grid.activity_bq),
                "active_surface_cells": int(np.count_nonzero(grid.activity_bq)),
                "surface_shape": "irregular spill/runoff field with detached droplets",
                "rectangular_work_zone_frame": False,
                "initial_activity_bq": grid.initial_total_activity_bq,
                "overview_image": str(before_path),
                "decontamination_head_detail_image": str(tool_detail_path),
            }
            print("AROUNDER_RENDER_RESULT " + json.dumps(result), flush=True)
            return 0

        set_camera_view(
            eye=np.asarray([6.8, -7.6, 5.15]),
            target=np.asarray([0.15, 0.15, 0.48]),
            camera_prim_path="/OmniverseKit_Persp",
        )
        for _ in range(20):
            world.step(render=True)

        initial_activity = grid.total_activity_bq
        start_position = transform_and_pose(stage)[1]
        previous_position = start_position
        path_length_m = 0.0
        contacted: set[int] = set()
        start_wall = time.monotonic()
        next_frame = start_wall
        last_log_second = -1
        frame_index = 0
        during_captured = False
        completion_wall: float | None = None
        last_blocked_reason: str | None = None

        while simulation_app.is_running():
            elapsed = time.monotonic() - start_wall
            if elapsed >= ARGS.duration:
                break
            router.update(DT_S)
            world.step(render=True)
            transform, position, yaw = transform_and_pose(stage)
            frame_distance = math.dist(position[:2], previous_position[:2])
            path_length_m += frame_distance
            scan_phase = 2.0 * math.pi * 0.52 * elapsed
            scan_y_m = 0.5 * HEAD_STROKE_M * math.sin(scan_phase)
            scan_visual_speed_m_s = abs(
                0.5 * HEAD_STROKE_M * 2.0 * math.pi * 0.52 * math.cos(scan_phase)
            )
            brush_point = transform.Transform(Gf.Vec3d(HEAD_X_M, scan_y_m, HEAD_BRUSH_Z_M))
            seal_gap_m = abs(float(brush_point[2]) - float(grid.center_world_m[2]))
            sealed = auto.water_requested and seal_gap_m <= 0.055
            update_process_visuals(visuals, scan_y_m, sealed)
            update_tether(visuals, transform, roller_points)

            physical_base_speed = frame_distance / max(DT_S * ARGS.time_scale, 1e-12)
            physical_scan_speed = scan_visual_speed_m_s / ARGS.time_scale
            process_speed = max(physical_base_speed, physical_scan_speed)
            step = None
            if sealed and ARGS.method == "water":
                nozzle = transform.Transform(Gf.Vec3d(HEAD_X_M, scan_y_m, HEAD_NOZZLE_Z_M))
                direction = transform.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0))
                step = water_process.apply(
                    water_spec,
                    nozzle_world_m=tuple(float(value) for value in nozzle),
                    jet_direction_world=tuple(float(value) for value in direction),
                    surface_speed_m_s=process_speed,
                    dt_s=DT_S * ARGS.time_scale,
                )
                last_blocked_reason = step.blocked_reason
            elif sealed:
                head_center = transform.Transform(Gf.Vec3d(HEAD_X_M, scan_y_m, HEAD_BRUSH_Z_M))
                step = grid.apply_tool(
                    dry_tool,
                    tool_center_world_m=tuple(float(value) for value in head_center),
                    tool_yaw_rad=yaw,
                    surface_speed_m_s=process_speed,
                    dt_s=DT_S * ARGS.time_scale,
                )
            if step is not None:
                contacted.update(step.contacted_cells)

            if frame_index % 6 == 0:
                update_surface_visuals(grid, water_process, color_attributes, activity_attributes)
            physical_elapsed_s = elapsed * ARGS.time_scale
            panel.update(grid, auto, sealed, water_state, physical_elapsed_s, pressure_mpa)
            if not during_captured and grid.removed_fraction >= 0.35:
                request_capture(during_path)
                during_captured = True
            if auto.completed:
                completion_wall = completion_wall or time.monotonic()
                if time.monotonic() - completion_wall >= 8.0:
                    break

            previous_position = position
            second = int(elapsed)
            if second != last_log_second:
                last_log_second = second
                recovery = water_state.recovered_water_l / max(water_state.applied_water_l, 1e-12)
                print(
                    f"AROUNDER_TELEMETRY t={elapsed:.1f} plant_t={physical_elapsed_s:.0f} "
                    f"phase={auto.phase!r} seal={sealed} pressure_mpa={pressure_mpa:.0f} "
                    f"activity_bq={grid.total_activity_bq:.0f} "
                    f"df={initial_activity / max(grid.total_activity_bq, 1e-12):.2f} "
                    f"treated_pct={grid.treated_fraction * 100.0:.1f} path_m={path_length_m:.2f} "
                    f"water_l={water_state.applied_water_l:.1f} "
                    f"recovery_pct={recovery * 100.0:.1f}",
                    flush=True,
                )
            frame_index += 1
            next_frame += DT_S
            sleep_s = next_frame - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)

        controller.command_twist(TwistCommand())
        update_process_visuals(visuals, 0.0, False)
        final_activity = grid.total_activity_bq
        end_position = transform_and_pose(stage)[1]
        update_surface_visuals(grid, water_process, color_attributes, activity_attributes)
        request_capture(after_path)
        for _ in range(40):
            world.step(render=True)

        applied = water_state.applied_water_l
        recovery_fraction = water_state.recovered_water_l / max(applied, 1e-12)
        decontamination_factor = initial_activity / max(final_activity, 1e-12)
        passed = bool(
            auto.completed
            and decontamination_factor >= 5.0
            and grid.treated_fraction >= 0.80
            and recovery_fraction >= 0.95
            and abs(water_process.mass_balance_error_bq) <= max(1e-5, initial_activity * 1e-10)
        )
        result = {
            "passed": passed,
            "reference_robot": (
                "Hitachi-GE / IRID Arounder low-section high-pressure-water decontamination machine"
            ),
            "reference_urls": [HITACHI_REFERENCE, IRID_REFERENCE],
            "isaac_sim_target": "6.0.1",
            "operation_mode": ARGS.operation_mode,
            "pressure_mpa": pressure_mpa,
            "target_treatment_rate_m2_h": TARGET_TREATMENT_RATE_M2_H,
            "time_scale": ARGS.time_scale,
            "robot_mass_kg": 840.0,
            "folded_dimensions_mm": [600, 1529, 1255],
            "nozzle_stroke_mm": 450,
            "route_completed": auto.completed,
            "route_segments": len(route),
            "path_length_m": path_length_m,
            "robot_start_position_m": start_position,
            "robot_end_position_m": end_position,
            "initial_activity_bq": initial_activity,
            "final_activity_bq": final_activity,
            "decontamination_factor": decontamination_factor,
            "activity_reduction_fraction": 1.0 - final_activity / initial_activity,
            "treated_fraction": grid.treated_fraction,
            "contacted_cells": len(contacted),
            "total_cells": len(grid.activity_bq),
            "water_applied_l": applied,
            "water_recovered_l": water_state.recovered_water_l,
            "water_recovery_fraction": recovery_fraction,
            "water_discharged_l": water_state.discharged_water_l,
            "captured_activity_bq": water_state.captured_activity_bq,
            "discharged_activity_bq": water_state.discharged_activity_bq,
            "mass_balance_error_bq": water_process.mass_balance_error_bq,
            "water_balance_error_l": water_process.water_balance_error_l,
            "last_blocked_reason": last_blocked_reason,
            "before_image": str(before_path),
            "during_image": str(during_path) if during_captured else None,
            "after_image": str(after_path),
        }
        print("AROUNDER_VALIDATION_RESULT " + json.dumps(result), flush=True)
        return 0 if passed else 1
    except Exception as error:
        result = {"passed": False, "error": str(error), "traceback": traceback.format_exc()}
        print("AROUNDER_VALIDATION_ERROR " + json.dumps(result), flush=True)
        return 1
    finally:
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if router is not None:
            router.close()
        if world is not None:
            world.stop()
            world.clear()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
