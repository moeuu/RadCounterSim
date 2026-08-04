#!/usr/bin/env python3
"""Run an Arounder-type high-pressure-water decontamination demonstration."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

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
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.time_scale <= 0.0:
        parser.error("--time-scale must be positive")
    return args


ARGS = parse_arguments()

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
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
from radcounter.core.surface_decontamination import DecontaminationTool, SurfaceSourceGrid
from radcounter.core.water_decontamination import (
    WaterDecontaminationState,
    WaterJetSpec,
    WaterSurfaceDecontaminator,
)

ROBOT_ID = "arounder_research_replica"
ROBOT_PATH = "/World/Arounder"
SOURCE_PATH = "/World/ReactorBuilding/ContaminatedFloor"
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
        "steel": define_material(stage, "StainlessSteel", (0.52, 0.56, 0.57), metallic=0.82, roughness=0.24),
        "dark_steel": define_material(stage, "DarkSteel", (0.09, 0.11, 0.12), metallic=0.72),
        "pipe": define_material(stage, "PipeSteel", (0.31, 0.35, 0.36), metallic=0.65),
        "blue": define_material(stage, "WaterLineBlue", (0.03, 0.24, 0.62), metallic=0.18),
        "red": define_material(stage, "ProcessRed", (0.60, 0.055, 0.035), metallic=0.12),
        "green": define_material(stage, "RecoveryGreen", (0.04, 0.38, 0.17), metallic=0.10),
        "glass": define_material(stage, "LensGlass", (0.025, 0.14, 0.19), metallic=0.15, roughness=0.08),
        "brush": define_material(stage, "ContainmentBrush", (0.018, 0.018, 0.014), roughness=0.98),
        "water": define_material(stage, "WaterJet", (0.05, 0.40, 0.95), roughness=0.08, opacity=0.42),
        "wet": define_material(stage, "WetFloor", (0.025, 0.12, 0.20), roughness=0.10, opacity=0.58),
        "white": define_material(stage, "PaintedWhite", (0.72, 0.74, 0.70), roughness=0.54),
        "orange": define_material(stage, "SafetyOrange", (0.94, 0.24, 0.025), roughness=0.45),
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


def add_curve(stage, path: str, points, width_m: float, color) -> object:
    curve = UsdGeom.BasisCurves.Define(stage, path)
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateCurveVertexCountsAttr([len(points)])
    points_attr = curve.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    curve.CreateWidthsAttr([width_m])
    curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return points_attr


def create_environment(stage, materials) -> tuple[tuple[float, float, float], ...]:
    UsdGeom.Xform.Define(stage, "/World/ReactorBuilding")
    add_cube(
        stage,
        "/World/ReactorBuilding/Floor",
        (16.0, 12.0, 0.16),
        (0.0, 0.0, -0.08),
        materials["epoxy"],
        collision=True,
    )
    add_cube(stage, "/World/ReactorBuilding/WallNorth", (16.0, 0.26, 5.0), (0.0, 5.86, 2.5), materials["concrete"], collision=True)
    add_cube(stage, "/World/ReactorBuilding/WallWest", (0.26, 12.0, 5.0), (-7.86, 0.0, 2.5), materials["concrete"], collision=True)
    add_cube(stage, "/World/ReactorBuilding/WallEast", (0.26, 8.0, 5.0), (7.86, 2.0, 2.5), materials["concrete"], collision=True)

    for index, (x_m, y_m) in enumerate(((-5.4, 4.2), (0.0, 4.2), (5.4, 4.2), (-5.4, -4.1), (5.4, -4.1))):
        add_cube(stage, f"/World/ReactorBuilding/Columns/C{index}", (0.52, 0.52, 4.8), (x_m, y_m, 2.4), materials["concrete"], collision=True)
        add_cube(stage, f"/World/ReactorBuilding/Columns/C{index}_Foot", (0.82, 0.82, 0.16), (x_m, y_m, 0.08), materials["concrete"], collision=True)

    add_cylinder(stage, "/World/ReactorBuilding/BiologicalShield", 1.62, 4.3, (4.8, 2.25, 2.15), materials["concrete"], collision=True)
    add_cylinder(stage, "/World/ReactorBuilding/BiologicalShieldRing", 1.82, 0.26, (4.8, 2.25, 0.18), materials["dark_steel"], collision=True)

    pipe_specs = (
        ("Condensate", 0.16, 1.25, materials["blue"]),
        ("FireMain", 0.13, 1.85, materials["red"]),
        ("Vent", 0.22, 2.55, materials["pipe"]),
    )
    for name, radius, z_m, material in pipe_specs:
        add_cylinder(stage, f"/World/ReactorBuilding/PipeRack/{name}", radius, 13.8, (0.0, 5.25, z_m), material, axis="X")
    for index, x_m in enumerate((-6.2, -3.0, 0.2, 3.4, 6.3)):
        add_cube(stage, f"/World/ReactorBuilding/PipeRack/Support{index}", (0.12, 0.55, 3.1), (x_m, 5.22, 1.55), materials["dark_steel"], collision=True)

    for index, (x_m, radius, height) in enumerate(((5.8, 0.48, 1.8), (6.65, 0.36, 1.45))):
        add_cylinder(stage, f"/World/ReactorBuilding/ProcessSkid/Tank{index}", radius, height, (x_m, -3.7, height * 0.5 + 0.15), materials["white"], collision=True)
        add_cylinder(stage, f"/World/ReactorBuilding/ProcessSkid/Tank{index}Cap", radius * 0.82, 0.08, (x_m, -3.7, height + 0.18), materials["dark_steel"])
    add_cube(stage, "/World/ReactorBuilding/ProcessSkid/Base", (2.25, 1.55, 0.18), (5.9, -3.7, 0.09), materials["dark_steel"], collision=True)
    for step in range(6):
        add_cube(stage, f"/World/ReactorBuilding/Stairs/Step{step}", (0.75, 0.32, 0.10), (6.7, -1.65 + step * 0.28, 0.05 + step * 0.11), materials["steel"], collision=True)

    drain_y = -1.42
    add_cube(stage, "/World/ReactorBuilding/Drain/Channel", (4.2, 0.30, 0.025), (0.35, drain_y, 0.012), materials["dark_steel"])
    for index, x_m in enumerate(np.linspace(-1.65, 2.35, 24)):
        add_cube(stage, f"/World/ReactorBuilding/Drain/Bar{index:02d}", (0.035, 0.29, 0.035), (float(x_m), drain_y, 0.028), materials["steel"])

    add_cube(stage, "/World/ReactorBuilding/UtilitySkid/Base", (2.8, 1.75, 0.16), (-5.55, -3.65, 0.08), materials["dark_steel"], collision=True)
    add_cylinder(stage, "/World/ReactorBuilding/UtilitySkid/RecoveryTank", 0.53, 1.55, (-6.15, -3.65, 0.92), materials["white"], collision=True)
    add_cylinder(stage, "/World/ReactorBuilding/UtilitySkid/Pump", 0.25, 0.64, (-4.65, -3.80, 0.42), materials["blue"], axis="X")
    add_cube(stage, "/World/ReactorBuilding/UtilitySkid/ControlCabinet", (0.58, 0.42, 1.05), (-4.75, -3.10, 0.61), materials["yellow_dark"], collision=True)
    add_cylinder(stage, "/World/ReactorBuilding/HoseReel/Drum", 0.43, 0.68, (-5.15, -2.55, 0.78), materials["yellow"], axis="Y")
    add_cylinder(stage, "/World/ReactorBuilding/HoseReel/FlangeL", 0.55, 0.07, (-5.15, -2.93, 0.78), materials["yellow_dark"], axis="Y")
    add_cylinder(stage, "/World/ReactorBuilding/HoseReel/FlangeR", 0.55, 0.07, (-5.15, -2.17, 0.78), materials["yellow_dark"], axis="Y")
    add_cube(stage, "/World/ReactorBuilding/HoseReel/StandL", (0.10, 0.10, 1.25), (-5.58, -2.85, 0.63), materials["yellow_dark"])
    add_cube(stage, "/World/ReactorBuilding/HoseReel/StandR", (0.10, 0.10, 1.25), (-4.72, -2.85, 0.63), materials["yellow_dark"])

    roller_points = ((-3.75, -2.25, 0.20), (-2.55, -1.85, 0.20))
    for index, point in enumerate(roller_points):
        add_cylinder(stage, f"/World/ReactorBuilding/CornerRollers/R{index}", 0.12, 0.32, point, materials["orange"], axis="Z")
        add_cube(stage, f"/World/ReactorBuilding/CornerRollers/R{index}Base", (0.38, 0.38, 0.04), (point[0], point[1], 0.02), materials["dark_steel"])

    for index in range(5):
        x_m = -6.6 + index * 0.55
        add_cube(stage, f"/World/ReactorBuilding/Barrier/Post{index}", (0.06, 0.06, 0.85), (x_m, -2.65, 0.43), materials["orange"])
    add_cube(stage, "/World/ReactorBuilding/Barrier/Rail", (2.25, 0.05, 0.06), (-5.5, -2.65, 0.72), materials["orange"])

    for index, (x_m, y_m, yaw) in enumerate(((-6.5, 1.0, 17.0), (3.4, -4.3, -8.0), (6.7, 0.2, 25.0))):
        add_cube(stage, f"/World/ReactorBuilding/Debris/Chunk{index}", (0.34, 0.22, 0.14), (x_m, y_m, 0.08), materials["concrete"], rotation_xyz=(0.0, 0.0, yaw), collision=True)

    for index, x_m in enumerate((-4.8, 0.0, 4.8)):
        light = UsdLux.RectLight.Define(stage, f"/World/ReactorBuilding/Lights/L{index}")
        light.CreateIntensityAttr(18_000.0)
        light.CreateWidthAttr(2.2)
        light.CreateHeightAttr(0.25)
        light.CreateColorAttr(Gf.Vec3f(0.78, 0.88, 1.0))
        light.AddTranslateOp().Set(Gf.Vec3d(x_m, 0.0, 4.65))
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
    prim.CreateAttribute("rad:robot:reference", Sdf.ValueTypeNames.String).Set("Hitachi-GE Arounder")
    prim.CreateAttribute("rad:stream:focus", Sdf.ValueTypeNames.Bool).Set(True)

    add_cube(stage, f"{ROBOT_PATH}/LowerChassis", (1.28, 0.52, 0.30), (-0.05, 0.0, -0.10), materials["yellow_dark"], collision=True)
    add_cube(stage, f"{ROBOT_PATH}/MainBody", (1.12, 0.54, 0.42), (-0.13, 0.0, 0.16), materials["yellow"], collision=True)
    add_cube(stage, f"{ROBOT_PATH}/RearPowerPack", (0.46, 0.50, 0.55), (-0.42, 0.0, 0.52), materials["yellow"])
    add_cube(stage, f"{ROBOT_PATH}/TopServicePanel", (0.62, 0.45, 0.06), (-0.12, 0.0, 0.48), materials["dark_steel"])

    for side_name, y_m in (("Left", 0.36), ("Right", -0.36)):
        add_cube(stage, f"{ROBOT_PATH}/Tracks/{side_name}/Belt", (1.50, 0.18, 0.34), (-0.08, y_m, -0.18), materials["track"], collision=True)
        for wheel_index, x_m in enumerate((-0.57, -0.18, 0.22, 0.57)):
            add_cylinder(stage, f"{ROBOT_PATH}/Tracks/{side_name}/RoadWheel{wheel_index}", 0.135, 0.19, (x_m, y_m, -0.18), materials["steel"], axis="Y")
            add_cylinder(stage, f"{ROBOT_PATH}/Tracks/{side_name}/Hub{wheel_index}", 0.055, 0.205, (x_m, y_m, -0.18), materials["yellow_dark"], axis="Y")
        for tread_index, x_m in enumerate(np.linspace(-0.70, 0.54, 10)):
            for surface, z_m in (("Top", -0.005), ("Bottom", -0.355)):
                add_cube(stage, f"{ROBOT_PATH}/Tracks/{side_name}/{surface}Tread{tread_index:02d}", (0.105, 0.205, 0.030), (float(x_m), y_m, z_m), materials["track"])

    shoulder = (-0.25, 0.46)
    elbow = (0.16, 0.73)
    wrist = (0.52, 0.39)
    tool_mount = (0.79, -0.10)
    add_cylinder(stage, f"{ROBOT_PATH}/Arm/BaseYaw", 0.19, 0.16, (-0.25, 0.0, 0.43), materials["yellow_dark"], axis="Z")
    add_link(stage, f"{ROBOT_PATH}/Arm/Boom", shoulder, elbow, 0.0, 0.13, materials["yellow"])
    add_link(stage, f"{ROBOT_PATH}/Arm/Forearm", elbow, wrist, 0.0, 0.12, materials["steel"])
    add_link(stage, f"{ROBOT_PATH}/Arm/WristLink", wrist, tool_mount, 0.0, 0.105, materials["steel"])
    add_link(stage, f"{ROBOT_PATH}/Arm/HydraulicCylinderA", (-0.30, 0.34), (0.09, 0.68), 0.10, 0.052, materials["steel"])
    add_link(stage, f"{ROBOT_PATH}/Arm/HydraulicCylinderB", (0.05, 0.65), (0.47, 0.33), -0.10, 0.045, materials["yellow_dark"])
    for index, point in enumerate((shoulder, elbow, wrist, tool_mount)):
        add_cylinder(stage, f"{ROBOT_PATH}/Arm/Joint{index}", 0.105, 0.22, (point[0], 0.0, point[1]), materials["dark_steel"], axis="Y")

    head = UsdGeom.Xform.Define(stage, f"{ROBOT_PATH}/Arm/DeconHead")
    head.AddTranslateOp().Set(Gf.Vec3d(HEAD_X_M, 0.0, -0.26))
    add_cube(stage, f"{ROBOT_PATH}/Arm/DeconHead/Top", (0.42, 0.58, 0.075), (0.0, 0.0, 0.09), materials["steel"])
    add_cube(stage, f"{ROBOT_PATH}/Arm/DeconHead/FrontFrame", (0.075, 0.58, 0.18), (0.17, 0.0, 0.015), materials["steel"])
    add_cube(stage, f"{ROBOT_PATH}/Arm/DeconHead/RearFrame", (0.075, 0.58, 0.18), (-0.17, 0.0, 0.015), materials["steel"])
    add_cube(stage, f"{ROBOT_PATH}/Arm/DeconHead/BrushFront", (0.055, 0.58, 0.10), (0.19, 0.0, -0.075), materials["brush"])
    add_cube(stage, f"{ROBOT_PATH}/Arm/DeconHead/BrushRear", (0.055, 0.58, 0.10), (-0.19, 0.0, -0.075), materials["brush"])
    add_cube(stage, f"{ROBOT_PATH}/Arm/DeconHead/BrushLeft", (0.34, 0.045, 0.10), (0.0, 0.29, -0.075), materials["brush"])
    add_cube(stage, f"{ROBOT_PATH}/Arm/DeconHead/BrushRight", (0.34, 0.045, 0.10), (0.0, -0.29, -0.075), materials["brush"])
    add_cylinder(stage, f"{ROBOT_PATH}/Arm/DeconHead/SuctionPort", 0.075, 0.13, (-0.10, 0.0, 0.17), materials["green"], axis="Z")

    carriage = UsdGeom.Xform.Define(stage, f"{ROBOT_PATH}/Arm/DeconHead/NozzleCarriage")
    nozzle_translate_op = carriage.AddTranslateOp()
    nozzle_translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    add_cube(stage, f"{ROBOT_PATH}/Arm/DeconHead/NozzleCarriage/RailCar", (0.12, 0.10, 0.055), (0.0, 0.0, 0.035), materials["yellow_dark"])
    add_cylinder(stage, f"{ROBOT_PATH}/Arm/DeconHead/NozzleCarriage/Nozzle", 0.028, 0.105, (0.04, 0.0, -0.02), materials["steel"], axis="Z")
    spray_prims = []
    for index, offset_y in enumerate((-0.025, 0.0, 0.025)):
        cone = UsdGeom.Cone.Define(stage, f"{ROBOT_PATH}/Arm/DeconHead/NozzleCarriage/ContainedJet{index}")
        cone.CreateAxisAttr("Z")
        cone.CreateRadiusAttr(0.035 + 0.008 * index)
        cone.CreateHeightAttr(0.11)
        cone.AddTranslateOp().Set(Gf.Vec3d(0.04, offset_y, -0.095))
        bind(cone, materials["water"])
        cone.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        spray_prims.append(cone)

    for camera_index, y_m in enumerate((-0.18, 0.18)):
        add_cube(stage, f"{ROBOT_PATH}/Vision/Camera{camera_index}Body", (0.14, 0.12, 0.11), (0.52, y_m, 0.54), materials["dark_steel"])
        add_cylinder(stage, f"{ROBOT_PATH}/Vision/Camera{camera_index}Lens", 0.043, 0.035, (0.60, y_m, 0.54), materials["glass"], axis="X")
        add_sphere(stage, f"{ROBOT_PATH}/Vision/WorkLight{camera_index}", 0.065, (0.58, y_m * 0.45, 0.38), materials["white"])
    seal_lamp = add_sphere(stage, f"{ROBOT_PATH}/Vision/SealLamp", 0.045, (0.30, 0.0, 0.66), materials["red"])
    seal_color_attr = seal_lamp.CreateDisplayColorAttr([Gf.Vec3f(0.88, 0.04, 0.02)])

    add_curve(stage, f"{ROBOT_PATH}/Hoses/HighPressure", [(-0.65, -0.16, 0.62), (-0.12, -0.16, 0.78), (0.42, -0.15, 0.44), (0.88, -0.12, -0.08)], 0.036, (0.06, 0.22, 0.72))
    add_curve(stage, f"{ROBOT_PATH}/Hoses/Recovery", [(-0.65, 0.16, 0.58), (-0.12, 0.18, 0.74), (0.45, 0.18, 0.40), (0.86, 0.13, -0.04)], 0.058, (0.05, 0.08, 0.07))
    add_curve(stage, f"{ROBOT_PATH}/Hoses/Control", [(-0.65, 0.05, 0.66), (-0.05, 0.05, 0.82), (0.52, 0.06, 0.38)], 0.018, (0.95, 0.34, 0.02))

    UsdGeom.Xform.Define(stage, "/World/Utilities")
    tether_points_attrs = (
        add_curve(stage, "/World/Utilities/Tether/Pressure", [(0.0, 0.0, 0.0)] * 5, 0.040, (0.04, 0.16, 0.60)),
        add_curve(stage, "/World/Utilities/Tether/Recovery", [(0.0, 0.0, 0.0)] * 5, 0.064, (0.025, 0.035, 0.03)),
        add_curve(stage, "/World/Utilities/Tether/PowerControl", [(0.0, 0.0, 0.0)] * 5, 0.020, (0.94, 0.28, 0.02)),
    )
    return RobotVisuals(nozzle_translate_op, tuple(spray_prims), seal_color_attr, tether_points_attrs)


def activity_field(cells_x: int, cells_y: int) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, cells_x)
    y = np.linspace(-1.0, 1.0, cells_y)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    hotspot = 2.8 * np.exp(-((xx - 0.30) ** 2 + (yy + 0.18) ** 2) / 0.12)
    pipe_drip = 1.4 * np.exp(-((xx + 0.48) ** 2) / 0.045) * np.exp(-((yy - 0.22) ** 2) / 0.25)
    mottling = 0.22 * (np.sin(7.0 * xx) * np.cos(5.0 * yy) + 1.0)
    return (115_000.0 * (1.0 + hotspot + pipe_drip + mottling)).reshape(-1)


def create_surface_source(stage) -> tuple[SurfaceSourceGrid, list, list]:
    cells_x, cells_y = 32, 16
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
    root_prim.CreateAttribute("rad:surface:coating", Sdf.ValueTypeNames.String).Set("radiation-resistant epoxy")
    root_prim.CreateAttribute("rad:source:initialActivityBq", Sdf.ValueTypeNames.Double).Set(grid.initial_total_activity_bq)
    color_attributes = []
    activity_attributes = []
    raw_colors = grid.color_rgb()
    epoxy = np.asarray([0.19, 0.27, 0.25])
    colors = 0.62 * raw_colors + 0.38 * epoxy
    for index, center in enumerate(grid.centers_world_m):
        cell = UsdGeom.Cube.Define(stage, f"{SOURCE_PATH}/Cell_{index:03d}")
        cell.CreateSizeAttr(1.0)
        cell.AddTranslateOp().Set(Gf.Vec3d(*center))
        cell.AddScaleOp().Set(Gf.Vec3f(grid.cell_size_x_m * 0.985, grid.cell_size_y_m * 0.985, 0.012))
        color_attribute = cell.CreateDisplayColorAttr([Gf.Vec3f(*colors[index])])
        color_attributes.append(color_attribute)
        prim = cell.GetPrim()
        prim.CreateAttribute("rad:role", Sdf.ValueTypeNames.String).Set("source")
        prim.CreateAttribute("rad:source:type", Sdf.ValueTypeNames.String).Set("surface_cell")
        activity_attr = prim.CreateAttribute("rad:source:activityBq", Sdf.ValueTypeNames.Double)
        activity_attr.Set(float(grid.activity_bq[index]))
        activity_attributes.append(activity_attr)
        prim.CreateAttribute("rad:decon:enabled", Sdf.ValueTypeNames.Bool).Set(True)
    return grid, color_attributes, activity_attributes


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
        add_cube(stage, f"/World/ReactorBuilding/WorkZone/{name}", size, position, materials["orange"])
    for index, x_m in enumerate(np.linspace(cx - half_x, cx + half_x, 8)):
        add_cube(stage, f"/World/ReactorBuilding/WorkZone/SurveyMark{index}", (0.025, 0.16, 0.02), (float(x_m), cy + half_y + 0.18, 0.022), materials["white"])


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
            segments.append(RouteSegment((forward_end_x, float(row)), True, f"sealed wash lane {row_index + 1}"))
            if row_index + 1 < len(rows):
                segments.append(RouteSegment((reverse_start_x, float(rows[row_index + 1])), False, "head closed / reposition"))
        else:
            segments.append(RouteSegment((reverse_end_x, float(row)), True, f"sealed wash lane {row_index + 1}"))
            if row_index + 1 < len(rows):
                segments.append(RouteSegment((forward_start_x, float(rows[row_index + 1])), False, "head closed / reposition"))
    return start, segments


def transform_and_pose(stage) -> tuple[Gf.Matrix4d, tuple[float, float, float], float]:
    transform = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT_PATH)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
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
        self.seal.text = "Head seal: CLOSED | jet + vacuum ON" if sealed else "Head seal: OPEN | jet interlocked"
        self.process.text = f"Process: {pressure_mpa:.0f} MPa | 450 mm internal nozzle scan"
        self.activity.text = f"Surface activity: {grid.total_activity_bq:,.0f} Bq"
        df = grid.initial_total_activity_bq / max(grid.total_activity_bq, 1e-12)
        self.df.text = f"Decontamination factor: {df:.2f}"
        self.treated.text = f"Treated area: {grid.treated_fraction * 100.0:.1f}%"
        self.route.text = f"Route segment: {min(auto.index + 1, len(auto.route))} / {len(auto.route)}"
        self.water.text = f"Water: {state.applied_water_l:.1f} L supplied / {state.recovered_water_l:.1f} L recovered"
        recovery = state.recovered_water_l / max(state.applied_water_l, 1e-12) * 100.0
        self.recovery.text = f"Recovery: {recovery:.1f}% | discharge: {state.discharged_water_l:.1f} L"
        self.waste.text = f"Captured activity: {state.captured_activity_bq:,.0f} Bq"
        hours = int(physical_elapsed_s // 3600)
        minutes = int((physical_elapsed_s % 3600) // 60)
        seconds = int(physical_elapsed_s % 60)
        self.clock.text = f"Plant operation time: {hours:02d}:{minutes:02d}:{seconds:02d} ({ARGS.time_scale:.0f}x view)"
        self.progress_model.set_value(float(min(grid.removed_fraction, 1.0)))


def operation_parameters(mode: str) -> tuple[float, float]:
    return {
        "surface-wash": (50.0, 0.035),
        "coating-strip": (120.0, 0.060),
        "scarify": (200.0, 0.085),
    }[mode]


def create_water_spec(mode: str) -> WaterJetSpec:
    pressure_mpa, coefficient = operation_parameters(mode)
    return WaterJetSpec(
        flow_rate_l_min=6.0,
        pressure_mpa=pressure_mpa,
        reference_pressure_mpa=50.0,
        spray_cone_angle_deg=14.0,
        minimum_footprint_radius_m=0.080,
        min_standoff_m=0.10,
        max_standoff_m=0.26,
        max_incidence_angle_deg=12.0,
        removal_coefficient_m2_per_l=coefficient,
        max_surface_speed_m_s=0.006,
        activity_capture_fraction=0.98,
        water_recovery_fraction=0.97,
        runoff_redeposition_fraction=0.10,
        surface_water_retention_fraction=0.01,
        require_wastewater_collection=True,
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
    visuals.seal_color_attr.Set([Gf.Vec3f(0.04, 0.86, 0.18) if sealed else Gf.Vec3f(0.88, 0.04, 0.02)])


def update_surface_visuals(grid, water_process, color_attributes, activity_attributes) -> None:
    colors = water_process.visual_color_rgb()
    epoxy = np.asarray([0.19, 0.27, 0.25])
    colors = 0.68 * colors + 0.32 * epoxy
    for index, (color_attr, activity_attr) in enumerate(zip(color_attributes, activity_attributes, strict=True)):
        color_attr.Set([Gf.Vec3f(*colors[index])])
        activity_attr.Set(float(grid.activity_bq[index]))


def request_capture(path: Path) -> None:
    path.unlink(missing_ok=True)
    viewport = viewport_utility.get_active_viewport()
    if viewport is None:
        raise RuntimeError("active viewport is unavailable")
    viewport_utility.capture_viewport_to_file(viewport, str(path))


def main() -> int:
    ARGS.output.mkdir(parents=True, exist_ok=True)
    result_path = ARGS.output / "result.json"
    before_path = ARGS.output / "before.png"
    during_path = ARGS.output / "during.png"
    after_path = ARGS.output / "after.png"
    result: dict[str, object] = {"passed": False}
    world = None
    router = None
    try:
        world = World(stage_units_in_meters=1.0, physics_dt=DT_S, rendering_dt=DT_S)
        stage = omni.usd.get_context().get_stage()
        materials = create_materials(stage)
        roller_points = create_environment(stage, materials)
        grid, color_attributes, activity_attributes = create_surface_source(stage)
        create_work_zone(stage, materials, grid)
        start_xyz, route = build_route(grid)
        visuals = create_robot(stage, materials, start_xyz)

        dome = UsdLux.DomeLight.Define(stage, "/World/ReactorBuilding/Ambient")
        dome.CreateIntensityAttr(340.0)
        dome.CreateColorAttr(Gf.Vec3f(0.56, 0.66, 0.76))

        world.reset()
        controller = RigidArounderController(stage)
        auto = ArounderAutoController(stage, route, ARGS.time_scale)
        router = IsaacRobotInputRouter({ROBOT_ID: controller})
        router.attach_devices(ROBOT_ID, ROBOT_ID)
        router.register_auto_controller(ROBOT_ID, auto)

        pressure_mpa, _coefficient = operation_parameters(ARGS.operation_mode)
        panel = ArounderWindow(grid.initial_total_activity_bq, pressure_mpa)
        water_state = WaterDecontaminationState(supply_remaining_l=1000.0, wastewater_capacity_l=1000.0)
        water_process = WaterSurfaceDecontaminator(
            grid,
            water_state,
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
            scan_visual_speed_m_s = abs(0.5 * HEAD_STROKE_M * 2.0 * math.pi * 0.52 * math.cos(scan_phase))
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
                    f"activity_bq={grid.total_activity_bq:.0f} df={initial_activity / max(grid.total_activity_bq, 1e-12):.2f} "
                    f"treated_pct={grid.treated_fraction * 100.0:.1f} path_m={path_length_m:.2f} "
                    f"water_l={water_state.applied_water_l:.1f} recovery_pct={recovery * 100.0:.1f}",
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
            "reference_robot": "Hitachi-GE / IRID Arounder low-section high-pressure-water decontamination machine",
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
