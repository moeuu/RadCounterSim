"""Low-cost Isaac visuals derived from traceable real nuclear-response robots.

These are original procedural reference models, not manufacturer CAD.  Their
dimensions, major mechanisms, payload locations and operating roles follow the
official sources recorded in :mod:`radcounter.core.robots.reference`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from radcounter.core.robots import get_real_robot_reference


@dataclass(frozen=True, slots=True)
class SpawnedReferenceRobot:
    model_id: str
    prim_path: str
    base_link_path: str
    translation_op: Any
    yaw_op: Any
    sensor_links: dict[str, str]


def _color(prim: Any, rgb: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])


def _cube(stage: Any, path: str, size: tuple[float, float, float], xyz=(0.0, 0.0, 0.0), rgb=(0.3, 0.3, 0.3), rpy=(0.0, 0.0, 0.0)) -> Any:
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    if any(rpy):
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rpy))
    xform.AddScaleOp().Set(Gf.Vec3f(*size))
    _color(cube.GetPrim(), rgb)
    return cube.GetPrim()


def _cylinder(stage: Any, path: str, radius: float, height: float, xyz=(0.0, 0.0, 0.0), rgb=(0.2, 0.2, 0.2), axis="Z", rpy=(0.0, 0.0, 0.0)) -> Any:
    from pxr import Gf, UsdGeom

    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateRadiusAttr(radius)
    cylinder.CreateHeightAttr(height)
    cylinder.CreateAxisAttr(axis)
    xform = UsdGeom.Xformable(cylinder)
    xform.AddTranslateOp().Set(Gf.Vec3d(*xyz))
    if any(rpy):
        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rpy))
    _color(cylinder.GetPrim(), rgb)
    return cylinder.GetPrim()


def _sphere(stage: Any, path: str, radius: float, xyz=(0.0, 0.0, 0.0), rgb=(0.2, 0.2, 0.2)) -> Any:
    from pxr import Gf, UsdGeom

    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius)
    UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(*xyz))
    _color(sphere.GetPrim(), rgb)
    return sphere.GetPrim()


def _camera(stage: Any, path: str, xyz: tuple[float, float, float], scale=1.0) -> None:
    _cube(stage, path + "/Housing", (0.08 * scale, 0.10 * scale, 0.07 * scale), xyz, (0.08, 0.09, 0.10))
    _cylinder(stage, path + "/Lens", 0.027 * scale, 0.018 * scale, (xyz[0] + 0.047 * scale, xyz[1], xyz[2]), (0.03, 0.10, 0.16), axis="X")
    _cylinder(stage, path + "/LampL", 0.017 * scale, 0.012 * scale, (xyz[0] + 0.047 * scale, xyz[1] + 0.037 * scale, xyz[2]), (0.95, 0.90, 0.65), axis="X")
    _cylinder(stage, path + "/LampR", 0.017 * scale, 0.012 * scale, (xyz[0] + 0.047 * scale, xyz[1] - 0.037 * scale, xyz[2]), (0.95, 0.90, 0.65), axis="X")


def _track(stage: Any, path: str, xyz: tuple[float, float, float], length: float, width: float, height: float, wheel_count: int = 4) -> None:
    _cube(stage, path + "/Belt", (length, width, height), xyz, (0.035, 0.040, 0.042))
    for index in range(wheel_count):
        x = xyz[0] - 0.36 * length + index * (0.72 * length / max(1, wheel_count - 1))
        _cylinder(stage, f"{path}/RoadWheel{index}", height * 0.32, width * 1.04, (x, xyz[1], xyz[2]), (0.20, 0.22, 0.20), axis="Y")


def _root(stage: Any, prim_path: str, model_id: str) -> tuple[Any, Any, str]:
    from pxr import Gf, Sdf, UsdGeom

    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)
    reference = get_real_robot_reference(model_id)
    root = UsdGeom.Xform.Define(stage, prim_path)
    xform = UsdGeom.Xformable(root)
    translation = xform.AddTranslateOp()
    translation.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    yaw = xform.AddRotateZOp()
    yaw.Set(0.0)
    prim = root.GetPrim()
    attributes = (
        ("rad:robot:referenceModelId", Sdf.ValueTypeNames.String, reference.id),
        ("rad:robot:manufacturer", Sdf.ValueTypeNames.String, reference.manufacturer),
        ("rad:robot:model", Sdf.ValueTypeNames.String, reference.model),
        ("rad:robot:geometryFidelity", Sdf.ValueTypeNames.String, "reference_procedural"),
        ("rad:robot:notManufacturerCAD", Sdf.ValueTypeNames.Bool, True),
        ("rad:robot:sourceUrl", Sdf.ValueTypeNames.String, reference.source_urls[0]),
        ("rad:robot:massKg", Sdf.ValueTypeNames.Double, reference.mass_kg or 0.0),
        ("rad:robot:dimensionsM", Sdf.ValueTypeNames.Double3, Gf.Vec3d(*reference.dimensions_m)),
    )
    for name, value_type, value in attributes:
        prim.CreateAttribute(name, value_type, custom=True).Set(value)
    base_path = prim_path + "/base_link"
    UsdGeom.Xform.Define(stage, base_path)
    return translation, yaw, base_path


def _packbot(stage: Any, prim_path: str) -> SpawnedReferenceRobot:
    model_id = "irobot-packbot-fukushima"
    translate, yaw, base = _root(stage, prim_path, model_id)
    black = (0.045, 0.050, 0.050)
    deck = (0.47, 0.46, 0.39)
    _track(stage, base + "/LeftTrack", (0.0, 0.205, 0.095), 0.52, 0.12, 0.17)
    _track(stage, base + "/RightTrack", (0.0, -0.205, 0.095), 0.52, 0.12, 0.17)
    _cube(stage, base + "/Chassis", (0.43, 0.30, 0.12), (-0.01, 0.0, 0.16), deck)
    _cube(stage, base + "/ElectronicsDeck", (0.28, 0.25, 0.09), (-0.07, 0.0, 0.255), (0.24, 0.25, 0.23))
    for side, y in (("L", 0.205), ("R", -0.205)):
        _cube(stage, base + f"/FrontFlipper{side}", (0.23, 0.09, 0.075), (0.30, y, 0.115), black, (0.0, -14.0, 0.0))
        _cylinder(stage, base + f"/FlipperHub{side}", 0.060, 0.125, (0.22, y, 0.11), deck, axis="Y")
    _cylinder(stage, base + "/MastPan", 0.045, 0.10, (0.02, 0.0, 0.35), black)
    _cube(stage, base + "/Mast", (0.045, 0.045, 0.25), (0.02, 0.0, 0.47), deck)
    _cube(stage, base + "/ManipulatorBoom", (0.28, 0.055, 0.055), (0.145, 0.0, 0.52), deck, (0.0, -22.0, 0.0))
    camera_link = base + "/camera_link"
    from pxr import UsdGeom
    UsdGeom.Xform.Define(stage, camera_link)
    _camera(stage, camera_link, (0.30, 0.0, 0.61), 0.9)
    lidar_link = base + "/lidar_link"
    UsdGeom.Xform.Define(stage, lidar_link)
    _cylinder(stage, lidar_link + "/Lidar", 0.045, 0.075, (0.02, 0.0, 0.68), (0.08, 0.12, 0.14))
    radiation_link = base + "/radiation_link"
    UsdGeom.Xform.Define(stage, radiation_link)
    _cube(stage, radiation_link + "/GammaProbe", (0.16, 0.075, 0.10), (0.03, 0.16, 0.41), (0.93, 0.64, 0.08))
    _cylinder(stage, base + "/Antenna", 0.007, 0.36, (-0.17, -0.09, 0.48), black)
    return SpawnedReferenceRobot(model_id, prim_path, base, translate, yaw, {"camera": camera_link, "lidar": lidar_link, "radiation": radiation_link})


def _cage_curves(stage: Any, path: str, radius: float) -> None:
    from pxr import Gf, UsdGeom, Vt

    points = []
    counts = []
    segments = 32
    for plane in ("xy", "xz", "yz"):
        ring = []
        for index in range(segments + 1):
            angle = 2.0 * math.pi * index / segments
            a, b = radius * math.cos(angle), radius * math.sin(angle)
            ring.append({"xy": (a, b, 0.0), "xz": (a, 0.0, b), "yz": (0.0, a, b)}[plane])
        points.extend(Gf.Vec3f(*value) for value in ring)
        counts.append(len(ring))
    curves = UsdGeom.BasisCurves.Define(stage, path)
    curves.CreateTypeAttr("linear")
    curves.CreateWrapAttr("nonperiodic")
    curves.CreateCurveVertexCountsAttr(Vt.IntArray(counts))
    curves.CreatePointsAttr(Vt.Vec3fArray(points))
    curves.CreateWidthsAttr(Vt.FloatArray([0.009] * len(counts)))
    curves.SetWidthsInterpolation("constant")
    _color(curves.GetPrim(), (0.82, 0.84, 0.83))


def _elios3(stage: Any, prim_path: str) -> SpawnedReferenceRobot:
    model_id = "flyability-elios3-rad"
    translate, yaw, base = _root(stage, prim_path, model_id)
    from pxr import UsdGeom

    _cage_curves(stage, base + "/CollisionTolerantCage", 0.235)
    _cube(stage, base + "/AvionicsBody", (0.25, 0.11, 0.085), (0.0, 0.0, 0.0), (0.10, 0.11, 0.12))
    _cube(stage, base + "/Battery", (0.14, 0.12, 0.10), (-0.10, 0.0, 0.035), (0.20, 0.21, 0.22))
    for index, (x, y) in enumerate(((0.105, 0.105), (0.105, -0.105), (-0.105, 0.105), (-0.105, -0.105))):
        _cube(stage, base + f"/RotorArm{index}", (0.18, 0.025, 0.025), (0.5 * x, 0.5 * y, 0.0), (0.16, 0.17, 0.18), (0.0, 0.0, math.degrees(math.atan2(y, x))))
        _cylinder(stage, base + f"/Rotor{index}", 0.078, 0.012, (x, y, 0.0), (0.04, 0.045, 0.05))
        _cylinder(stage, base + f"/Motor{index}", 0.025, 0.035, (x, y, 0.0), (0.48, 0.50, 0.49))
    camera_link = base + "/camera_link"
    UsdGeom.Xform.Define(stage, camera_link)
    _camera(stage, camera_link, (0.155, 0.0, -0.015), 0.72)
    lidar_link = base + "/lidar_link"
    UsdGeom.Xform.Define(stage, lidar_link)
    _cylinder(stage, lidar_link + "/OusterOS0", 0.043, 0.065, (-0.015, 0.0, -0.095), (0.10, 0.12, 0.13))
    radiation_link = base + "/radiation_link"
    UsdGeom.Xform.Define(stage, radiation_link)
    _cube(stage, radiation_link + "/MirionRDS32", (0.14, 0.062, 0.052), (-0.105, 0.0, -0.115), (0.95, 0.82, 0.10))
    _cube(stage, radiation_link + "/RDS32Display", (0.045, 0.064, 0.030), (-0.095, 0.0, -0.148), (0.08, 0.24, 0.20))
    return SpawnedReferenceRobot(model_id, prim_path, base, translate, yaw, {"camera": camera_link, "lidar": lidar_link, "radiation": radiation_link})


def _arm_visual(stage: Any, base: str, name: str, side: float, abrasive: bool) -> str:
    yellow = (0.88, 0.62, 0.06)
    joints = ((0.02, side * 0.22, 0.74), (0.12, side * 0.28, 0.92), (0.32, side * 0.30, 1.04), (0.49, side * 0.27, 0.91))
    for index, point in enumerate(joints):
        _sphere(stage, f"{base}/{name}/Joint{index + 1}", 0.055 if index < 2 else 0.043, point, (0.16, 0.17, 0.16))
    for index, (start, end) in enumerate(zip(joints, joints[1:], strict=True)):
        midpoint = tuple(0.5 * (a + b) for a, b in zip(start, end, strict=True))
        length = math.dist(start, end)
        yaw = math.degrees(math.atan2(end[1] - start[1], end[0] - start[0]))
        pitch = -math.degrees(math.atan2(end[2] - start[2], math.hypot(end[0] - start[0], end[1] - start[1])))
        _cube(stage, f"{base}/{name}/Link{index + 1}", (length, 0.065, 0.065), midpoint, yellow, (0.0, pitch, yaw))
    tool_path = base + ("/left_tool0" if side > 0 else "/right_tool0")
    from pxr import UsdGeom
    UsdGeom.Xform.Define(stage, tool_path)
    end = joints[-1]
    if abrasive:
        _cylinder(stage, tool_path + "/BlastNozzle", 0.032, 0.22, (end[0] + 0.10, end[1], end[2] - 0.02), (0.20, 0.22, 0.23), axis="X")
        _cylinder(stage, tool_path + "/RecoveryShroud", 0.075, 0.05, (end[0] + 0.21, end[1], end[2] - 0.02), (0.08, 0.09, 0.09), axis="X")
    else:
        _cube(stage, tool_path + "/GripperPalm", (0.11, 0.10, 0.055), (end[0] + 0.07, end[1], end[2]), (0.18, 0.19, 0.19))
        _cube(stage, tool_path + "/FingerA", (0.12, 0.025, 0.025), (end[0] + 0.15, end[1] + 0.045, end[2]), (0.10, 0.11, 0.11))
        _cube(stage, tool_path + "/FingerB", (0.12, 0.025, 0.025), (end[0] + 0.15, end[1] - 0.045, end[2]), (0.10, 0.11, 0.11))
    return tool_path


def _meister(stage: Any, prim_path: str) -> SpawnedReferenceRobot:
    model_id = "mhi-meister"
    translate, yaw, base = _root(stage, prim_path, model_id)
    yellow = (0.88, 0.62, 0.06)
    for longitudinal, x in (("Front", 0.34), ("Rear", -0.34)):
        for side, y in (("Left", 0.285), ("Right", -0.285)):
            _track(stage, f"{base}/{longitudinal}{side}Crawler", (x, y, 0.15), 0.48, 0.13, 0.22, 3)
            _cylinder(stage, f"{base}/{longitudinal}{side}Pivot", 0.07, 0.15, (x, y, 0.23), yellow, axis="Y")
    _cube(stage, base + "/LowerChassis", (0.72, 0.46, 0.20), (0.0, 0.0, 0.28), yellow)
    _cube(stage, base + "/TiltingUpperBody", (0.50, 0.45, 0.38), (-0.05, 0.0, 0.53), yellow)
    _cube(stage, base + "/ElectronicsCabinet", (0.30, 0.34, 0.36), (-0.24, 0.0, 0.77), yellow)
    left_tool = _arm_visual(stage, base, "LeftArm7Axis", 1.0, abrasive=False)
    right_tool = _arm_visual(stage, base, "RightArm7Axis", -1.0, abrasive=True)
    from pxr import UsdGeom
    camera_link = base + "/head_camera_link"
    UsdGeom.Xform.Define(stage, camera_link)
    _camera(stage, camera_link, (0.12, 0.0, 1.16), 1.0)
    _cylinder(stage, base + "/CableReel", 0.15, 0.10, (-0.47, 0.0, 0.58), (0.08, 0.09, 0.09), axis="Y")
    return SpawnedReferenceRobot(model_id, prim_path, base, translate, yaw, {"camera": camera_link, "left_tool": left_tool, "right_tool": right_tool})


def _arounder(stage: Any, prim_path: str) -> SpawnedReferenceRobot:
    model_id = "hitachi-ge-arounder"
    translate, yaw, base = _root(stage, prim_path, model_id)
    yellow = (0.90, 0.63, 0.04)
    _track(stage, base + "/LeftCrawler", (0.0, 0.235, 0.18), 1.05, 0.13, 0.25, 5)
    _track(stage, base + "/RightCrawler", (0.0, -0.235, 0.18), 1.05, 0.13, 0.25, 5)
    _cube(stage, base + "/PowerPack", (0.56, 0.43, 0.55), (-0.20, 0.0, 0.51), yellow)
    for index, point in enumerate(((0.05, 0.0, 0.72), (0.27, 0.0, 0.87), (0.49, 0.0, 0.61), (0.63, 0.0, 0.32))):
        _sphere(stage, base + f"/TreatmentArm/Joint{index + 1}", 0.055, point, (0.13, 0.14, 0.14))
    _cube(stage, base + "/TreatmentArm/Upper", (0.34, 0.07, 0.07), (0.17, 0.0, 0.80), yellow, (0.0, -33.0, 0.0))
    _cube(stage, base + "/TreatmentArm/Fore", (0.36, 0.07, 0.07), (0.38, 0.0, 0.74), yellow, (0.0, 43.0, 0.0))
    head = base + "/treatment_head"
    from pxr import UsdGeom
    UsdGeom.Xform.Define(stage, head)
    _cube(stage, head + "/RecoveryHead", (0.34, 0.52, 0.13), (0.66, 0.0, 0.10), (0.22, 0.24, 0.24))
    _cube(stage, head + "/BrushSkirt", (0.35, 0.53, 0.035), (0.66, 0.0, 0.025), (0.025, 0.030, 0.030))
    _cylinder(stage, head + "/ReciprocatingNozzle", 0.015, 0.42, (0.66, 0.0, 0.13), (0.12, 0.42, 0.68), axis="Y")
    _cylinder(stage, head + "/SuctionPort", 0.04, 0.08, (0.59, -0.18, 0.17), (0.08, 0.09, 0.09), axis="X")
    camera_link = base + "/camera_link"
    UsdGeom.Xform.Define(stage, camera_link)
    _camera(stage, camera_link, (0.24, 0.0, 0.94), 0.9)
    return SpawnedReferenceRobot(model_id, prim_path, base, translate, yaw, {"camera": camera_link, "treatment_head": head})


_BUILDERS = {
    "irobot-packbot-fukushima": _packbot,
    "flyability-elios3-rad": _elios3,
    "mhi-meister": _meister,
    "hitachi-ge-arounder": _arounder,
}


def spawn_reference_robot(stage: Any, model_id: str, prim_path: str) -> SpawnedReferenceRobot:
    """Author a lightweight, dimensioned real-robot reference model."""

    get_real_robot_reference(model_id)
    return _BUILDERS[model_id](stage, prim_path)
