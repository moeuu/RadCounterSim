#!/usr/bin/env python3
"""Render and exercise the Manchester 500 L drum store in Isaac Sim 6."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

from isaacsim import SimulationApp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _free_vram_gb() -> float | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        )
        return float(output.splitlines()[0].strip()) / 1024.0
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return None


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value)
    return cleaned if cleaned and not cleaned[0].isdigit() else f"asset_{cleaned}"


def _set_pose(prim, pose, *, scale=None) -> None:
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*pose.translation_m))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(*(math.degrees(value) for value in pose.rotation_rpy_rad)))
    if scale is not None:
        xform.AddScaleOp().Set(Gf.Vec3f(*scale))


async def _convert_visuals(model, cache_root: Path) -> dict[Path, Path]:
    import omni.kit.asset_converter

    converter = omni.kit.asset_converter.get_instance()
    converted: dict[Path, Path] = {}
    unique_meshes = dict.fromkeys(visual.mesh_path for visual in model.visuals)
    for index, source in enumerate(unique_meshes, start=1):
        relative = source.relative_to(model.source_path.parent)
        destination = (cache_root / relative).with_suffix(".usd")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            converted[source] = destination
            continue
        context = omni.kit.asset_converter.AssetConverterContext()
        for attribute, value in (
            ("ignore_materials", False),
            ("merge_all_meshes", False),
            ("embed_textures", False),
            ("use_meter_as_world_unit", False),
        ):
            if hasattr(context, attribute):
                setattr(context, attribute, value)
        print(f"[convert {index}/{len(unique_meshes)}] {relative}", flush=True)
        task = converter.create_converter_task(str(source), str(destination), None, context)
        if not await task.wait_until_finished():
            error = (
                task.get_error_message() if hasattr(task, "get_error_message") else "unknown error"
            )
            raise RuntimeError(f"asset conversion failed for {source}: {error}")
        converted[source] = destination
    return converted


def _create_omnipbr_material(stage, path: str, textures):
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")

    texture_inputs = (
        ("diffuse_texture", textures.base_color),
        ("normalmap_texture", textures.normal),
        ("reflectionroughness_texture", textures.roughness),
        ("metallic_texture", textures.metallic),
        ("emissive_color_texture", textures.emissive),
    )
    for name, texture_path in texture_inputs:
        if texture_path is not None:
            shader.CreateInput(name, Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture_path)))
    return material


def _world_bounds(stage, root_path: str):
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    return cache.ComputeWorldBound(stage.GetPrimAtPath(root_path)).ComputeAlignedRange()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-root",
        type=Path,
        default=REPO_ROOT / ".cache/datasets/manchester-nuclear-assets/500L_Drum_Store",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/manchester_drum_store_gui_validation",
    )
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()

    from radcounter.core.environment.sdf_visual import load_sdf_visual_model

    model = load_sdf_visual_model(args.model_root.resolve() / "model.sdf")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    free_vram_before = _free_vram_gb()
    constrained = free_vram_before is not None and free_vram_before < 6.0
    width, height = (960, 540) if constrained else (1600, 900)
    renderer = "MinimalRendering" if constrained else "RayTracedLighting"

    app = SimulationApp(
        {
            "headless": False,
            "width": width,
            "height": height,
            "renderer": renderer,
            "anti_aliasing": 0 if constrained else 3,
            "minimal_shading_mode": 2,
            "multi_gpu": False,
            "max_gpu_count": 1,
            "physics_gpu": -1 if constrained else 0,
        }
    )

    import omni.kit.app
    import omni.kit.viewport.utility
    import omni.usd
    from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    model_root = UsdGeom.Xform.Define(stage, "/World/Manchester500L")
    _set_pose(model_root.GetPrim(), model.pose)
    UsdGeom.Scope.Define(stage, "/World/Looks")

    cache_root = REPO_ROOT / ".cache/usd/manchester-500l"
    converted = asyncio.get_event_loop().run_until_complete(_convert_visuals(model, cache_root))

    link_roots: dict[str, object] = {}
    visual_roots: list[tuple[object, object]] = []
    materials = []
    source_units: dict[Path, float] = {}
    for index, visual in enumerate(model.visuals):
        link_path = f"/World/Manchester500L/{_safe_name(visual.link_name)}"
        if visual.link_name not in link_roots:
            link = UsdGeom.Xform.Define(stage, link_path)
            _set_pose(link.GetPrim(), visual.link_pose)
            link_roots[visual.link_name] = link.GetPrim()
        visual_path = f"{link_path}/{_safe_name(visual.visual_name)}_{index:02d}"
        visual_root = UsdGeom.Xform.Define(stage, visual_path)
        converted_path = converted[visual.mesh_path]
        if converted_path not in source_units:
            source_stage = Usd.Stage.Open(str(converted_path), load=Usd.Stage.LoadNone)
            source_units[converted_path] = UsdGeom.GetStageMetersPerUnit(source_stage)
        unit_ratio = source_units[converted_path] / UsdGeom.GetStageMetersPerUnit(stage)
        effective_scale = tuple(value * unit_ratio for value in visual.mesh_scale)
        _set_pose(visual_root.GetPrim(), visual.visual_pose, scale=effective_scale)
        visual_root.GetPrim().GetReferences().AddReference(str(converted[visual.mesh_path]))
        material = _create_omnipbr_material(
            stage,
            f"/World/Looks/{_safe_name(visual.link_name)}_{index:02d}",
            visual.textures,
        )
        materials.append(material)
        visual_roots.append((visual_root.GetPrim(), material))

    for _ in range(20):
        app.update()
    if not constrained:
        for root_prim, material in visual_roots:
            for prim in stage.Traverse():
                if prim.GetPath().HasPrefix(root_prim.GetPath()) and prim.IsA(UsdGeom.Gprim):
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    for link_name, link_prim in link_roots.items():
        if "roof" in link_name.lower():
            UsdGeom.Imageable(link_prim).MakeInvisible()

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(650.0)
    dome.CreateColorAttr(Gf.Vec3f(0.78, 0.84, 0.92))
    for index, (position, intensity) in enumerate(
        (
            ((0.0, 0.0, 5.0), 1400.0),
            ((-4.0, 0.0, 4.0), 1100.0),
            ((4.0, 0.0, 4.0), 1100.0),
        )
    ):
        light = UsdLux.SphereLight.Define(stage, f"/World/Lights/WorkLight_{index}")
        light.AddTranslateOp().Set(Gf.Vec3d(*position))
        light.CreateIntensityAttr(intensity)
        light.CreateRadiusAttr(0.35)
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.91, 0.78))

    for _ in range(20):
        app.update()
    bounds = _world_bounds(stage, "/World/Manchester500L")
    minimum = bounds.GetMin()
    maximum = bounds.GetMax()
    center = (minimum + maximum) * 0.5
    extent = maximum - minimum

    robot_path = "/World/InspectionRobot"
    robot_pose = UsdGeom.Xform.Define(stage, robot_path)
    robot = UsdGeom.Xform.Define(stage, f"{robot_path}/Jackal")
    try:
        from isaacsim.storage.native import get_assets_root_path

        asset_root = get_assets_root_path()
        if asset_root:
            robot.GetPrim().GetReferences().AddReference(
                f"{asset_root}/Isaac/Robots/Clearpath/Jackal/jackal.usd"
            )
    except Exception as error:
        print(f"Jackal asset lookup failed: {error}", flush=True)
    robot_z = float(minimum[2]) + 0.16
    route_start = Gf.Vec3d(float(center[0] - 0.22 * extent[0]), float(center[1]), robot_z)
    route_end = Gf.Vec3d(float(center[0] + 0.22 * extent[0]), float(center[1]), robot_z)
    robot_translate = robot_pose.AddTranslateOp()
    robot_translate.Set(route_start)

    camera = UsdGeom.Camera.Define(stage, "/World/InspectionCamera")
    eye = Gf.Vec3d(
        float(center[0] - 0.08 * extent[0]),
        float(center[1] - 0.58 * extent[1]),
        float(maximum[2] + 0.48 * max(extent[0], extent[1])),
    )
    target = Gf.Vec3d(float(center[0]), float(center[1]), float(minimum[2] + 0.22 * extent[2]))
    camera_transform = camera.AddTransformOp()
    camera_transform.Set(Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse())
    camera.CreateFocalLengthAttr(17.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.05, 10000.0))
    viewport = omni.kit.viewport.utility.get_active_viewport()
    viewport.set_active_camera(camera.GetPath())
    viewport.set_texture_resolution((width, height))

    frame_times_ms: list[float] = []
    started = time.perf_counter()
    previous = started
    while app.is_running() and time.perf_counter() - started < args.duration:
        elapsed = time.perf_counter() - started
        phase = 0.5 - 0.5 * math.cos(2.0 * math.pi * elapsed / max(args.duration, 1.0))
        robot_translate.Set(route_start * (1.0 - phase) + route_end * phase)
        app.update()
        now = time.perf_counter()
        frame_times_ms.append((now - previous) * 1000.0)
        previous = now

    overview_screenshot = output_dir / "manchester_500l_overview.png"
    omni.kit.viewport.utility.capture_viewport_to_file(viewport, str(overview_screenshot))
    for _ in range(30):
        app.update()

    robot_translate.Set((route_start + route_end) * 0.5)
    close_eye = Gf.Vec3d(
        float(center[0]),
        float(center[1] - 0.25 * extent[1]),
        float(minimum[2] + 1.65),
    )
    close_target = Gf.Vec3d(float(center[0]), float(center[1]), float(minimum[2] + 0.45))
    camera_transform.Set(
        Gf.Matrix4d().SetLookAt(close_eye, close_target, Gf.Vec3d(0, 0, 1)).GetInverse()
    )
    for _ in range(30):
        app.update()
    screenshot = output_dir / "manchester_500l_gui.png"
    omni.kit.viewport.utility.capture_viewport_to_file(viewport, str(screenshot))
    for _ in range(30):
        app.update()

    ordered = sorted(frame_times_ms)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] if ordered else None
    report = {
        "status": "passed" if screenshot.is_file() and screenshot.stat().st_size > 0 else "failed",
        "data_class": "realistic_cc_by_simulation_asset_not_a_facility_scan",
        "source_doi": "10.48420/25224974.v1",
        "isaac_sim": "6.0.1",
        "renderer_requested": renderer,
        "low_free_vram_mode": constrained,
        "free_vram_gb_before": free_vram_before,
        "resolution": [width, height],
        "visual_count": len(model.visuals),
        "converted_mesh_count": len(converted),
        "pbr_material_count": len(materials),
        "material_mode": "source_base_color" if constrained else "full_sdf_pbr",
        "full_pbr_binding_active": not constrained,
        "source_usd_meters_per_unit": sorted(set(source_units.values())),
        "bounds_m": {"min": list(minimum), "max": list(maximum)},
        "robot": "Clearpath Jackal",
        "robot_route_m": {"start": list(route_start), "end": list(route_end)},
        "frames": len(frame_times_ms),
        "mean_frame_ms": sum(frame_times_ms) / len(frame_times_ms) if frame_times_ms else None,
        "p95_frame_ms": p95,
        "screenshot": str(screenshot),
        "overview_screenshot": str(overview_screenshot),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    app.close()


if __name__ == "__main__":
    main()
