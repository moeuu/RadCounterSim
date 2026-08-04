#!/usr/bin/env python3
"""Render and replay a robot survey in the measured JSI TRIGA environment."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ISAAC_EXTENSION = REPO_ROOT / "source/extensions/radcounter.isaac"
if str(ISAAC_EXTENSION) not in sys.path:
    sys.path.insert(0, str(ISAAC_EXTENSION))

from radcounter.core.environment import (  # noqa: E402
    CoordinateSystemConfig,
    EnvironmentFormat,
    EnvironmentImportConfig,
    EnvironmentImportPipeline,
    LengthUnit,
)
from radcounter.core.environment.importers import load_pcd_xyz_rgb  # noqa: E402
from radcounter.core.rendering import (  # noqa: E402
    DigitalTwinAssetConfig,
    DigitalTwinRenderingConfig,
    DigitalTwinSourceType,
    EnvironmentEffectsConfig,
    FacilityLightConfig,
    FacilityLightingConfig,
    LightType,
    RendererPolicyConfig,
    RenderPurpose,
    RenderQualityTier,
    probe_gpu_capabilities,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / ".cache/datasets/jsi-triga-2026",
    )
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=REPO_ROOT / "artifacts/jsi_triga_gui_validation",
    )
    return parser.parse_args()


def _environment_manifest(pcd_path: Path) -> Path:
    result = EnvironmentImportPipeline().import_environment(
        EnvironmentImportConfig(
            environment_id="jsi-triga-mark-ii-2026",
            uri=str(pcd_path.resolve()),
            format=EnvironmentFormat.PCD,
            coordinate_system=CoordinateSystemConfig(units=LengthUnit.M),
            default_material_id="concrete",
            point_cloud_voxel_size_m=0.10,
            max_point_cloud_voxels=200_000,
            max_triangles=3_000_000,
            cache_directory=str(REPO_ROOT / ".cache/environments"),
        ),
        base_directory=REPO_ROOT,
    )
    return result.manifest_path


def _rendering_config(
    manifest: Path,
    bounds: tuple[np.ndarray, np.ndarray],
    forced_tier: RenderQualityTier | None,
):
    low, high = bounds
    center = (low + high) * 0.5
    return DigitalTwinRenderingConfig(
        asset=DigitalTwinAssetConfig(
            physics_manifest_uri=str(manifest),
            source_type=DigitalTwinSourceType.LIDAR,
            prim_path="/World/TRIGA/SurfaceTwin",
            preserve_source_materials=False,
        ),
        lighting=FacilityLightingConfig(
            calibration_id="jsi-triga-survey-neutral-v1",
            lights=(
                FacilityLightConfig(
                    id="survey_ambient",
                    light_type=LightType.DOME,
                    intensity=280.0,
                    color_rgb=(0.68, 0.74, 0.80),
                ),
                FacilityLightConfig(
                    id="overhead_a",
                    light_type=LightType.RECT,
                    translation_m=(float(center[0] - 4.0), float(center[1]), 4.2),
                    rotation_rpy_deg=(0.0, 0.0, 0.0),
                    width_m=3.0,
                    height_m=0.35,
                    intensity=4200.0,
                    color_temperature_k=4300.0,
                ),
                FacilityLightConfig(
                    id="overhead_b",
                    light_type=LightType.RECT,
                    translation_m=(float(center[0] + 5.0), float(center[1]), 4.2),
                    width_m=3.0,
                    height_m=0.35,
                    intensity=4200.0,
                    color_temperature_k=4300.0,
                ),
            ),
        ),
        effects=EnvironmentEffectsConfig(
            bounds_min_m=tuple(float(value) for value in low),
            bounds_max_m=tuple(float(value) for value in high + np.asarray([0.0, 0.0, 2.0])),
            fog_density=0.0015,
            dust_particle_count=1200,
            spray_particle_count=1800,
        ),
        renderer=RendererPolicyConfig(
            target_frame_rate_hz=30.0,
            adaptive=True,
            forced_tier=forced_tier,
        ),
        render_products=(),
        base_directory=str(REPO_ROOT),
    )


def _color_map(values: np.ndarray) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(value, [2.0, 98.0])
    normalized = np.clip((value - low) / max(high - low, 1.0e-9), 0.0, 1.0)
    return np.column_stack(
        (
            np.clip(2.0 * normalized, 0.0, 1.0),
            np.clip(2.0 - 2.0 * np.abs(normalized - 0.5), 0.0, 1.0),
            np.clip(2.0 * (1.0 - normalized), 0.0, 1.0),
        )
    )


def _author_points(stage, path: str, points: np.ndarray, colors: np.ndarray, width: float):
    from pxr import Gf, Sdf, UsdGeom

    authored = UsdGeom.Points.Define(stage, path)
    authored.CreatePointsAttr([Gf.Vec3f(*value) for value in points])
    authored.CreateWidthsAttr([width] * len(points))
    color = UsdGeom.PrimvarsAPI(authored.GetPrim()).CreatePrimvar(
        "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
    )
    color.Set([Gf.Vec3f(*value) for value in colors])
    return authored


def _asset_root() -> str:
    try:
        from isaacsim.storage.native import get_assets_root_path
    except ImportError:
        from isaacsim.core.utils.nucleus import get_assets_root_path
    root = get_assets_root_path()
    if not root:
        raise RuntimeError("Isaac Sim asset root is unavailable; cannot load Clearpath Jackal")
    return root.rstrip("/")


def _author_jackal(stage, start: np.ndarray):
    from pxr import Gf, UsdGeom

    path = "/World/Robots/SurveyJackal"
    root = UsdGeom.Xform.Define(stage, path)
    root.GetPrim().GetReferences().AddReference(
        f"{_asset_root()}/Isaac/Robots/Clearpath/Jackal/jackal.usd"
    )
    xform = UsdGeom.Xformable(root.GetPrim())
    xform.ClearXformOpOrder()
    translate = xform.AddTranslateOp()
    yaw = xform.AddRotateZOp()
    translate.Set(Gf.Vec3d(float(start[0]), float(start[1]), 0.02))
    yaw.Set(0.0)
    return path, translate, yaw


def _author_camera(stage, low: np.ndarray, high: np.ndarray) -> str:
    from pxr import Gf, UsdGeom

    center = (low + high) * 0.5
    extent = high - low
    eye = center + np.asarray([extent[0] * 0.55, -extent[1] * 0.72, 13.0])
    target = center + np.asarray([0.0, 0.0, 0.45])
    path = "/World/Cameras/TRIGAOverview"
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(32.0)
    camera.CreateClippingRangeAttr((0.05, 10_000.0))
    matrix = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0.0, 0.0, 1.0)
    ).GetInverse()
    UsdGeom.Xformable(camera.GetPrim()).AddTransformOp().Set(matrix)
    return path


def main() -> None:
    args = arguments()
    dataset = args.dataset.expanduser().resolve()
    pcd_path = dataset / "lidar_maps_combined_cleaned.pcd"
    radiation_path = dataset / "rad_data_combined.csv"
    if not pcd_path.is_file() or not radiation_path.is_file():
        raise FileNotFoundError(
            f"Dataset is incomplete at {dataset}; run scripts/fetch_jsi_triga_dataset.sh"
        )
    points, colors = load_pcd_xyz_rgb(pcd_path)
    if colors is None:
        colors = np.full_like(points, 0.55)
    radiation = np.loadtxt(radiation_path, delimiter=",", ndmin=2)
    radiation_points = radiation[:, :3]
    radiation_counts = radiation[:, 3]
    low, high = np.min(points, axis=0), np.max(points, axis=0)
    manifest = _environment_manifest(pcd_path)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    gpu = probe_gpu_capabilities()
    available_vram_gb = min(
        gpu.vram_gb,
        gpu.free_vram_gb if gpu.free_vram_gb is not None else gpu.vram_gb,
    )
    low_memory = available_vram_gb < 6.0
    forced_tier = RenderQualityTier.FALLBACK if available_vram_gb < 3.5 else (
        RenderQualityTier.WEAK if low_memory else None
    )
    effective_capture = args.capture and not low_memory

    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": args.headless,
            "width": 960 if low_memory else 1600,
            "height": 540 if low_memory else 900,
            "renderer": "RaytracedLighting",
            "physics_gpu": -1 if low_memory else 0,
        }
    )
    import omni.timeline
    import omni.usd
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    from pxr import Gf, UsdGeom

    from radcounter.isaac.rendering import NuclearDigitalTwinRuntime

    context = omni.usd.get_context()
    context.new_stage()
    for _ in range(4):
        app.update()
    stage = context.get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    purpose = RenderPurpose.CAPTURE if effective_capture else RenderPurpose.INTERACTIVE
    runtime = NuclearDigitalTwinRuntime(
        stage,
        _rendering_config(manifest, (low, high), forced_tier),
        purpose=purpose,
    )
    load_task = asyncio.ensure_future(runtime.load())
    while not load_task.done() and app.is_running():
        app.update()
    if load_task.exception() is not None:
        raise load_task.exception()

    _author_points(
        stage,
        "/World/TRIGA/MeasuredColorPointCloud",
        points,
        colors,
        width=0.045,
    )
    _author_points(
        stage,
        "/World/TRIGA/MeasuredRadiationTrack",
        radiation_points + np.asarray([0.0, 0.0, 0.03]),
        _color_map(radiation_counts),
        width=0.075,
    )
    route_stride = max(1, len(radiation_points) // 1800)
    route = radiation_points[::route_stride]
    _, robot_translate, robot_yaw = _author_jackal(stage, route[0])
    camera_path = _author_camera(stage, low, high)
    viewport = get_active_viewport()
    viewport.set_active_camera(camera_path)

    timeline = omni.timeline.get_timeline_interface()
    started = time.monotonic()
    previous = started
    frame_times: list[float] = []
    route_distance = 0.0
    route_index = 0
    previous_position = route[0].copy()
    while app.is_running() and time.monotonic() - started < args.duration_s:
        now = time.monotonic()
        route_index = min(
            len(route) - 2,
            int((now - started) / max(args.duration_s, 0.001) * (len(route) - 1)),
        )
        fraction = (
            (now - started) / max(args.duration_s, 0.001) * (len(route) - 1) - route_index
        )
        position = route[route_index] * (1.0 - fraction) + route[route_index + 1] * fraction
        delta = route[route_index + 1] - route[route_index]
        robot_translate.Set(Gf.Vec3d(float(position[0]), float(position[1]), 0.02))
        robot_yaw.Set(float(math.degrees(math.atan2(delta[1], delta[0]))))
        app.update()
        elapsed_ms = (time.monotonic() - now) * 1000.0
        frame_times.append(elapsed_ms)
        runtime.observe_frame_time(elapsed_ms)
        route_distance += float(np.linalg.norm(position[:2] - previous_position[:2]))
        previous_position = position
        previous = now

    timeline.pause()
    for _ in range(180 if effective_capture else 60):
        app.update()
    screenshot = args.output_directory / "jsi_triga_gui.png"
    capture_viewport_to_file(viewport, str(screenshot))
    for _ in range(120):
        app.update()

    report = {
        "dataset": "JSI TRIGA Mark II May 2026",
        "license": "BSD-3-Clause",
        "pcd_points": int(len(points)),
        "radiation_samples": int(len(radiation_points)),
        "bounds_m": [low.tolist(), high.tolist()],
        "radiation_counts": {
            "minimum": float(np.min(radiation_counts)),
            "median": float(np.median(radiation_counts)),
            "maximum": float(np.max(radiation_counts)),
        },
        "renderer": load_task.result().renderer_mode,
        "quality_tier": load_task.result().renderer_tier,
        "gpu": gpu.name,
        "gpu_total_vram_gb": gpu.vram_gb,
        "gpu_free_vram_at_start_gb": gpu.free_vram_gb,
        "low_memory_fallback": low_memory,
        "capture_requested": args.capture,
        "capture_executed": effective_capture,
        "frames": len(frame_times),
        "frame_time_ms": {
            "mean": float(np.mean(frame_times)),
            "p95": float(np.percentile(frame_times, 95.0)),
            "maximum": float(np.max(frame_times)),
        },
        "final_route_index": route_index,
        "approximate_route_distance_m": route_distance,
        "screenshot": str(screenshot.resolve()),
        "screenshot_exists": screenshot.is_file(),
        "screenshot_bytes": screenshot.stat().st_size if screenshot.is_file() else 0,
        "manifest": str(manifest),
    }
    report_path = args.output_directory / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    timeline.stop()
    runtime.close()
    app.close()


if __name__ == "__main__":
    main()
