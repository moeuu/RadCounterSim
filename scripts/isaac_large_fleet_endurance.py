#!/usr/bin/env python3
"""Ten-minute GUI endurance run over a streamed 4 km x 4 km environment."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import random
import resource
import time
from pathlib import Path
from statistics import fmean

import numpy as np
import yaml
from isaacsim import SimulationApp

from radcounter.core.performance import detect_hardware, profile_path_for_hardware


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--output", default=".cache/large-fleet-endurance")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


ARGS = parse_args()
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
HARDWARE = detect_hardware()
PROFILE_PATH = (
    profile_path_for_hardware(HARDWARE, REPOSITORY_ROOT)
    if ARGS.profile == "auto"
    else Path(ARGS.profile).expanduser().resolve()
)
PROFILE = yaml.safe_load(PROFILE_PATH.read_text(encoding="utf-8"))
if ARGS.duration is not None:
    PROFILE["duration_s"] = ARGS.duration
OUTPUT = Path(ARGS.output).expanduser().resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)
print(
    "ENDURANCE_PROFILE "
    + json.dumps(
        {
            "selected_profile": PROFILE["profile"],
            "profile_path": str(PROFILE_PATH),
            "gpu_name": HARDWARE.gpu_name,
            "vram_mb": HARDWARE.vram_mb,
            "driver_version": HARDWARE.driver_version,
            "detection_source": HARDWARE.detection_source,
        },
        sort_keys=True,
    ),
    flush=True,
)

VIEWPORT = PROFILE["viewport"]
APP = SimulationApp(
    {
        "headless": ARGS.headless,
        "width": int(VIEWPORT["width_px"]),
        "height": int(VIEWPORT["height_px"]),
        "renderer": VIEWPORT["renderer"],
    }
)

import carb
import omni.timeline
import omni.ui as ui
import omni.usd
from isaacsim.core.rendering_manager import ViewportManager
from omni.kit.viewport.utility import capture_viewport_to_file
from pxr import Gf, Usd, UsdGeom, UsdLux, Vt
from radcounter.isaac.robot import IsaacRobotSensorRigManager, spawn_reference_robot

from radcounter.core.performance import (
    AdaptiveWorkloadGovernor,
    QualityLevel,
    RateGate,
    RuntimeBudget,
    TileResidencyPlanner,
)
from radcounter.core.robots import (
    CameraSensorConfig,
    LidarDimension,
    LidarSensorConfig,
    RadiationSensorConfig,
    RobotImportConfig,
    RobotSensorType,
    RobotType,
    SensorMountConfig,
    SensorPoseConfig,
    reference_config,
)

QUALITY_NAMES = {
    "high": QualityLevel.HIGH,
    "balanced": QualityLevel.BALANCED,
    "conservative": QualityLevel.CONSERVATIVE,
    "minimal": QualityLevel.MINIMAL,
}


def configure_renderer() -> None:
    settings = carb.settings.get_settings()
    settings.set("/app/renderer/resolution/width", int(VIEWPORT["width_px"]))
    settings.set("/app/renderer/resolution/height", int(VIEWPORT["height_px"]))
    settings.set("/rtx/raytracing/fractionalCutoutOpacity", False)
    settings.set("/rtx/post/aa/op", 3)
    settings.set("/rtx/shadows/enabled", False)
    settings.set("/renderer/multiGpu/enabled", False)


def set_color(prim, color: tuple[float, float, float]) -> None:
    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*color)])


def scaled_cube(path: str, scale: tuple[float, float, float], color) -> Usd.Prim:
    cube = UsdGeom.Cube.Define(STAGE, path)
    cube.CreateSizeAttr(1.0)
    UsdGeom.Xformable(cube).AddScaleOp().Set(Gf.Vec3f(*scale))
    set_color(cube.GetPrim(), color)
    return cube.GetPrim()


def create_environment_instancers(tile_size: float):
    STAGE.DefinePrim("/World/Environment", "Xform")
    STAGE.DefinePrim("/World/Environment/Prototypes", "Xform")
    ground = scaled_cube(
        "/World/Environment/Prototypes/GroundTile",
        (tile_size, tile_size, 0.08),
        (0.20, 0.26, 0.22),
    )
    low = scaled_cube(
        "/World/Environment/Prototypes/LowObstacle",
        (5.0, 5.0, 2.5),
        (0.32, 0.36, 0.38),
    )
    tall = scaled_cube(
        "/World/Environment/Prototypes/TallObstacle",
        (7.0, 7.0, 10.0),
        (0.24, 0.30, 0.34),
    )
    source = UsdGeom.Sphere.Define(STAGE, "/World/Environment/Prototypes/Source")
    source.CreateRadiusAttr(0.45)
    set_color(source.GetPrim(), (0.95, 0.24, 0.08))

    floors = UsdGeom.PointInstancer.Define(STAGE, "/World/Environment/ResidentGround")
    floors.CreatePrototypesRel().SetTargets([ground.GetPath()])
    obstacles = UsdGeom.PointInstancer.Define(STAGE, "/World/Environment/ResidentObstacles")
    obstacles.CreatePrototypesRel().SetTargets([low.GetPath(), tall.GetPath()])
    sources = UsdGeom.PointInstancer.Define(STAGE, "/World/Environment/ResidentSources")
    sources.CreatePrototypesRel().SetTargets([source.GetPath()])
    return floors, obstacles, sources


def tile_content(key: tuple[int, int], tile_size: float, count: int):
    seed = ((key[0] * 73856093) ^ (key[1] * 19349663)) & 0xFFFFFFFF
    rng = random.Random(seed)
    cx = (key[0] + 0.5) * tile_size
    cy = (key[1] + 0.5) * tile_size
    obstacles = []
    attempts = 0
    while len(obstacles) < count and attempts < count * 8:
        attempts += 1
        ox = cx + rng.uniform(-0.38, 0.38) * tile_size
        oy = cy + rng.uniform(-0.38, 0.38) * tile_size
        in_route_corridor = any(
            (abs(abs(ox) - half_side) < 10.0 and abs(oy) <= half_side + 12.0)
            or (abs(abs(oy) - half_side) < 10.0 and abs(ox) <= half_side + 12.0)
            for half_side in (800.0, 900.0)
        )
        if in_route_corridor:
            continue
        obstacles.append(
            (
                ox,
                oy,
                1 if rng.random() < 0.30 else 0,
            )
        )
    source_activity = 2.0e8 + rng.random() * 8.0e8
    source = (
        cx + rng.uniform(-0.30, 0.30) * tile_size,
        cy + rng.uniform(-0.30, 0.30) * tile_size,
        source_activity,
    )
    return obstacles, source


def update_resident_geometry(tiles, decision, tile_size):
    floor_positions = []
    obstacle_positions = []
    obstacle_indices = []
    source_positions = []
    active_sources = []
    for key in tiles:
        cx, cy = PLANNER.center(key)
        floor_positions.append(Gf.Vec3f(cx, cy, -0.04))
        obstacles, source = tile_content(key, tile_size, decision.obstacles_per_tile)
        for ox, oy, proto in obstacles:
            height = 5.0 if proto == 0 else 20.0
            obstacle_positions.append(Gf.Vec3f(ox, oy, height * 0.5))
            obstacle_indices.append(proto)
        sx, sy, activity = source
        source_positions.append(Gf.Vec3f(sx, sy, 0.5))
        active_sources.append((sx, sy, 0.5, activity))

    FLOORS.GetPositionsAttr().Set(Vt.Vec3fArray(floor_positions))
    FLOORS.GetProtoIndicesAttr().Set(Vt.IntArray([0] * len(floor_positions)))
    OBSTACLES.GetPositionsAttr().Set(Vt.Vec3fArray(obstacle_positions))
    OBSTACLES.GetProtoIndicesAttr().Set(Vt.IntArray(obstacle_indices))
    SOURCES.GetPositionsAttr().Set(Vt.Vec3fArray(source_positions))
    SOURCES.GetProtoIndicesAttr().Set(Vt.IntArray([0] * len(source_positions)))
    return active_sources, len(obstacle_positions)


def create_robot_proxy(path: str, reference_model_id: str):
    robot = spawn_reference_robot(STAGE, reference_model_id, path)
    return robot.translation_op, robot.yaw_op


def square_route(elapsed_s: float, speed: float, side: float, altitude: float, phase: float = 0.0):
    perimeter = 4.0 * side
    distance = (elapsed_s * speed + phase) % perimeter
    half = 0.5 * side
    if distance < side:
        return -half + distance, -half, altitude
    if distance < 2.0 * side:
        return half, -half + distance - side, altitude
    if distance < 3.0 * side:
        return half - (distance - 2.0 * side), half, altitude
    return -half, half - (distance - 3.0 * side), altitude


def square_route_heading(elapsed_s: float, speed: float, side: float, phase: float = 0.0) -> float:
    distance_on_route = (elapsed_s * speed + phase) % (4.0 * side)
    if distance_on_route < side:
        return 0.0
    if distance_on_route < 2.0 * side:
        return 90.0
    if distance_on_route < 3.0 * side:
        return 180.0
    return -90.0


def distance(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def radiation_rates(position, sources, elapsed_s: float):
    unshielded = 0.2
    shielded = 0.2
    decontaminated = 0.2
    mitigation = PROFILE["mitigation"]
    shield_active = elapsed_s >= float(mitigation["shield_at_s"])
    decon_active = elapsed_s >= float(mitigation["water_decon_at_s"])
    lead = float(mitigation["lead_transmission"])
    retained = float(mitigation["retained_activity_after_water_decon"])
    for sx, sy, sz, activity in sources:
        r2 = max(0.25, (position[0] - sx) ** 2 + (position[1] - sy) ** 2 + (position[2] - sz) ** 2)
        contribution = activity * 2.5e-6 / (4.0 * math.pi * r2)
        unshielded += contribution
        shielded += contribution * (lead if shield_active and sx > position[0] else 1.0)
        decontaminated += contribution * (retained if decon_active else 1.0)
    return unshielded, shielded, decontaminated


def set_overview_camera(target, distance_m: float = 42.0):
    ViewportManager.set_camera_view(
        OVERVIEW_CAMERA,
        eye=[target[0] - 30.0, target[1] - 30.0, target[2] + distance_m],
        target=list(target),
    )


def percentile(values, fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1)))]


def capture(name: str) -> str:
    path = str(OUTPUT / name)
    ViewportManager.wait_for_viewport(viewport=SURVEY_VIEWPORT, max_frames=60)
    capture_viewport_to_file(SURVEY_VIEWPORT, path)
    return path


def save_rgb_frame(data, name: str) -> str:
    from PIL import Image

    pixels = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
    pixels = np.asarray(pixels)
    if pixels.ndim == 4 and pixels.shape[0] == 1:
        pixels = pixels[0]
    if pixels.shape[-1] > 3:
        pixels = pixels[..., :3]
    if np.issubdtype(pixels.dtype, np.floating):
        pixels = np.clip(pixels * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        pixels = pixels.astype(np.uint8, copy=False)
    path = OUTPUT / name
    Image.fromarray(pixels).save(path)
    return str(path)


configure_renderer()
CONTEXT = omni.usd.get_context()
CONTEXT.new_stage()
STAGE = CONTEXT.get_stage()
UsdGeom.SetStageUpAxis(STAGE, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(STAGE, 1.0)
STAGE.DefinePrim("/World", "Xform")

sun = UsdLux.DistantLight.Define(STAGE, "/World/Sun")
sun.CreateIntensityAttr(2500.0)
sun.CreateAngleAttr(0.7)
sun_xform = UsdGeom.Xformable(sun)
sun_xform.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 25.0, 20.0))

environment = PROFILE["environment"]
tile_size = float(environment["tile_size_m"])
FLOORS, OBSTACLES, SOURCES = create_environment_instancers(tile_size)
PLANNER = TileResidencyPlanner(
    tile_size,
    float(environment["logical_size_m"]),
    int(environment["maximum_resident_tiles"]),
)

STAGE.DefinePrim("/World/Robots", "Xform")
ugv_translate, ugv_rotate = create_robot_proxy(
    "/World/Robots/SurveyUGV", "irobot-packbot-fukushima"
)
drone_translate, drone_rotate = create_robot_proxy(
    "/World/Robots/SurveyDrone", "flyability-elios3-rad"
)

OVERVIEW_CAMERA = UsdGeom.Camera.Define(STAGE, "/World/OverviewCamera")
OVERVIEW_CAMERA.CreateFocalLengthAttr(24.0)
OVERVIEW_CAMERA.CreateClippingRangeAttr(Gf.Vec2f(0.1, 5000.0))
SURVEY_VIEWPORT = ViewportManager.get_viewport_api("Viewport")
if SURVEY_VIEWPORT is None:
    SURVEY_VIEWPORT = ViewportManager.create_viewport_window(
        title="RadCounterSim Survey",
        resolution=(int(VIEWPORT["width_px"]), int(VIEWPORT["height_px"])),
        camera="/OmniverseKit_Persp",
    ).viewport_api
ViewportManager.set_resolution(
    (int(VIEWPORT["width_px"]), int(VIEWPORT["height_px"])),
    render_product_or_viewport=SURVEY_VIEWPORT,
)
ViewportManager.set_camera(OVERVIEW_CAMERA, render_product_or_viewport=SURVEY_VIEWPORT)
set_overview_camera((0.0, 0.0, 2.0))

sensor_profile = PROFILE["sensors"]
ugv_config = RobotImportConfig(
    id="survey_ugv",
    uri=str(REPOSITORY_ROOT / "assets/robots/packbot_fukushima.urdf"),
    prim_path="/World/Robots/SurveyUGV",
    robot_type=RobotType.WHEELED,
    reference=reference_config("irobot-packbot-fukushima"),
    sensors=(
        SensorMountConfig(
            id="roof_lidar",
            type=RobotSensorType.LIDAR,
            parent_link="base_link",
            update_rate_hz=float(sensor_profile["lidar_tick_hz"]),
            pose=SensorPoseConfig(translation_m=(0.02, 0.0, 0.68)),
            lidar=LidarSensorConfig(
                dimension=LidarDimension.VOLUMETRIC,
                model=str(sensor_profile["lidar_model"]),
                draw_point_cloud=False,
            ),
        ),
        SensorMountConfig(
            id="gm_counter",
            type=RobotSensorType.RADIATION,
            parent_link="base_link",
            pose=SensorPoseConfig(translation_m=(0.03, 0.16, 0.41)),
            radiation=RadiationSensorConfig(detector_model_id="gm_tube"),
        ),
    ),
)
drone_config = RobotImportConfig(
    id="survey_drone",
    uri=str(REPOSITORY_ROOT / "assets/robots/elios3_rad.urdf"),
    prim_path="/World/Robots/SurveyDrone",
    robot_type=RobotType.AERIAL,
    reference=reference_config("flyability-elios3-rad"),
    sensors=(
        SensorMountConfig(
            id="mapping_rgbd",
            type=RobotSensorType.CAMERA,
            parent_link="base_link",
            update_rate_hz=float(sensor_profile["camera_tick_hz"]),
            pose=SensorPoseConfig(
                translation_m=(0.155, 0.0, -0.015),
                rotation_rpy_deg=(0.0, -78.0, 0.0),
            ),
            camera=CameraSensorConfig(
                width_px=int(sensor_profile["camera_width_px"]),
                height_px=int(sensor_profile["camera_height_px"]),
                annotators=("rgb", "distance_to_image_plane"),
                clipping_range_m=(0.2, 200.0),
            ),
        ),
        SensorMountConfig(
            id="czt_counter",
            type=RobotSensorType.RADIATION,
            parent_link="base_link",
            pose=SensorPoseConfig(translation_m=(-0.2, 0.0, -0.1)),
            radiation=RadiationSensorConfig(detector_model_id="czt"),
        ),
    ),
)

RIG = IsaacRobotSensorRigManager(STAGE)
RIG.mount_robot(ugv_config)
RIG.mount_robot(drone_config)
TIMELINE = omni.timeline.get_timeline_interface()
TIMELINE.play()

quality = QUALITY_NAMES[str(environment["initial_quality"])]
GOVERNOR = AdaptiveWorkloadGovernor(
    RuntimeBudget(
        target_fps=float(PROFILE["target_fps"]),
        initial_quality=quality,
        max_resident_tiles=int(environment["maximum_resident_tiles"]),
    )
)
GATES = RateGate()

window = ui.Window("RadCounterSim Endurance", width=430, height=310)
with window.frame, ui.VStack(spacing=5):
    ui.Label("4 km x 4 km survey: PackBot + Elios 3 RAD")
    phase_label = ui.Label("Starting")
    fps_label = ui.Label("FPS: --")
    quality_label = ui.Label("Quality: --")
    tile_label = ui.Label("Resident tiles: --")
    robot_label = ui.Label("Robot distance: --")
    sensor_label = ui.Label("LiDAR/Camera: --")
    radiation_label = ui.Label("Radiation: --")
    memory_label = ui.Label("Memory: --")

for _ in range(30):
    APP.update()

duration_s = float(PROFILE["duration_s"])
robot_profile = PROFILE["robots"]
frame_times = []
fps_window = []
peak_rss_mb = 0.0
peak_resident_tiles = 0
peak_obstacles = 0
lidar_reads = 0
camera_rgb_reads = 0
camera_depth_reads = 0
last_rgb_frame = None
sensor_errors: dict[str, int] = {}
radiation_samples = 0
radiation_unshielded = []
radiation_shielded = []
radiation_decontaminated = []
radiation_elapsed = []
active_sources = []
resident_tiles = ()
previous_ugv = None
previous_drone = None
ugv_distance = 0.0
drone_distance = 0.0
screenshots = []
start = time.perf_counter()
last_frame = start
last_telemetry = -30.0
captured_middle = False
captured_sensor_start = False
passed = False

try:
    while True:
        now = time.perf_counter()
        elapsed = now - start
        if elapsed >= duration_s:
            break

        APP.update()
        frame_end = time.perf_counter()
        frame_time = frame_end - last_frame
        last_frame = frame_end
        frame_times.append(frame_time)
        fps_window.append(frame_time)
        if len(fps_window) > 120:
            fps_window.pop(0)
        decision = GOVERNOR.observe(frame_time)

        ugv = square_route(
            elapsed,
            float(robot_profile["ugv_speed_m_s"]),
            1600.0,
            0.0,
        )
        drone = square_route(
            elapsed,
            float(robot_profile["drone_speed_m_s"]),
            1800.0,
            float(robot_profile["drone_altitude_m"]) + 2.0 * math.sin(elapsed * 0.08),
            phase=900.0,
        )
        ugv_translate.Set(Gf.Vec3d(*ugv))
        drone_translate.Set(Gf.Vec3d(*drone))
        ugv_rotate.Set(square_route_heading(elapsed, float(robot_profile["ugv_speed_m_s"]), 1600.0))
        drone_rotate.Set(
            square_route_heading(
                elapsed,
                float(robot_profile["drone_speed_m_s"]),
                1800.0,
                phase=900.0,
            )
        )
        if previous_ugv is not None:
            ugv_distance += distance(ugv, previous_ugv)
            drone_distance += distance(drone, previous_drone)
        previous_ugv, previous_drone = ugv, drone

        if GATES.due("environment", elapsed, decision.environment_update_hz):
            resident_tiles = PLANNER.select(
                ((ugv[0], ugv[1]), (drone[0], drone[1])),
                decision.tile_radius,
            )
            active_sources, obstacle_count = update_resident_geometry(
                resident_tiles, decision, tile_size
            )
            peak_resident_tiles = max(peak_resident_tiles, len(resident_tiles))
            peak_obstacles = max(peak_obstacles, obstacle_count)

        if GATES.due("lidar", elapsed, decision.lidar_read_hz):
            try:
                data, _ = RIG.read("survey_ugv", "roof_lidar", "generic-model-output")
                if data is not None:
                    lidar_reads += 1
            except Exception as exc:
                sensor_errors[type(exc).__name__] = sensor_errors.get(type(exc).__name__, 0) + 1

        if GATES.due("camera", elapsed, decision.camera_read_hz):
            try:
                rgb, _ = RIG.read("survey_drone", "mapping_rgbd", "rgb")
                if rgb is not None:
                    camera_rgb_reads += 1
                    last_rgb_frame = rgb
                    if not captured_sensor_start and elapsed >= 1.0:
                        screenshots.append(save_rgb_frame(last_rgb_frame, "drone_rgb_start.png"))
                        captured_sensor_start = True
                depth, _ = RIG.read("survey_drone", "mapping_rgbd", "distance_to_image_plane")
                if depth is not None:
                    camera_depth_reads += 1
            except Exception as exc:
                sensor_errors[type(exc).__name__] = sensor_errors.get(type(exc).__name__, 0) + 1

        if GATES.due("radiation", elapsed, decision.radiation_update_hz):
            ugv_rates = radiation_rates(ugv, active_sources, elapsed)
            drone_rates = radiation_rates(drone, active_sources, elapsed)
            radiation_unshielded.append(0.5 * (ugv_rates[0] + drone_rates[0]))
            radiation_shielded.append(0.5 * (ugv_rates[1] + drone_rates[1]))
            radiation_decontaminated.append(0.5 * (ugv_rates[2] + drone_rates[2]))
            radiation_elapsed.append(elapsed)
            radiation_samples += 2

        if GATES.due("ui", elapsed, 4.0):
            fps = 1.0 / max(1e-6, fmean(fps_window))
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            peak_rss_mb = max(peak_rss_mb, rss_mb)
            phase = "survey"
            if elapsed >= float(PROFILE["mitigation"]["water_decon_at_s"]):
                phase = "post-water-decon survey"
            elif elapsed >= float(PROFILE["mitigation"]["shield_at_s"]):
                phase = "post-shield survey"
            phase_label.text = f"{phase}: {elapsed:6.1f}/{duration_s:.0f} s"
            fps_label.text = f"FPS: {fps:5.1f}  frame p95 pending"
            quality_label.text = f"Quality: {decision.quality.name.lower()}"
            tile_label.text = f"Resident: {len(resident_tiles)}/{PLANNER.logical_tile_count} tiles"
            robot_label.text = f"UGV {ugv_distance:.0f} m | Drone {drone_distance:.0f} m"
            sensor_label.text = (
                f"LiDAR {lidar_reads} | RGB {camera_rgb_reads} | Depth {camera_depth_reads}"
            )
            current_rad = radiation_unshielded[-1] if radiation_unshielded else 0.0
            radiation_label.text = f"Radiation samples {radiation_samples} | {current_rad:.2f} cps"
            memory_label.text = f"Process peak RSS: {rss_mb:.0f} MiB"
            set_overview_camera(
                (
                    0.5 * (ugv[0] + drone[0]),
                    0.5 * (ugv[1] + drone[1]),
                    0.5 * (ugv[2] + drone[2]),
                )
            )

        if elapsed - last_telemetry >= 30.0:
            last_telemetry = elapsed
            fps = 1.0 / max(1e-6, fmean(fps_window))
            print(
                "ENDURANCE_TELEMETRY "
                + json.dumps(
                    {
                        "elapsed_s": round(elapsed, 1),
                        "fps": round(fps, 2),
                        "quality": decision.quality.name.lower(),
                        "resident_tiles": len(resident_tiles),
                        "ugv_distance_m": round(ugv_distance, 1),
                        "drone_distance_m": round(drone_distance, 1),
                        "lidar_reads": lidar_reads,
                        "camera_rgb_reads": camera_rgb_reads,
                        "radiation_samples": radiation_samples,
                        "peak_rss_mb": round(peak_rss_mb, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if not captured_middle and elapsed >= 0.5 * duration_s:
            if last_rgb_frame is not None:
                screenshots.append(save_rgb_frame(last_rgb_frame, "drone_rgb_middle.png"))
            captured_middle = True

    for _ in range(20):
        APP.update()
    if last_rgb_frame is not None:
        screenshots.append(save_rgb_frame(last_rgb_frame, "drone_rgb_final.png"))

    actual_duration = time.perf_counter() - start
    average_fps = len(frame_times) / max(actual_duration, 1e-9)
    shield_at = float(PROFILE["mitigation"]["shield_at_s"])
    decon_at = float(PROFILE["mitigation"]["water_decon_at_s"])
    shield_pairs = [
        (base, mitigated)
        for sample_t, base, mitigated in zip(
            radiation_elapsed,
            radiation_unshielded,
            radiation_shielded,
            strict=True,
        )
        if shield_at <= sample_t < decon_at
    ]
    decon_pairs = [
        (base, mitigated)
        for sample_t, base, mitigated in zip(
            radiation_elapsed,
            radiation_unshielded,
            radiation_decontaminated,
            strict=True,
        )
        if sample_t >= decon_at
    ]
    if not shield_pairs:
        shield_pairs = list(
            zip(radiation_unshielded, radiation_shielded, strict=True)
        )
    if not decon_pairs:
        decon_pairs = list(
            zip(radiation_unshielded, radiation_decontaminated, strict=True)
        )
    post_shield_base, post_shield = zip(*shield_pairs, strict=True)
    post_decon_base, post_decon = zip(*decon_pairs, strict=True)
    shield_ratio = fmean(post_shield) / max(fmean(post_shield_base), 1e-12)
    decon_ratio = fmean(post_decon) / max(fmean(post_decon_base), 1e-12)
    passed = all(
        (
            actual_duration >= duration_s * 0.98,
            average_fps >= float(PROFILE["minimum_acceptable_fps"]),
            peak_resident_tiles <= int(environment["maximum_resident_tiles"]),
            lidar_reads >= max(1, int(duration_s * 0.25)),
            camera_rgb_reads >= max(1, int(duration_s * 0.25)),
            camera_depth_reads >= max(1, int(duration_s * 0.25)),
            radiation_samples >= max(2, int(duration_s)),
            not sensor_errors,
        )
    )
    result = {
        "passed": passed,
        "profile": PROFILE["profile"],
        "robot_references": {
            "ugv": "irobot-packbot-fukushima",
            "aerial": "flyability-elios3-rad",
            "geometry_fidelity": "reference_procedural",
        },
        "hardware": {
            "gpu_name": HARDWARE.gpu_name,
            "vram_mb": HARDWARE.vram_mb,
            "driver_version": HARDWARE.driver_version,
            "detected_tier": HARDWARE.tier.value,
            "detection_source": HARDWARE.detection_source,
        },
        "requested_duration_s": duration_s,
        "actual_duration_s": actual_duration,
        "frames": len(frame_times),
        "average_fps": average_fps,
        "frame_time_p50_ms": 1000.0 * percentile(frame_times, 0.50),
        "frame_time_p95_ms": 1000.0 * percentile(frame_times, 0.95),
        "frame_time_p99_ms": 1000.0 * percentile(frame_times, 0.99),
        "peak_process_rss_mb": peak_rss_mb,
        "logical_environment_size_m": float(environment["logical_size_m"]),
        "logical_tile_count": PLANNER.logical_tile_count,
        "peak_resident_tiles": peak_resident_tiles,
        "peak_resident_obstacles": peak_obstacles,
        "ugv_distance_m": ugv_distance,
        "drone_distance_m": drone_distance,
        "lidar_reads": lidar_reads,
        "camera_rgb_reads": camera_rgb_reads,
        "camera_depth_reads": camera_depth_reads,
        "radiation_samples": radiation_samples,
        "shield_to_unshielded_ratio": shield_ratio,
        "water_decon_to_untreated_ratio": decon_ratio,
        "sensor_errors": sensor_errors,
        "quality_transitions": [
            {
                "frame": frame,
                "from": old.name.lower(),
                "to": new.name.lower(),
                "mean_frame_ms": mean_s * 1000.0,
            }
            for frame, old, new, mean_s in GOVERNOR.transitions
        ],
        "screenshots": screenshots,
    }
    (OUTPUT / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("ENDURANCE_RESULT " + json.dumps(result, sort_keys=True), flush=True)
finally:
    TIMELINE.stop()
    RIG.close()
    APP.close()

raise SystemExit(0 if passed else 2)
