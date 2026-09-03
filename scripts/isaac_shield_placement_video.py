#!/usr/bin/env python3
"""Render an articulated Ridgeback+Franka placing a shield before a surface source."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib
import json
import math
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "source/extensions/radcounter.isaac"
for search_path in (ROOT, EXTENSION):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

_radcounter_package = importlib.import_module("radcounter")
_extension_namespace = str(EXTENSION / "radcounter")
if _extension_namespace not in _radcounter_package.__path__:
    _radcounter_package.__path__.append(_extension_namespace)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--capture-stride", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/shield-placement-video",
    )
    args = parser.parse_args()
    if args.duration <= 0.0:
        parser.error("--duration must be positive")
    if args.capture_stride <= 0:
        parser.error("--capture-stride must be positive")
    return args


ARGS = arguments()

from isaacsim import SimulationApp

APP = SimulationApp(
    {
        "headless": ARGS.headless,
        "width": 1280,
        "height": 720,
        "renderer": "RaytracedLighting",
        "window_title": "RadCounterSim - Physical Shield Placement",
    }
)


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


def world_position(stage: Any, path: str) -> np.ndarray:
    from pxr import Gf, UsdGeom

    transform = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))
    return np.asarray(transform.Transform(Gf.Vec3d()), dtype=np.float64)


def set_camera(stage: Any) -> None:
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Gf, UsdGeom, UsdLux

    camera_path = "/World/ShieldVideoCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(23.0)
    view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(-2.75, -3.45, 2.55),
        Gf.Vec3d(1.15, -0.05, 0.62),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    camera_xform = UsdGeom.Xformable(camera)
    camera_xform.ClearXformOpOrder()
    camera_xform.MakeMatrixXform().Set(view.GetInverse())
    dome = UsdLux.DomeLight.Define(stage, "/World/ShieldVideoLight")
    dome.CreateIntensityAttr(920.0)
    dome.CreateColorAttr(Gf.Vec3f(0.70, 0.78, 0.88))
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("active viewport is unavailable")
    viewport.set_active_camera(camera_path)


def author_irregular_drum_surface_source(stage: Any) -> dict[str, Any]:
    from pxr import Gf, Sdf, UsdGeom

    drum_path = "/World/HiddenContaminatedDrum"
    drum = stage.GetPrimAtPath(drum_path)
    drum.GetAttribute("rad:role").Set("contaminated_object")
    drum.GetAttribute("rad:source:enabled").Set(False)
    source_path = drum_path + "/AdheredSurfaceContamination"
    angular_cells = 44
    vertical_cells = 22
    theta_values = np.linspace(math.pi - 1.18, math.pi + 1.18, angular_cells)
    z_values = np.linspace(-0.42, 0.42, vertical_cells)
    theta, zz = np.meshgrid(theta_values, z_values, indexing="xy")
    lobe_a = np.exp(-(((theta - math.pi + 0.24) / 0.48) ** 2 + ((zz + 0.05) / 0.27) ** 2))
    lobe_b = 0.78 * np.exp(
        -(((theta - math.pi - 0.48) / 0.28) ** 2 + ((zz - 0.16) / 0.18) ** 2)
    )
    lobe_c = 0.58 * np.exp(
        -(((theta - math.pi + 0.73) / 0.20) ** 2 + ((zz - 0.25) / 0.13) ** 2)
    )
    roughness = 0.17 * np.sin(11.0 * theta + 17.0 * zz) + 0.11 * np.cos(
        23.0 * theta - 8.0 * zz
    )
    active = lobe_a + lobe_b + lobe_c + roughness > 0.43
    active &= ~(
        ((theta - math.pi + 0.08) / 0.14) ** 2 + ((zz - 0.02) / 0.11) ** 2 < 1.0
    )
    active |= (
        ((theta - math.pi - 0.93) / 0.08) ** 2 + ((zz + 0.32) / 0.07) ** 2 < 1.0
    )

    radius_m = 0.326
    delta_theta = float(theta_values[1] - theta_values[0])
    delta_z = float(z_values[1] - z_values[0])
    points: list[Gf.Vec3f] = []
    counts: list[int] = []
    indices: list[int] = []
    weights: list[float] = []
    colors: list[Gf.Vec3f] = []
    field = np.maximum(0.08, lobe_a + lobe_b + lobe_c + 0.45 * roughness)
    maximum = max(float(field[active].max()), 1e-12)
    for row, column in np.argwhere(active):
        center_theta = float(theta[row, column])
        center_z = float(zz[row, column])
        base_index = len(points)
        for angle, z_m in (
            (center_theta - 0.52 * delta_theta, center_z - 0.52 * delta_z),
            (center_theta + 0.52 * delta_theta, center_z - 0.52 * delta_z),
            (center_theta + 0.52 * delta_theta, center_z + 0.52 * delta_z),
            (center_theta - 0.52 * delta_theta, center_z + 0.52 * delta_z),
        ):
            points.append(Gf.Vec3f(radius_m * math.cos(angle), radius_m * math.sin(angle), z_m))
        counts.append(4)
        indices.extend((base_index, base_index + 1, base_index + 2, base_index + 3))
        weight = float(field[row, column])
        weights.append(weight)
        fraction = weight / maximum
        colors.append(Gf.Vec3f(0.34 + 0.48 * fraction, 0.045, 0.018))

    total_activity_bq = 8.5e8
    normalized = np.asarray(weights, dtype=np.float64)
    activities = total_activity_bq * normalized / normalized.sum()
    mesh = UsdGeom.Mesh.Define(stage, source_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set(colors)
    prim = mesh.GetPrim()
    for name, value_type, value in (
        ("rad:role", Sdf.ValueTypeNames.String, "source"),
        ("rad:source:type", Sdf.ValueTypeNames.String, "surface"),
        ("rad:source:isotopeId", Sdf.ValueTypeNames.String, "Cs-137"),
        ("rad:source:activityBq", Sdf.ValueTypeNames.Double, total_activity_bq),
        ("rad:source:faceActivityBq", Sdf.ValueTypeNames.DoubleArray, activities.tolist()),
        ("rad:source:irregularMask", Sdf.ValueTypeNames.Bool, True),
        ("rad:source:movableWithPrim", Sdf.ValueTypeNames.Bool, True),
        ("rad:source:enabled", Sdf.ValueTypeNames.Bool, True),
    ):
        prim.CreateAttribute(name, value_type, custom=True).Set(value)
    return {
        "path": source_path,
        "active_faces": int(np.count_nonzero(active)),
        "candidate_faces": int(active.size),
        "total_activity_bq": total_activity_bq,
    }


def simplify_scene(stage: Any) -> None:
    from pxr import UsdGeom

    for path in (
        "/World/MeasurementRobot",
        "/World/DeconWorkbench",
        "/World/DeconWorkSurface",
        "/World/ContaminatedFloor",
        "/World/MovableObstacle",
        "/World/DisposalZone",
    ):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid() and prim.IsA(UsdGeom.Imageable):
            UsdGeom.Imageable(prim).MakeInvisible()
    # The generic task author adds stand-off grasp markers to movable props.
    # They are useful for controller audits but looked like unsupported,
    # floating hardware in the research render.  The contaminated drum is not
    # manipulated in this scene, so remove its marker and keep only the
    # physical shield handle that Franka actually grasps.
    drum_handle = "/World/HiddenContaminatedDrum/ManipulatorHandle"
    if stage.GetPrimAtPath(drum_handle).IsValid():
        stage.RemovePrim(drum_handle)
    # This scene is exclusively a shielding task. Hide the cyan
    # decontamination pad before the first recorded frame instead of waiting
    # until the pick-and-place controller starts.
    decon_tool = "/World/CountermeasureRobot/panda_hand/RadCounterDeconTool"
    if stage.GetPrimAtPath(decon_tool).IsValid():
        UsdGeom.Imageable(stage.GetPrimAtPath(decon_tool)).MakeInvisible()


def ground_fix_drum(stage: Any) -> dict[str, Any]:
    """Turn the source drum into a floor-supported static PhysX collider."""

    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    drum_path = "/World/HiddenContaminatedDrum"
    drum = stage.GetPrimAtPath(drum_path)
    if not drum.IsValid():
        raise RuntimeError("contaminated drum is missing")
    xform = UsdGeom.Xformable(drum)
    translate_ops = [
        operation
        for operation in xform.GetOrderedXformOps()
        if operation.GetOpType() == UsdGeom.XformOp.TypeTranslate
    ]
    if len(translate_ops) != 1:
        raise RuntimeError("contaminated drum must have one translate operation")
    position = translate_ops[0].Get()
    # The drum is 1.0 m tall. A centre height of 0.5 m puts its collision
    # cylinder directly on the z=0 floor instead of leaving a 5 cm gap.
    translate_ops[0].Set(Gf.Vec3d(float(position[0]), float(position[1]), 0.5))
    UsdPhysics.RigidBodyAPI.Apply(drum).CreateRigidBodyEnabledAttr(False)
    for name in ("rad:manipulation:movable", "rad:manipulation:removable"):
        attribute = drum.GetAttribute(name)
        if attribute:
            attribute.Set(False)
    drum.CreateAttribute(
        "rad:manipulation:groundFixed",
        Sdf.ValueTypeNames.Bool,
        custom=True,
    ).Set(True)
    return {
        "path": drum_path,
        "physics_mode": "static collider",
        "bottom_height_m": 0.0,
        "ground_fixed": True,
    }


class FilteredContactAudit:
    """Record pair-wise PhysX forces for floor and drum collision gates."""

    def __init__(self, contact_view: Any, *, physics_dt_s: float) -> None:
        self.contact_view = contact_view
        self.physics_dt_s = physics_dt_s
        self.samples = 0
        self.peak_floor_force_n = 0.0
        self.peak_drum_force_n = 0.0
        self.drum_contact_samples = 0

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=np.float64)

    def sample(self) -> None:
        matrix = self._numpy(
            self.contact_view.get_contact_force_matrix(dt=self.physics_dt_s)
        )
        if matrix.ndim != 3 or matrix.shape[-2:] != (2, 3):
            raise RuntimeError(
                f"expected shield contact matrix (*, 2, 3), got {matrix.shape}"
            )
        magnitudes = np.linalg.norm(matrix, axis=2)
        floor_force = float(np.max(magnitudes[:, 0], initial=0.0))
        drum_force = float(np.max(magnitudes[:, 1], initial=0.0))
        self.samples += 1
        self.peak_floor_force_n = max(self.peak_floor_force_n, floor_force)
        self.peak_drum_force_n = max(self.peak_drum_force_n, drum_force)
        if drum_force > 0.5:
            self.drum_contact_samples += 1

    def summary(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "peak_floor_force_n": self.peak_floor_force_n,
            "peak_drum_force_n": self.peak_drum_force_n,
            "drum_contact_samples": self.drum_contact_samples,
            "collision_free": self.drum_contact_samples == 0,
        }


class FrameRecorder:
    def __init__(self, output: Path, *, stride: int) -> None:
        from omni.kit.viewport.utility import get_active_viewport

        self.frame_dir = output / ".frames"
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        for frame in self.frame_dir.glob("frame_*.png"):
            frame.unlink()
        self.viewport = get_active_viewport()
        if self.viewport is None:
            raise RuntimeError("active viewport is unavailable")
        self.stride = stride
        self.physics_steps = 0
        self.frame_count = 0

    def observe(self) -> None:
        from omni.kit.viewport.utility import capture_viewport_to_file

        self.physics_steps += 1
        if self.physics_steps % self.stride:
            return
        path = self.frame_dir / f"frame_{self.frame_count:04d}.png"
        capture_viewport_to_file(self.viewport, file_path=str(path), is_hdr=False)
        self.frame_count += 1

    def wait_for_frames(self, world: Any) -> None:
        deadline = time.monotonic() + 8.0
        found = len(list(self.frame_dir.glob("frame_*.png")))
        while found < self.frame_count and time.monotonic() < deadline:
            world.step(render=True)
            time.sleep(0.02)
            found = len(list(self.frame_dir.glob("frame_*.png")))
        if found != self.frame_count:
            raise RuntimeError(f"expected {self.frame_count} frames, found {found}")

    def encode(self, output: Path, duration_s: float) -> Path:
        video = output / "manipulator_surface_source_shield_placement_20s.mp4"
        video.unlink(missing_ok=True)
        source_fps = self.frame_count / duration_s
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                f"{source_fps:.9f}",
                "-i",
                str(self.frame_dir / "frame_%04d.png"),
                "-t",
                f"{duration_s:.6f}",
                "-vf",
                "format=yuv420p",
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
                str(video),
            ],
            check=True,
        )
        for frame in self.frame_dir.glob("frame_*.png"):
            frame.unlink()
        self.frame_dir.rmdir()
        return video


class RecordingStepper:
    def __init__(
        self,
        world: Any,
        recorder: FrameRecorder,
        telemetry: Any,
        contact_audit: FilteredContactAudit,
    ) -> None:
        self.world = world
        self.recorder = recorder
        self.telemetry = telemetry
        self.contact_audit = contact_audit

    def step(self, *, render: bool = False) -> None:
        del render
        self.world.step(render=True)
        self.telemetry.sample()
        self.contact_audit.sample()
        self.recorder.observe()


def run() -> dict[str, Any]:
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import RigidPrim, SingleArticulation
    from radcounter.isaac.robot import (
        PhysxManipulationTelemetry,
        RealRobotAssetConfig,
        RidgebackFrankaController,
        add_real_robot_references,
        author_real_robot_task_scene,
        create_decontamination_activity_map,
        enable_real_robot_extensions,
    )

    ARGS.output.mkdir(parents=True, exist_ok=True)
    enable_real_robot_extensions()
    for _ in range(20):
        APP.update()
    context = omni.usd.get_context()
    scene_path = ROOT / "assets/environments/radcounter_vertical_slice.usda"
    if not context.open_stage(str(scene_path)):
        raise RuntimeError(f"failed to open {scene_path}")
    for _ in range(25):
        APP.update()
    stage = context.get_stage()
    config = RealRobotAssetConfig()
    asset_manifest = add_real_robot_references(stage, config=config)
    for _ in range(100):
        APP.update()
    activity_map = create_decontamination_activity_map(ARGS.output / "workbench_activity.npz")
    author_real_robot_task_scene(
        stage,
        activity_map,
        ROOT / "configs/decontamination/concrete_surface.synthetic.yaml",
        config=config,
    )
    surface_source = author_irregular_drum_surface_source(stage)
    simplify_scene(stage)
    drum_fixing = ground_fix_drum(stage)
    set_camera(stage)

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
    )
    articulation = world.scene.add(
        SingleArticulation(config.countermeasure_root, name="shield_video_ridgeback_franka")
    )
    shield_contact_view = world.scene.add(
        RigidPrim(
            prim_paths_expr=config.shield_path,
            name="shield_floor_contact_view",
            track_contact_forces=True,
            contact_filter_prim_paths_expr=[
                "/World/Environment/FloorCollision",
                "/World/HiddenContaminatedDrum/Drum",
            ],
            max_contact_count=64,
        )
    )
    world.reset()
    for _ in range(90):
        world.step(render=True)

    recorder = FrameRecorder(ARGS.output, stride=ARGS.capture_stride)
    telemetry = PhysxManipulationTelemetry(
        articulation,
        joint_names=RidgebackFrankaController.arm_joint_names,
        rigid_contact_view=shield_contact_view,
        physics_dt_s=1.0 / 60.0,
        contact_threshold_n=0.5,
    )
    contact_audit = FilteredContactAudit(shield_contact_view, physics_dt_s=1.0 / 60.0)
    stepper = RecordingStepper(world, recorder, telemetry, contact_audit)
    controller = RidgebackFrankaController(
        stage,
        stepper,
        config=config,
        articulation=articulation,
    )
    for _ in range(60):
        stepper.step(render=True)
    stow_report = controller.stow_arm()
    if not stow_report.success:
        raise RuntimeError(f"Franka could not reach its transport posture: {stow_report}")
    if not controller.move_base((-0.08, 0.0, 0.0)):
        raise RuntimeError("Ridgeback could not establish the shield work-zone approach")

    source_position = world_position(stage, "/World/HiddenContaminatedDrum")
    # Keep the 50 mm plate near the source while leaving collision margin for
    # the wider shield base. At this offset the plate face is about 0.30 m
    # from the contaminated surface and the base is about 0.19 m from the
    # cylindrical drum collider.
    destination = np.asarray(
        (float(source_position[0] - 0.65), float(source_position[1]), 0.0),
        dtype=np.float64,
    )
    source_surface_x_m = float(source_position[0] - 0.326)
    shield_source_face_x_m = float(destination[0] + 0.025)
    source_shield_clearance_m = source_surface_x_m - shield_source_face_x_m
    placement_route = (
        (-0.05, -1.25, 0.0),
        (float(destination[0]), -1.25, 0.0),
        (float(destination[0]), float(destination[1] - 0.95), math.pi / 2.0),
    )
    report = controller.execute_shield_pick_and_place(
        pickup_base_xy_yaw=(-0.05, -0.62, 0.0),
        placement_base_route_xy_yaw=placement_route,
        destination_root_position_m=destination,
        # Retract the plate over the Ridgeback before driving or turning; the
        # lift pose leaves the payload extended in front of the mobile base.
        transport_hand_offset_from_grasp_m=(-0.25, 0.0, 0.42),
    )
    for _ in range(120):
        stepper.step(render=True)
    if not report.success:
        raise RuntimeError(f"physical shield placement failed: {report}")
    contact_summary = contact_audit.summary()
    if not contact_summary["collision_free"]:
        raise RuntimeError(f"shield contacted the fixed drum: {contact_summary}")
    telemetry_summary = telemetry.summary()
    if telemetry_summary.samples <= 0:
        raise RuntimeError("PhysX telemetry did not collect any physics samples")
    if telemetry_summary.overall_peak_abs_joint_effort_nm <= 0.01:
        raise RuntimeError("PhysX did not report a nonzero Franka joint effort")
    if telemetry_summary.overall_peak_joint_reaction_force_n <= 0.1:
        raise RuntimeError("PhysX did not report a nonzero Franka joint reaction force")
    if telemetry_summary.peak_rigid_contact_force_n <= 0.5:
        raise RuntimeError("PhysX did not report shield-to-floor contact reaction")
    recorder.wait_for_frames(world)
    video = recorder.encode(ARGS.output, ARGS.duration)
    final_shield = world_position(stage, config.shield_path)
    result = {
        "passed": True,
        "video": str(video),
        "duration_s": ARGS.duration,
        "resolution": [1280, 720],
        "encoded_fps": 30,
        "captured_frames": recorder.frame_count,
        "physics_steps": recorder.physics_steps,
        "robot": "Clearpath Ridgeback + Franka Emika Panda",
        "robot_asset": asset_manifest["countermeasure_usd"],
        "motion": "PhysX articulation + Lula IK + finger joints + fixed grasp constraint",
        "shield_mass_kg": 2.2,
        "surface_source": surface_source,
        "drum_fixing": drum_fixing,
        "initial_stow": stow_report,
        "shield_destination_m": destination,
        "shield_final_position_m": final_shield,
        "source_shield_clearance_m": source_shield_clearance_m,
        "shield_motion": report,
        "shield_contact_audit": contact_summary,
        "physx_telemetry": telemetry_summary,
    }
    world.stop()
    return result


def main() -> int:
    result_path = ARGS.output / "result.json"
    result: dict[str, Any]
    try:
        result = run()
        print("SHIELD_PLACEMENT_VIDEO_RESULT " + json.dumps(jsonable(result)), flush=True)
        return 0
    except Exception as error:
        result = {
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        print("SHIELD_PLACEMENT_VIDEO_ERROR " + json.dumps(result), flush=True)
        return 1
    finally:
        ARGS.output.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(jsonable(result), indent=2) + "\n", encoding="utf-8")
        APP.close()


if __name__ == "__main__":
    raise SystemExit(main())
