#!/usr/bin/env python3
"""Launch Isaac Sim GUI and validate command plus autonomous robot motion."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(
    0,
    str(ROOT / "source/extensions/radcounter.isaac"),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument(
        "--result",
        type=Path,
        default=ROOT / ".cache/control-validation/result.json",
    )
    parser.add_argument(
        "--ready",
        type=Path,
        default=ROOT / ".cache/control-validation/ready",
    )
    return parser.parse_args()


ARGS = _arguments()

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": 1280,
        "height": 800,
        "renderer": "RaytracedLighting",
        "window_title": "RadCounterSim Multi-Input Robot Validation",
    }
)


import omni.usd
from isaacsim.core.api import World
from isaacsim.core.experimental.prims import RigidPrim
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics
from radcounter.isaac.robot.control_window import RobotControlWindow
from radcounter.isaac.robot.input_router import IsaacRobotInputRouter

from radcounter.core.robots.command_io import RobotCommandServer
from radcounter.core.robots.control import JointCommand, TwistCommand
from radcounter.core.robots.input import ControlSource

ROBOT_ID = "validation_robot"
ROBOT_PATH = "/World/ValidationRobot"


def _color(geometry, rgb: tuple[float, float, float]) -> None:
    geometry.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])


def _collision_shape(prim) -> None:
    UsdPhysics.CollisionAPI.Apply(prim)


def create_validation_robot(stage) -> str:
    root = UsdGeom.Xform.Define(stage, ROBOT_PATH)
    root.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.28))
    root_prim = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    UsdPhysics.MassAPI.Apply(root_prim).CreateMassAttr(24.0)
    root_prim.CreateAttribute("rad:robot:id", Sdf.ValueTypeNames.String).Set(ROBOT_ID)
    root_prim.CreateAttribute("rad:stream:focus", Sdf.ValueTypeNames.Bool).Set(True)

    body = UsdGeom.Cube.Define(stage, f"{ROBOT_PATH}/Body")
    body.CreateSizeAttr(1.0)
    body.AddScaleOp().Set(Gf.Vec3f(0.55, 0.34, 0.16))
    _color(body, (0.08, 0.32, 0.48))
    _collision_shape(body.GetPrim())

    deck = UsdGeom.Cube.Define(stage, f"{ROBOT_PATH}/DetectorDeck")
    deck.CreateSizeAttr(1.0)
    deck.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.25))
    deck.AddScaleOp().Set(Gf.Vec3f(0.28, 0.24, 0.08))
    _color(deck, (0.85, 0.52, 0.08))
    _collision_shape(deck.GetPrim())

    mast = UsdGeom.Cylinder.Define(stage, f"{ROBOT_PATH}/DetectorMast")
    mast.CreateAxisAttr("Z")
    mast.CreateRadiusAttr(0.055)
    mast.CreateHeightAttr(0.52)
    mast.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.54))
    _color(mast, (0.85, 0.87, 0.82))
    _collision_shape(mast.GetPrim())

    detector = UsdGeom.Sphere.Define(stage, f"{ROBOT_PATH}/Detector")
    detector.CreateRadiusAttr(0.13)
    detector.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.83))
    _color(detector, (0.91, 0.76, 0.09))
    _collision_shape(detector.GetPrim())

    for side, y in (("Left", 0.37), ("Right", -0.37)):
        for end, x in (("Front", 0.34), ("Rear", -0.34)):
            wheel = UsdGeom.Cylinder.Define(
                stage,
                f"{ROBOT_PATH}/{side}{end}Wheel",
            )
            wheel.CreateAxisAttr("Y")
            wheel.CreateRadiusAttr(0.14)
            wheel.CreateHeightAttr(0.07)
            wheel.AddTranslateOp().Set(Gf.Vec3d(x, y, -0.10))
            _color(wheel, (0.035, 0.04, 0.045))
            _collision_shape(wheel.GetPrim())
    return ROBOT_PATH


class RigidMobileRobotController:
    """Physical rigid-base plugin used to validate the common input router."""

    def __init__(self, stage, prim_path: str) -> None:
        self.stage = stage
        self.prim_path = prim_path
        self.prim = RigidPrim(prim_path)
        self.prim.initialize_cpp_data_view()

    def command_twist(self, command: TwistCommand) -> None:
        transform = UsdGeom.Xformable(
            self.stage.GetPrimAtPath(self.prim_path)
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        x_axis = transform.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0)).GetNormalized()
        y_axis = transform.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0)).GetNormalized()
        velocity = x_axis * command.linear_x_m_s + y_axis * command.linear_y_m_s
        self.prim.set_velocities(
            linear_velocities=np.asarray(
                [[velocity[0], velocity[1], 0.0]],
                dtype=np.float32,
            ),
            angular_velocities=np.asarray(
                [[0.0, 0.0, command.angular_z_rad_s]],
                dtype=np.float32,
            ),
        )

    def command_joint(self, command: JointCommand) -> None:
        raise RuntimeError(f"Validation base has no joints: {command.names}")

    def command_gripper(self, name: str, fraction_closed: float) -> None:
        raise RuntimeError(f"Validation base has no gripper: {name}")


def robot_position(stage) -> tuple[float, float, float]:
    transform = UsdGeom.Xformable(stage.GetPrimAtPath(ROBOT_PATH)).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    value = transform.ExtractTranslation()
    return float(value[0]), float(value[1]), float(value[2])


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.dist(a[:2], b[:2])


def main() -> int:
    ARGS.result.parent.mkdir(parents=True, exist_ok=True)
    ARGS.ready.parent.mkdir(parents=True, exist_ok=True)
    ARGS.result.unlink(missing_ok=True)
    ARGS.ready.unlink(missing_ok=True)
    result: dict[str, object] = {"passed": False}
    server = None
    router = None
    world = None
    try:
        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
        world.scene.add_default_ground_plane()
        stage = omni.usd.get_context().get_stage()
        create_validation_robot(stage)
        light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        light.CreateIntensityAttr(1_800.0)
        light.AddRotateXYZOp().Set(Gf.Vec3f(-38.0, 22.0, 24.0))
        world.reset()
        controller = RigidMobileRobotController(stage, ROBOT_PATH)
        server = RobotCommandServer(port=ARGS.port)
        server.start()
        router = IsaacRobotInputRouter(
            {ROBOT_ID: controller},
            command_server=server,
            linear_speed_m_s=0.8,
            angular_speed_rad_s=1.2,
        )
        router.attach_devices(ROBOT_ID, ROBOT_ID)

        elapsed_s = 0.0

        def autonomous_provider(_dt_s: float):
            if 8.0 <= elapsed_s < 12.5:
                return TwistCommand(
                    linear_x_m_s=0.42,
                    angular_z_rad_s=0.55,
                )
            return None

        router.register_auto_controller(ROBOT_ID, autonomous_provider)
        window = RobotControlWindow(router, ROBOT_ID, ARGS.port)
        set_camera_view(
            eye=np.asarray([4.2, 4.2, 3.1]),
            target=np.asarray([0.0, 0.0, 0.25]),
            camera_prim_path="/OmniverseKit_Persp",
        )
        start_pose = robot_position(stage)
        command_first = None
        command_last = None
        auto_first = None
        auto_last = None
        seen_sources: set[str] = set()
        start_wall = time.monotonic()
        next_frame = start_wall
        ARGS.ready.write_text(
            json.dumps({"port": ARGS.port, "robot_id": ROBOT_ID}) + "\n",
            encoding="utf-8",
        )
        print(
            f"CONTROL_VALIDATION_READY port={ARGS.port} robot={ROBOT_ID}",
            flush=True,
        )

        while simulation_app.is_running() and elapsed_s < ARGS.duration:
            now = time.monotonic()
            elapsed_s = now - start_wall
            router.update(1.0 / 60.0)
            world.step(render=True)
            pose = robot_position(stage)
            source = router.active_source(ROBOT_ID)
            if source is not None:
                seen_sources.add(source.value)
            if source is ControlSource.COMMAND:
                command_first = command_first or pose
                command_last = pose
            if source is ControlSource.AUTO:
                auto_first = auto_first or pose
                auto_last = pose
            window.update(pose)
            if int(elapsed_s * 4) % 4 == 0:
                print(
                    f"CONTROL_TELEMETRY t={elapsed_s:.2f} "
                    f"source={source.value if source else 'none'} "
                    f"x={pose[0]:.3f} y={pose[1]:.3f}",
                    flush=True,
                )
            next_frame += 1.0 / 60.0
            sleep_s = next_frame - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)

        end_pose = robot_position(stage)
        command_motion = (
            distance(command_first, command_last)
            if command_first is not None and command_last is not None
            else 0.0
        )
        auto_motion = (
            distance(auto_first, auto_last)
            if auto_first is not None and auto_last is not None
            else 0.0
        )
        total_motion = distance(start_pose, end_pose)
        passed = (
            ControlSource.COMMAND.value in seen_sources
            and ControlSource.AUTO.value in seen_sources
            and command_motion >= 0.20
            and auto_motion >= 0.20
            and total_motion >= 0.35
        )
        result = {
            "passed": passed,
            "robot_id": ROBOT_ID,
            "seen_sources": sorted(seen_sources),
            "start_pose_m": start_pose,
            "end_pose_m": end_pose,
            "command_motion_m": command_motion,
            "auto_motion_m": auto_motion,
            "total_motion_m": total_motion,
            "duration_s": elapsed_s,
        }
        print("CONTROL_VALIDATION_RESULT " + json.dumps(result, sort_keys=True), flush=True)
        return 0 if passed else 2
    except Exception as exc:
        result = {
            "passed": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print("CONTROL_VALIDATION_ERROR " + json.dumps(result), flush=True)
        return 1
    finally:
        ARGS.result.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        ARGS.ready.unlink(missing_ok=True)
        if router is not None:
            router.detach_devices()
        if server is not None:
            server.close()
        if world is not None:
            world.stop()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
