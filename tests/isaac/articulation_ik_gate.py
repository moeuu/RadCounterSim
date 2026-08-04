"""Actual PhysX articulation and injected-IK control gate."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "source/extensions/radcounter.isaac"
for path in (ROOT, EXTENSION):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class _Stepper:
    def __init__(self, app) -> None:
        self.app = app

    def step(self, *, render: bool = False) -> None:
        del render
        self.app.update()


class _PlanarIk:
    def __init__(self, pivot_m: np.ndarray) -> None:
        self.pivot_m = pivot_m
        self.calls = 0

    def solve(self, position_m, orientation_xyzw):
        del orientation_xyzw
        self.calls += 1
        displacement = np.asarray(position_m, dtype=np.float64) - self.pivot_m
        angle = math.atan2(float(displacement[1]), float(displacement[0]))
        return np.asarray([angle], dtype=np.float64), True


def _build_articulation(stage) -> np.ndarray:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    world = UsdGeom.Xform.Define(stage, "/World")
    del world
    physics = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics.CreateGravityMagnitudeAttr(0.0)

    base = UsdGeom.Cube.Define(stage, "/World/TestRobot")
    base.CreateSizeAttr(0.2)
    base.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.0))
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    UsdPhysics.CollisionAPI.Apply(base.GetPrim())
    UsdPhysics.MassAPI.Apply(base.GetPrim()).CreateMassAttr(10.0)

    link = UsdGeom.Cube.Define(stage, "/World/TestRobot/EndEffector")
    link.CreateSizeAttr(0.16)
    link.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.0))
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    UsdPhysics.CollisionAPI.Apply(link.GetPrim())
    UsdPhysics.MassAPI.Apply(link.GetPrim()).CreateMassAttr(1.0)

    root = UsdPhysics.FixedJoint.Define(stage, "/World/TestRobot/RootJoint")
    root.CreateBody1Rel().SetTargets([Sdf.Path("/World/TestRobot")])
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/TestRobot/Joint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/TestRobot")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/TestRobot/EndEffector")])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-85.0)
    joint.CreateUpperLimitAttr(85.0)
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.1, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(-0.4, 0.0, 0.0))
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(1200.0)
    drive.CreateDampingAttr(120.0)
    drive.CreateMaxForceAttr(5000.0)
    drive.CreateTargetPositionAttr(0.0)
    return np.asarray((0.1, 0.0, 1.0), dtype=np.float64)


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import omni.timeline
        import omni.usd
        from radcounter.isaac.robot import IsaacPhysicsRobotController

        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        pivot = _build_articulation(stage)
        for _ in range(12):
            app.update()
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(20):
            app.update()

        ik = _PlanarIk(pivot)
        controller = IsaacPhysicsRobotController(
            stage,
            "/World/TestRobot",
            "/World/TestRobot/EndEffector",
            _Stepper(app),
            articulation_path="/World/TestRobot/RootJoint",
            ik_solver=ik,
        )
        target_angle = math.radians(32.0)
        target = pivot + np.asarray(
            (0.4 * math.cos(target_angle), 0.4 * math.sin(target_angle), 0.0)
        )
        assert controller.check_reachability(target)
        report = controller.move_end_effector(target, maximum_steps=500, tolerance_m=0.035)
        assert report.success, report
        positions = controller._articulation.get_joint_positions()
        if hasattr(positions, "numpy"):
            positions = positions.numpy()
        actual = float(np.asarray(positions).reshape(-1)[0])
        assert abs(actual - target_angle) <= 0.09, (actual, target_angle)
        print(
            json.dumps(
                {
                    "success": True,
                    "ik_calls": ik.calls,
                    "target_joint_rad": target_angle,
                    "actual_joint_rad": actual,
                    "physics_steps": report.steps,
                }
            ),
            flush=True,
        )
        return 0
    finally:
        try:
            import omni.timeline

            omni.timeline.get_timeline_interface().stop()
        except Exception:
            pass
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
