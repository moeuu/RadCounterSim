"""Live PhysX gate for rigid motion, joint grasp, and contact decontamination."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "source/extensions/radcounter.isaac"))


class _Stepper:
    def __init__(self, app: object) -> None:
        self.app = app

    def step(self, *, render: bool = False) -> None:
        del render
        self.app.update()


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    temporary_directory = Path(tempfile.mkdtemp(prefix="radcounter-physx-gate-"))
    try:
        import omni.timeline
        import omni.usd
        from pxr import Gf
        from radcounter.isaac.robot import (
            ContactDrivenDecontaminator,
            DecontaminationConfig,
            IsaacPhysicsRobotController,
            PhysicsControllerConfig,
        )

        context = omni.usd.get_context()
        if not context.open_stage(str(ROOT / "assets/environments/radcounter_vertical_slice.usda")):
            raise AssertionError("vertical-slice USD did not open")
        for _ in range(8):
            app.update()
        stage = context.get_stage()
        activity_copy = temporary_directory / "floor_activity.npz"
        shutil.copy2(ROOT / "assets/contaminated_objects/floor_activity.npz", activity_copy)
        digest = hashlib.sha256(activity_copy.read_bytes()).hexdigest()
        surface = stage.GetPrimAtPath("/World/ContaminatedFloor")
        surface.GetAttribute("rad:source:activityMapUri").Set(str(activity_copy))
        surface.GetAttribute("rad:decon:activityMapUri").Set(str(activity_copy))
        surface.GetAttribute("rad:source:activityMapSha256").Set(digest)
        surface.GetAttribute("rad:decon:activityMapSha256").Set(digest)

        robot = stage.GetPrimAtPath("/World/CountermeasureRobot")
        robot.GetAttribute("xformOp:translate").Set(Gf.Vec3d(-0.6, 0.0, 0.0))
        shield = stage.GetPrimAtPath("/World/LeadShield")
        shield.GetAttribute("xformOp:translate").Set(Gf.Vec3d(-0.05, 0.0, 0.9))
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(20):
            app.update()

        controller = IsaacPhysicsRobotController(
            stage,
            "/World/CountermeasureRobot",
            "/World/CountermeasureRobot/DeconTool",
            _Stepper(app),
            config=PhysicsControllerConfig(grasp_distance_m=1.2, maximum_settle_steps=30),
        )
        before_position = controller._base_pose()[0].copy()
        controller._set_twist((0.15, 0.0, 0.0), (0.0, 0.0, 0.0))
        for _ in range(12):
            app.update()
        controller.stop()
        after_position = controller._base_pose()[0].copy()
        assert after_position[0] > before_position[0]

        grasp = controller.grasp("/World/LeadShield")
        assert grasp.success, grasp
        assert stage.GetPrimAtPath("/World/CountermeasureRobot/RadCounterGraspJoint").IsValid()
        release = controller.release()
        assert release.success, release
        assert not stage.GetPrimAtPath("/World/CountermeasureRobot/RadCounterGraspJoint").IsValid()

        robot.GetAttribute("xformOp:translate").Set(Gf.Vec3d(-0.6, 0.0, 0.0))
        stage.GetPrimAtPath("/World/CountermeasureRobot/DeconTool").GetAttribute(
            "xformOp:translate"
        ).Set(Gf.Vec3d(0.6, 0.0, 0.04))
        for _ in range(8):
            app.update()
        with np.load(activity_copy, allow_pickle=False) as payload:
            activity_before = float(np.sum(payload["activity_bq"]))
        decontaminator = ContactDrivenDecontaminator(
            stage,
            "/World/CountermeasureRobot/DeconTool",
            "/World/ContaminatedFloor",
            DecontaminationConfig(max_contact_distance_m=0.06),
        )
        treatment = decontaminator.tick(0.5, 1)
        decontaminator.flush()
        with np.load(activity_copy, allow_pickle=False) as payload:
            activity_after = float(np.sum(payload["activity_bq"]))
        assert treatment.accepted_contacts > 0, treatment
        assert treatment.removed_activity_bq > 0.0
        assert activity_after < activity_before
        waste = stage.GetPrimAtPath("/World/DecontaminationWaste")
        assert waste.IsValid()
        assert (
            float(waste.GetAttribute("rad:source:activityBq").Get())
            == treatment.removed_activity_bq
        )
        print(
            json.dumps(
                {
                    "base_displacement_m": float(after_position[0] - before_position[0]),
                    "grasp": grasp.success,
                    "accepted_contacts": treatment.accepted_contacts,
                    "removed_activity_bq": treatment.removed_activity_bq,
                }
            ),
            flush=True,
        )
        timeline.stop()
        return 0
    finally:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
