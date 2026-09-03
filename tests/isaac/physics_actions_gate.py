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
            MeshWaterDecontaminationConfig,
            MeshWaterDecontaminator,
            PhysicsControllerConfig,
        )

        from radcounter.core.water_decontamination import (
            WaterDecontaminationState,
            WaterJetSpec,
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
        pending_treatment = decontaminator.tick(0.2, 1)
        assert pending_treatment.accepted_contacts > 0, pending_treatment
        assert pending_treatment.removed_activity_bq == 0.0
        assert pending_treatment.dwell_pending_triangle_indices
        treatment = decontaminator.tick(0.3, 2)
        decontaminator.flush()
        with np.load(activity_copy, allow_pickle=False) as payload:
            activity_after = float(np.sum(payload["activity_bq"]))
        assert treatment.accepted_contacts > 0, treatment
        assert treatment.removed_activity_bq > 0.0
        assert treatment.dwell_qualified_triangle_indices
        assert treatment.minimum_tool_dwell_s == 0.4
        assert abs(treatment.activity_balance_error_bq) < 1.0e-7
        assert activity_after < activity_before
        waste = stage.GetPrimAtPath("/World/DecontaminationWaste")
        assert waste.IsValid()
        assert (
            float(waste.GetAttribute("rad:source:activityBq").Get())
            == treatment.removed_activity_bq
        )
        water = MeshWaterDecontaminator(
            stage,
            "/World/CountermeasureRobot/DeconTool",
            "/World/ContaminatedFloor",
            WaterJetSpec(
                flow_rate_l_min=6.0,
                pressure_mpa=5.0,
                reference_pressure_mpa=5.0,
                spray_cone_angle_deg=20.0,
                minimum_footprint_radius_m=0.12,
                min_standoff_m=0.01,
                max_standoff_m=0.10,
                max_incidence_angle_deg=25.0,
                max_surface_speed_m_s=0.3,
                water_recovery_fraction=0.75,
                surface_water_retention_fraction=0.08,
                require_wastewater_collection=True,
            ),
            WaterDecontaminationState(1.0, 1.0),
            MeshWaterDecontaminationConfig(
                nozzle_origin_local_m=(-0.02736111, 0.14232143, 0.0),
            ),
        )
        water_result = water.tick(0.5, 3)
        water.flush()
        assert water_result.step.removed_activity_bq > 0.0, water_result
        assert water_result.step.contacted_cells
        assert abs(water_result.activity_balance_error_bq) < 1.0e-7
        assert abs(water_result.water_balance_error_l) < 1.0e-12
        assert water_result.treatment_data_status == "synthetic_validation_only"
        assert stage.GetPrimAtPath("/World/DecontaminationWastewater").IsValid()
        assert stage.GetPrimAtPath("/World/DecontaminationRunoff").IsValid()
        print(
            json.dumps(
                {
                    "base_displacement_m": float(after_position[0] - before_position[0]),
                    "grasp": grasp.success,
                    "accepted_contacts": treatment.accepted_contacts,
                    "pending_contacts_before_threshold": len(
                        pending_treatment.dwell_pending_triangle_indices
                    ),
                    "removed_activity_bq": treatment.removed_activity_bq,
                    "dry_activity_balance_error_bq": (treatment.activity_balance_error_bq),
                    "water_removed_activity_bq": water_result.step.removed_activity_bq,
                    "water_contacted_triangles": len(water_result.step.contacted_cells),
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
