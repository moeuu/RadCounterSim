"""Live gate for generic Franka object and obstacle IK approach poses."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "source/extensions/radcounter.isaac"
for path in (ROOT, EXTENSION):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
_radcounter_package = importlib.import_module("radcounter")
_extension_namespace = str(EXTENSION / "radcounter")
if _extension_namespace not in _radcounter_package.__path__:
    _radcounter_package.__path__.append(_extension_namespace)


class _Stepper:
    def __init__(self, world) -> None:
        self.world = world

    def step(self, *, render: bool = False) -> None:
        self.world.step(render=render)


def _position(stage, path: str) -> np.ndarray:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(path)
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    return np.asarray(matrix.Transform(Gf.Vec3d()), dtype=np.float64)


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    temporary_root = Path(tempfile.mkdtemp(prefix="radcounter-object-gate-"))
    try:
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.prims import SingleArticulation
        from radcounter.isaac.planning import (
            IsaacActionCandidateGenerator,
            SceneCandidateConfig,
        )
        from radcounter.isaac.robot import (
            RealRobotAssetConfig,
            RidgebackFrankaController,
            add_real_robot_references,
            author_real_robot_task_scene,
            create_decontamination_activity_map,
            enable_real_robot_extensions,
        )
        from radcounter.isaac.runtime.simulation import IsaacRadiationSimulation

        from radcounter.core.models import BeliefState, RevisionState
        from radcounter.core.models.actions import ActionType

        enable_real_robot_extensions()
        for _ in range(20):
            app.update()
        context = omni.usd.get_context()
        assert context.open_stage(
            str(ROOT / "assets/environments/radcounter_vertical_slice.usda")
        )
        for _ in range(20):
            app.update()
        stage = context.get_stage()
        config = replace(
            RealRobotAssetConfig(),
            decon_workbench_center_m=(14.39, 0.80, 1.15),
            shield_initial_position_m=(4.80, 2.80, 0.0),
        )
        add_real_robot_references(stage, config=config)
        for _ in range(120):
            app.update()
        activity = create_decontamination_activity_map(
            temporary_root / "activity.npz"
        )
        author_real_robot_task_scene(
            stage,
            activity,
            ROOT / "configs/decontamination/concrete_surface.synthetic.yaml",
            config=config,
        )
        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
        articulation = world.scene.add(
            SingleArticulation(config.countermeasure_root, name="object_gate_franka")
        )
        world.reset()
        for _ in range(90):
            world.step(render=False)
        controller = RidgebackFrankaController(
            stage,
            _Stepper(world),
            config=config,
            articulation=articulation,
        )
        audit = {}
        for object_path in (
            "/World/HiddenContaminatedDrum",
            "/World/MovableObstacle",
        ):
            target = stage.GetPrimAtPath(object_path)
            frame_name = str(target.GetAttribute("rad:manipulation:graspFrame").Get())
            grasp = _position(stage, f"{object_path}/{frame_name}")
            root = _position(stage, object_path)
            stow = controller.stow_arm()
            assert stow.success, stow
            assert controller.move_base((grasp[0] - 0.90, grasp[1], 0.0))
            approach = controller.move_hand(
                grasp + np.asarray((0.0, 0.0, 0.14)),
                tolerance_m=0.025,
            )
            root_after_approach = _position(stage, object_path)
            approach_displacement_m = float(np.linalg.norm(root_after_approach - root))
            audit[object_path] = {
                "root_m": root.tolist(),
                "grasp_m": grasp.tolist(),
                "approach": approach,
                "approach_displacement_m": approach_displacement_m,
            }
            assert approach.success, audit[object_path]
            assert approach_displacement_m <= 0.035, audit[object_path]

        # Exercise the exact physical relocation and shielded-disposal path
        # used by the paper scenario.  Candidate feasibility must remain
        # fail-closed; this gate does not bypass a rejected route or placement.
        world.reset()
        for _ in range(90):
            world.step(render=False)
        simulation = IsaacRadiationSimulation.from_config(
            stage,
            ROOT / "configs/scenarios/vertical_slice.runtime.json",
        )
        generator = IsaacActionCandidateGenerator(
            stage,
            simulation,
            controller=controller,
            config=SceneCandidateConfig(
                countermeasure_pose_path=config.panda_base_path,
                measurement_pose_path=config.measurement_articulation,
                end_effector_offset_m=(0.72, 0.0, 0.0),
                decon_end_effector_offset_m=(0.90, 0.0, 0.0),
                manipulator_workspace_m=0.95,
                mobile_clearance_m=0.55,
            ),
        )
        belief = BeliefState(
            (config.decon_surface_path,),
            np.asarray([2.4e7]),
            np.asarray([[4.0e12]]),
            RevisionState(),
        )

        def execute(candidate):
            parameters = candidate.action.parameters
            report = controller.execute_pick_and_place(
                str(parameters["object_path"]),
                parameters["pickup_base_position_m"],
                parameters["placement_base_position_m"],
                pickup_base_yaw_rad=float(parameters["pickup_base_yaw_rad"]),
                placement_base_yaw_rad=float(parameters["placement_base_yaw_rad"]),
                pickup_base_route_m=parameters["pickup_base_route_m"],
                placement_base_route_m=parameters["placement_base_route_m"],
                target_root_position_m=candidate.action.target_pose_world[:3, 3],
                placement_settle_tolerance_m=float(
                    parameters.get("placement_settle_tolerance_m", 0.15)
                ),
            )
            assert report.success, report
            return report

        drum_path = "/World/HiddenContaminatedDrum"
        move = next(
            candidate
            for candidate in generator.generate_move_remove_actions(belief)
            if candidate.action.action_type == ActionType.MOVE_OBJECT
            and candidate.action.target_prim_path == drum_path
        )
        assert all(vars(move.feasibility).values()), move
        move_report = execute(move)

        remove = next(
            candidate
            for candidate in generator.generate_move_remove_actions(belief)
            if candidate.action.action_type == ActionType.REMOVE_OBJECT
            and candidate.action.target_prim_path == drum_path
        )
        assert all(vars(remove.feasibility).values()), remove
        remove_report = execute(remove)
        disposal_report = controller.remove_to_disposal_zone(
            drum_path,
            str(remove.action.parameters["disposal_zone_path"]),
        )
        assert disposal_report.success, disposal_report
        drum = stage.GetPrimAtPath(drum_path)
        assert drum.GetAttribute("rad:source:enabled").Get() is True
        assert drum.GetAttribute("rad:source:contained").Get() is True
        assert (
            drum.GetAttribute("rad:source:containmentPrimPath").Get()
            == "/World/DisposalStorage"
        )
        obstacle_path = "/World/MovableObstacle"
        obstacle_move = next(
            candidate
            for candidate in generator.generate_move_remove_actions(belief)
            if candidate.action.action_type == ActionType.MOVE_OBJECT
            and candidate.action.target_prim_path == obstacle_path
        )
        assert all(vars(obstacle_move.feasibility).values()), obstacle_move
        obstacle_report = execute(obstacle_move)
        audit["physical_disposal"] = {
            "move": move_report,
            "remove": remove_report,
            "disposal": disposal_report,
            "obstacle_move": obstacle_report,
            "move_facts": vars(move.feasibility),
            "remove_facts": vars(remove.feasibility),
            "obstacle_move_facts": vars(obstacle_move.feasibility),
        }
        print(json.dumps(audit, default=str), flush=True)
    except BaseException:
        traceback.print_exc()
        shutil.rmtree(temporary_root, ignore_errors=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        app.close()
        shutil.rmtree(temporary_root, ignore_errors=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
