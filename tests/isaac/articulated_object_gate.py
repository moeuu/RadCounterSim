"""Live gate for generic Franka object and obstacle IK approach poses."""

from __future__ import annotations

import importlib
import json
import shutil
import sys
import tempfile
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
        from radcounter.isaac.robot import (
            RealRobotAssetConfig,
            RidgebackFrankaController,
            add_real_robot_references,
            author_real_robot_task_scene,
            create_decontamination_activity_map,
            enable_real_robot_extensions,
        )

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
        author_real_robot_task_scene(stage, activity, config=config)
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
            approach = controller.move_hand(grasp + np.asarray((0.0, 0.0, 0.14)))
            root_after_approach = _position(stage, object_path)
            approach_displacement_m = float(np.linalg.norm(root_after_approach - root))
            audit[object_path] = {
                "root_m": root.tolist(),
                "grasp_m": grasp.tolist(),
                "approach": approach,
                "approach_displacement_m": approach_displacement_m,
            }
            assert approach.success, audit[object_path]
            assert approach_displacement_m <= 0.03, audit[object_path]
        print(json.dumps(audit, default=str), flush=True)
        return 0
    finally:
        app.close()
        shutil.rmtree(temporary_root)


if __name__ == "__main__":
    raise SystemExit(main())
