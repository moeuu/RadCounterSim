#!/usr/bin/env python3
"""Run GUI-visible decontamination and shielding with real Isaac robot assets."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "source/extensions/radcounter.isaac"
NATIVE = ROOT / "build/native/python"
for search_path in (ROOT, EXTENSION, NATIVE):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

# The core package is a regular package while the Isaac code is supplied by a
# separate extension tree.  Extend the package search path before importing
# radcounter.isaac, matching the other standalone Isaac runners.
_radcounter_package = importlib.import_module("radcounter")
_extension_namespace = str(EXTENSION / "radcounter")
if _extension_namespace not in _radcounter_package.__path__:
    _radcounter_package.__path__.append(_extension_namespace)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--capture", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--keep-open", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "artifacts/real-robot/latest.json",
    )
    return parser.parse_args(argv)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class _Stepper:
    def __init__(self, app: Any, world: Any | None = None) -> None:
        self.app = app
        self.world = world

    def step(self, *, render: bool = False) -> None:
        if self.world is None:
            self.app.update()
        else:
            self.world.step(render=render)


class _Panel:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.phase = None
        self.detail = None
        self.window = None
        if not enabled:
            return
        import omni.ui as ui

        self._ui = ui
        self.phase = ui.SimpleStringModel("Loading real robot assets")
        self.detail = ui.SimpleStringModel("Ridgeback+Franka / Nova Carter")
        self.window = ui.Window("RadCounterSim Real Robot Validation", width=520, height=250)
        self.window.frame.set_build_fn(self._build)

    def _build(self) -> None:
        ui = self._ui
        with ui.VStack(spacing=10, height=0):
            ui.Label("REAL ROBOT COUNTERMEASURE", style={"font_size": 20})
            ui.Label("CURRENT PHASE", style={"font_size": 11, "color": 0xFF63C7FF})
            ui.Label("", model=self.phase, word_wrap=True, height=42)
            ui.Separator(height=3)
            ui.Label("", model=self.detail, word_wrap=True, height=100)
            ui.Label(
                "Motion is commanded through articulations and PhysX constraints.",
                style={"font_size": 11, "color": 0xFF62D2A2},
            )

    def update(self, phase: str, detail: str) -> None:
        print(json.dumps({"phase": phase, "detail": detail}), flush=True)
        if self.enabled:
            self.phase.set_value(phase)
            self.detail.set_value(detail)


def _create_activity_map(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        triangle_indices=np.arange(8, dtype=np.int64),
        activity_bq=np.full(8, 2.5e6, dtype=np.float64),
        cumulative_treatment_exposure=np.zeros(8, dtype=np.float64),
        last_treated_step=np.full(8, -1, dtype=np.int64),
    )


def _configure_camera(
    stage: Any,
    *,
    eye: tuple[float, float, float] = (5.2, -3.5, 3.2),
    target: tuple[float, float, float] = (1.45, -0.05, 0.55),
) -> None:
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Gf, UsdGeom, UsdLux

    camera_path = "/World/RealRobotValidationCamera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(24.0)
    view = Gf.Matrix4d().SetLookAt(
        Gf.Vec3d(*eye),
        Gf.Vec3d(*target),
        Gf.Vec3d(0.0, 0.0, 1.0),
    )
    camera_xform = UsdGeom.Xformable(camera)
    camera_xform.ClearXformOpOrder()
    camera_xform.MakeMatrixXform().Set(view.GetInverse())
    light = UsdLux.DomeLight.Define(stage, "/World/RealRobotValidationLight")
    light.CreateIntensityAttr(850.0)
    viewport = get_active_viewport()
    if viewport is not None:
        viewport.set_active_camera(camera_path)


def _capture(app: Any, output_directory: Path, name: str, enabled: bool) -> str | None:
    if not enabled:
        return None
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    viewport = get_active_viewport()
    if viewport is None:
        return None
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / f"{name}.png"
    capture_viewport_to_file(viewport, file_path=str(path), is_hdr=False)
    for _ in range(40):
        app.update()
        if path.is_file() and path.stat().st_size > 0:
            return str(path)
    return None


def _world_position(stage: Any, path: str) -> np.ndarray:
    from pxr import Gf, UsdGeom

    transform = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))
    return np.asarray(transform.Transform(Gf.Vec3d()), dtype=np.float64)


def _run(app: Any, args: argparse.Namespace, panel: _Panel) -> dict[str, Any]:
    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.robot.wheeled_robots.robots import WheeledRobot
    from radcounter.isaac.robot import (
        ContactDrivenDecontaminator,
        DecontaminationConfig,
        NovaCarterController,
        RealRobotAssetConfig,
        RidgebackFrankaController,
        add_real_robot_references,
        author_real_robot_task_scene,
        enable_real_robot_extensions,
    )
    from radcounter.isaac.runtime import IsaacRadiationSimulation

    enable_real_robot_extensions()
    for _ in range(20):
        app.update()
    context = omni.usd.get_context()
    scene_path = ROOT / "assets/environments/radcounter_vertical_slice.usda"
    if not context.open_stage(str(scene_path)):
        raise RuntimeError(f"failed to open {scene_path}")
    for _ in range(20):
        app.update()
    stage = context.get_stage()
    config = RealRobotAssetConfig()
    asset_manifest = add_real_robot_references(stage, config=config)
    panel.update("Loading official USD assets", "Ridgeback+Franka and Nova Carter")
    for _ in range(120):
        app.update()
    activity_path = args.artifact.parent / "workbench_activity.npz"
    _create_activity_map(activity_path)
    author_real_robot_task_scene(stage, activity_path, config=config)
    _configure_camera(stage)
    for _ in range(60):
        app.update()

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
    )
    franka_articulation = world.scene.add(
        SingleArticulation(config.countermeasure_root, name="radcounter_ridgeback_franka")
    )
    carter_articulation = world.scene.add(
        WheeledRobot(
            prim_path=config.measurement_articulation,
            name="radcounter_nova_carter",
            wheel_dof_names=list(NovaCarterController.wheel_names),
        )
    )
    world.reset()
    for _ in range(90):
        world.step(render=False)
    stepper = _Stepper(app, world)
    carter = NovaCarterController(
        stage,
        stepper,
        config=config,
        articulation=carter_articulation,
    )
    carter.set_initial_pose((-4.1, -2.5, 0.0))
    captures: dict[str, str | None] = {}
    captures["initial"] = _capture(
        app, args.artifact.parent / "frames", "00_initial", args.capture
    )

    simulation = IsaacRadiationSimulation.from_config(
        stage,
        ROOT / "configs/scenarios/vertical_slice.runtime.json",
    )
    protected_path = "/World/DetectorStations/Protected"
    hidden_source_path = "/World/HiddenContaminatedDrum"
    hidden_rate_before = simulation.expected_rates(
        detector_paths=[protected_path], source_paths=[hidden_source_path]
    )[protected_path]

    panel.update(
        "Measurement robot navigation",
        "Nova Carter is driving with left/right wheel velocity control.",
    )
    _configure_camera(
        stage,
        eye=(-5.2, -3.5, 2.2),
        target=(-3.45, -2.0, 0.35),
    )
    for _ in range(12):
        world.step(render=True)
    measurement_report = carter.navigate_to((-2.8, -1.55))
    captures["measurement"] = _capture(
        app, args.artifact.parent / "frames", "01_measurement", args.capture
    )
    if not measurement_report.success or measurement_report.displacement_m < 0.4:
        raise AssertionError(f"Nova Carter navigation failed: {measurement_report}")

    _configure_camera(stage)
    for _ in range(12):
        world.step(render=True)

    franka = RidgebackFrankaController(
        stage,
        stepper,
        config=config,
        articulation=franka_articulation,
    )

    panel.update(
        "Contact-driven decontamination",
        "Franka is lowering a physical pad and sweeping the contaminated mesh.",
    )
    if not franka.move_base((-0.08, 0.0, 0.0)):
        raise AssertionError(
            "Ridgeback could not establish workbench clearance: "
            f"target={franka.last_base_target.tolist()}, "
            f"actual={franka.last_base_positions.tolist()}"
        )
    decontaminator = ContactDrivenDecontaminator(
        stage,
        config.decon_tool_path,
        config.decon_surface_path,
        DecontaminationConfig(
            footprint_points_local_m=(
                (-0.06, -0.045, 0.0),
                (0.0, -0.045, 0.0),
                (0.06, -0.045, 0.0),
                (-0.06, 0.045, 0.0),
                (0.0, 0.045, 0.0),
                (0.06, 0.045, 0.0),
            ),
            treatment_axis_local=(0.0, 0.0, 1.0),
            max_contact_distance_m=0.045,
            max_surface_speed_m_s=0.35,
            rate_constant_s_inv=1.1,
            transfer_mode="transfer_to_waste",
        ),
    )
    decon_waypoints = (
        (0.54, -0.15, 0.790),
        (0.68, -0.15, 0.790),
        (0.82, -0.15, 0.790),
        (0.82, 0.15, 0.790),
        (0.68, 0.15, 0.790),
        (0.54, 0.15, 0.790),
    )
    decon_report = franka.execute_decontamination(
        decontaminator,
        decon_waypoints,
        dwell_frames=36,
    )
    captures["decontamination"] = _capture(
        app, args.artifact.parent / "frames", "02_decontamination", args.capture
    )
    if not decon_report.success:
        raise AssertionError(f"physical decontamination failed: {decon_report}")

    panel.update(
        "Shield pick-and-place",
        "Franka is opening, grasping, lifting, carrying, and releasing a 2.2 kg shield.",
    )
    source_position = _world_position(stage, hidden_source_path)
    detector_position = _world_position(stage, protected_path)
    destination = 0.5 * (source_position + detector_position)
    destination[2] = 0.0
    destination_base_x = float(destination[0])
    destination_base_y = float(destination[1] - 0.95)
    placement_route = (
        (-0.05, -1.25, 0.0),
        (destination_base_x, -1.25, 0.0),
        (destination_base_x, destination_base_y, math.pi / 2.0),
    )
    shield_report = franka.execute_shield_pick_and_place(
        pickup_base_xy_yaw=(-0.05, -0.62, 0.0),
        placement_base_route_xy_yaw=placement_route,
        destination_root_position_m=destination,
    )
    captures["shield_placed"] = _capture(
        app, args.artifact.parent / "frames", "03_shield_placed", args.capture
    )
    if not shield_report.success:
        raise AssertionError(f"physical shield placement failed: {shield_report}")

    simulation.synchronize()
    hidden_rate_after = simulation.expected_rates(
        detector_paths=[protected_path], source_paths=[hidden_source_path]
    )[protected_path]
    attenuation_ratio = hidden_rate_after / hidden_rate_before
    if attenuation_ratio >= 0.95:
        raise AssertionError(
            "placed shield did not attenuate the hidden source response: "
            f"ratio={attenuation_ratio:.6f}"
        )
    if franka.arm_joint_excursion_rad < 0.2:
        raise AssertionError("Franka arm joints did not execute a meaningful motion")

    panel.update(
        "PASS",
        "Both robots moved physically; contact decontamination and shield placement passed.",
    )
    result = {
        "success": True,
        "assets": asset_manifest,
        "robot_models": {
            "countermeasure": "Clearpath Ridgeback + Franka Panda",
            "measurement": "NVIDIA Nova Carter",
        },
        "dof_audit": {
            "countermeasure_dofs": list(franka.dof_names),
            "measurement_dofs": list(carter.robot.dof_names),
            "franka_arm_joint_excursion_rad": franka.arm_joint_excursion_rad,
        },
        "measurement_motion": measurement_report,
        "decontamination_motion": decon_report,
        "shield_motion": shield_report,
        "radiation_audit": {
            "hidden_source_rate_before_cps": hidden_rate_before,
            "hidden_source_rate_after_cps": hidden_rate_after,
            "attenuation_ratio": attenuation_ratio,
        },
        "motion_policy": {
            "teleport_during_operations": False,
            "base_control": "articulation joints / differential wheels",
            "arm_control": "Lula IK to 7 Franka joint targets",
            "grasp_control": "visible finger closure plus PhysX fixed joint",
            "decon_acceptance": "live PhysX ray contact only",
        },
        "captures": captures,
    }
    world.stop()
    return result


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": args.headless,
            "width": 1280,
            "height": 720,
            "window_width": 1440,
            "window_height": 900,
        }
    )
    panel = _Panel(not args.headless)
    try:
        result = _run(app, args, panel)
        _write_json(args.artifact, result)
        print("RADCOUNTER_REAL_ROBOT_PASS=" + json.dumps(_jsonable(result)), flush=True)
        if args.keep_open:
            while app.is_running():
                app.update()
        return 0
    except Exception as error:
        failure = {
            "success": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        _write_json(args.artifact, failure)
        print("RADCOUNTER_REAL_ROBOT_FAIL=" + json.dumps(failure), flush=True)
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
