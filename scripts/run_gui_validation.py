#!/usr/bin/env python3
"""Run every vertical-slice operation in one visible Isaac Sim session."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import sys
import time
import traceback
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "source/extensions/radcounter.isaac"
NATIVE = ROOT / "build/native/python"
for path in (ROOT, EXTENSION, NATIVE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
_radcounter_package = importlib.import_module("radcounter")
_extension_namespace = str(EXTENSION / "radcounter")
if _extension_namespace not in _radcounter_package.__path__:
    _radcounter_package.__path__.append(_extension_namespace)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-open", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--phase-hold-s", type=float, default=0.35)
    parser.add_argument("--frame-delay-s", type=float, default=0.0)
    parser.add_argument("--decon-duration-s", type=float, default=1.5)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "artifacts/gui-validation/latest.json",
    )
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "as_dict"):
        return _jsonable(value.as_dict())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    return str(value)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _complete(coroutine: Any) -> Any:
    """Complete a workflow coroutine that intentionally never suspends."""

    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    coroutine.close()
    raise RuntimeError("workflow coroutine unexpectedly yielded to the Kit event loop")


class _GuiStepper:
    def __init__(self, app: Any, frame_delay_s: float) -> None:
        self.app = app
        self.frame_delay_s = max(0.0, frame_delay_s)

    def step(self, *, render: bool = False) -> None:
        del render
        self.app.update()
        if self.frame_delay_s:
            time.sleep(self.frame_delay_s)


class _ValidationPanel:
    def __init__(self) -> None:
        import omni.ui as ui

        self._ui = ui
        self.phase = ui.SimpleStringModel("Preparing scene")
        self.progress = ui.SimpleStringModel("0 / 9 operations")
        self.result = ui.SimpleStringModel("No operation has completed yet")
        self.audit = ui.SimpleStringModel("Validation is running")
        self.window = ui.Window("RadCounterSim Full Validation", width=520, height=690)
        self.window.frame.set_build_fn(self._build)

    def _build(self) -> None:
        ui = self._ui
        with ui.VStack(spacing=10, height=0):
            ui.Label("RADCOUNTER / FULL GUI VALIDATION", style={"font_size": 18})
            ui.Label(
                "Measurement -> decon -> shield -> relocation -> disposal -> verification",
                word_wrap=True,
            )
            ui.Separator(height=4)
            ui.Label("CURRENT PHASE", style={"font_size": 11, "color": 0xFF6CB6FF})
            ui.Label("", model=self.phase, word_wrap=True, height=48)
            ui.Label("", model=self.progress, height=24)
            ui.Separator(height=4)
            ui.Label("LATEST RESULT", style={"font_size": 11, "color": 0xFFE3B341})
            ui.Label("", model=self.result, word_wrap=True, height=350)
            ui.Separator(height=4)
            ui.Label("AUDIT", style={"font_size": 11, "color": 0xFF6CB6FF})
            ui.Label("", model=self.audit, word_wrap=True, height=90)
            ui.Label(
                "Synthetic validation data only. Not calibrated for safety decisions.",
                word_wrap=True,
                style={"font_size": 11, "color": 0xFFDD7A6B},
            )

    def update(self, phase: str, completed: int, total: int, result: object = "") -> None:
        self.phase.set_value(phase)
        self.progress.set_value(f"{completed} / {total} operations")
        encoded = json.dumps(_jsonable(result), indent=2, sort_keys=True)
        self.result.set_value(encoded[-4000:])
        print(json.dumps({"phase": phase, "completed": completed, "total": total}), flush=True)

    def finish(self, success: bool, artifact: Path) -> None:
        if success:
            self.audit.set_value(
                "PASS: every requested GUI operation completed and final invariants passed.\n"
                f"Evidence: {artifact}"
            )
        else:
            self.audit.set_value(f"FAIL: inspect the latest result and {artifact}")


def _hold(app: Any, duration_s: float) -> None:
    deadline = time.monotonic() + max(0.0, duration_s)
    while time.monotonic() < deadline and app.is_running():
        app.update()


def _world_position(stage: Any, prim_path: str) -> np.ndarray:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"USD prim is unavailable: {prim_path}")
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    return np.asarray(transform.Transform(Gf.Vec3d()), dtype=np.float64)


def _surface_center(stage: Any, prim_path: str) -> np.ndarray:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get() or []
    if not points:
        raise RuntimeError(f"decontamination surface has no vertices: {prim_path}")
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    vertices = np.asarray(
        [transform.Transform(Gf.Vec3d(point)) for point in points], dtype=np.float64
    )
    return np.asarray(
        (
            float(np.mean(vertices[:, 0])),
            float(np.mean(vertices[:, 1])),
            float(np.max(vertices[:, 2])),
        )
    )


def _position_contact_tool(
    stage: Any,
    app: Any,
    surface_position_m: np.ndarray,
    *,
    standoff_m: float = 0.025,
    frames: int = 24,
) -> None:
    from pxr import Gf, UsdGeom

    robot = stage.GetPrimAtPath("/World/CountermeasureRobot")
    tool = stage.GetPrimAtPath("/World/CountermeasureRobot/DeconContactTool")
    robot_world = UsdGeom.XformCache().GetLocalToWorldTransform(robot)
    target_world = np.asarray(surface_position_m, dtype=np.float64).copy()
    target_world[2] += standoff_m
    target_local = np.asarray(
        robot_world.GetInverse().Transform(Gf.Vec3d(*map(float, target_world))),
        dtype=np.float64,
    )
    translate = tool.GetAttribute("xformOp:translate")
    start_local = np.asarray(translate.Get(), dtype=np.float64)
    for fraction in np.linspace(0.0, 1.0, max(frames, 2)):
        position = (1.0 - fraction) * start_local + fraction * target_local
        translate.Set(Gf.Vec3d(*map(float, position)))
        app.update()

    # The parent base can continue settling while the child tool is animated.
    # Resolve the exact local contact pose against the final parent transform.
    final_robot_world = UsdGeom.XformCache().GetLocalToWorldTransform(robot)
    final_target_local = final_robot_world.GetInverse().Transform(
        Gf.Vec3d(*map(float, target_world))
    )
    translate.Set(Gf.Vec3d(*map(float, final_target_local)))


def _probe_decon_contacts(decontaminator: Any) -> dict[str, Any]:
    import carb

    samples, center, axis = decontaminator._tool_samples()
    hits = []
    for origin in samples:
        hit = decontaminator._query.raycast_closest(
            carb.Float3(*map(float, origin)),
            carb.Float3(*map(float, axis)),
            decontaminator.config.max_contact_distance_m,
            True,
        )
        normal = hit.get("normal", (0.0, 0.0, 0.0))
        hits.append(
            {
                "origin_m": origin.tolist(),
                "hit": bool(hit.get("hit", False)),
                "collision": str(hit.get("collision", "")),
                "distance_m": float(hit.get("distance", -1.0)),
                "face_index": int(hit.get("faceIndex", -1)),
                "normal": [float(normal[index]) for index in range(3)],
            }
        )
    return {"center_m": center.tolist(), "axis": axis.tolist(), "hits": hits}


def _path_blockers(
    probe: Any,
    start_m: np.ndarray,
    target_m: np.ndarray,
    *,
    excluded_paths: tuple[str, ...] = (),
) -> list[str]:
    start = np.asarray(start_m, dtype=np.float64)
    target = np.asarray(target_m, dtype=np.float64)
    distance = float(np.linalg.norm(target[:2] - start[:2]))
    samples = np.linspace(start, target, max(2, int(np.ceil(distance / 0.12)) + 1))
    samples[:, 2] = np.maximum(samples[:, 2], 0.32)
    ignored = (
        probe.config.countermeasure_robot_path,
        probe.config.measurement_robot_path,
        *excluded_paths,
    )
    clearance = probe.config.mobile_clearance_m
    blockers: list[str] = []
    for path, lower, upper in probe._collision_bounds():
        if any(probe._is_descendant(path, item) for item in ignored):
            continue
        if upper[2] <= 0.08:
            continue
        expanded_lower = lower - np.asarray((clearance, clearance, 0.05))
        expanded_upper = upper + np.asarray((clearance, clearance, 0.05))
        inside = np.all((samples >= expanded_lower) & (samples <= expanded_upper), axis=1)
        if bool(np.any(inside)):
            blockers.append(path)
    return blockers


def _configure_camera(stage: Any) -> None:
    try:
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Gf, UsdGeom

        path = "/World/ValidationCamera"
        camera = UsdGeom.Camera.Define(stage, path)
        camera.CreateFocalLengthAttr(24.0)
        view = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(10.5, -13.5, 10.0),
            Gf.Vec3d(0.0, 0.0, 0.5),
            Gf.Vec3d(0.0, 0.0, 1.0),
        )
        UsdGeom.Xformable(camera).MakeMatrixXform().Set(view.GetInverse())
        viewport = get_active_viewport()
        if viewport is not None:
            viewport.set_active_camera(path)
    except Exception as error:
        print(f"camera configuration warning: {type(error).__name__}: {error}", flush=True)


def _prepare_activity_copy(stage: Any, artifact_root: Path) -> tuple[Path, float]:
    source = ROOT / "assets/contaminated_objects/floor_activity.npz"
    destination = artifact_root / "runtime_floor_activity.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    surface = stage.GetPrimAtPath("/World/ContaminatedFloor")
    for name in ("rad:source:activityMapUri", "rad:decon:activityMapUri"):
        surface.GetAttribute(name).Set(str(destination))
    for name in ("rad:source:activityMapSha256", "rad:decon:activityMapSha256"):
        surface.GetAttribute(name).Set(digest)
    with np.load(destination, allow_pickle=False) as payload:
        activity = float(np.sum(payload["activity_bq"]))
    return destination, activity


def _measurement_rows(items: Iterable[Any]) -> list[dict[str, Any]]:
    return [_jsonable(item) for item in items]


def _run_validation(app: Any, args: argparse.Namespace, panel: _ValidationPanel) -> dict:
    import omni.timeline
    import omni.usd
    from radcounter.isaac.planning import IsaacActionCandidateGenerator
    from radcounter.isaac.robot import (
        ContactDrivenDecontaminator,
        DecontaminationConfig,
        IsaacPhysicsRobotController,
        PhysicsControllerConfig,
    )
    from radcounter.isaac.runtime import IsaacRadiationSimulation
    from radcounter.isaac.ui.dashboard import RadCounterDashboard
    from radcounter.isaac.workflow import IsaacWorkflowServices

    from radcounter.core.actions import ResourceState
    from radcounter.core.models import BeliefState, RevisionState
    from radcounter.core.models.actions import ActionStatus, ActionType
    from radcounter.core.planning import DeterministicFeasibilityChecker

    context = omni.usd.get_context()
    stage_path = ROOT / "assets/environments/radcounter_vertical_slice.usda"
    if not context.open_stage(str(stage_path)):
        raise RuntimeError(f"failed to open {stage_path}")
    for _ in range(24):
        app.update()
    stage = context.get_stage()
    _configure_camera(stage)

    dashboard = RadCounterDashboard("radcounter.gui.validation")
    artifact_root = args.artifact.parent
    activity_path, floor_activity_before = _prepare_activity_copy(stage, artifact_root)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(30):
        app.update()

    simulation = IsaacRadiationSimulation.from_config(
        stage,
        ROOT / "configs/scenarios/vertical_slice.runtime.json",
    )
    dashboard.simulation = simulation
    stepper = _GuiStepper(app, args.frame_delay_s)
    controller_config = PhysicsControllerConfig(
        grasp_distance_m=0.24,
        maximum_navigation_steps=3200,
        maximum_settle_steps=240,
    )
    countermeasure_controller = IsaacPhysicsRobotController(
        stage,
        "/World/CountermeasureRobot",
        "/World/CountermeasureRobot/DeconTool",
        stepper,
        config=controller_config,
    )
    measurement_controller = IsaacPhysicsRobotController(
        stage,
        "/World/MeasurementRobot",
        "/World/MeasurementRobot/Detector",
        stepper,
        config=controller_config,
    )
    generator = IsaacActionCandidateGenerator(
        stage, simulation, controller=countermeasure_controller
    )
    belief = BeliefState(
        ("/World/ContaminatedFloor",),
        np.asarray([2.4e7]),
        np.asarray([[4.0e12]]),
        RevisionState(),
    )
    resources = ResourceState(
        remaining_measurement_time_s=180.0,
        remaining_work_time_s=600.0,
        remaining_robot_runtime_s={
            "/World/MeasurementRobot": 300.0,
            "/World/CountermeasureRobot": 600.0,
        },
        remaining_shield_units={"lead": 3},
        remaining_decon_media=120.0,
        remaining_countermeasure_count=12,
    )
    decontaminator = ContactDrivenDecontaminator(
        stage,
        "/World/CountermeasureRobot/DeconContactTool",
        "/World/ContaminatedFloor",
        DecontaminationConfig(max_contact_distance_m=0.06),
    )

    def estimator(measurement: object, previous: BeliefState | None) -> BeliefState:
        del measurement
        return belief if previous is None else previous

    services = IsaacWorkflowServices(
        simulation,
        generator,
        estimator,
        controller=countermeasure_controller,
        measurement_controller=measurement_controller,
        decontaminators={"/World/ContaminatedFloor": decontaminator},
        resources=resources,
        artifact_path=ROOT / "artifacts/ui/latest_workflow.json",
    )
    checker = DeterministicFeasibilityChecker()
    _complete(services.initialize())

    total_operations = 8
    completed = 0
    records: list[dict[str, Any]] = []
    initial_robot_position = _world_position(stage, "/World/MeasurementRobot")
    initial_shield_position = _world_position(stage, "/World/LeadShield")
    initial_drum_position = _world_position(stage, "/World/HiddenContaminatedDrum")
    initial_obstacle_position = _world_position(stage, "/World/MovableObstacle")
    protected = "/World/DetectorStations/Protected"
    initial_protected_rate = simulation.expected_rates(detector_paths=[protected])[protected]

    def execute_candidate(
        label: str,
        candidate: Any,
        *,
        action_override: Any | None = None,
        update_residual: bool = True,
    ) -> None:
        nonlocal belief, completed
        report = checker.evaluate(candidate, resources)
        if not report.feasible:
            action = candidate.action if action_override is None else action_override
            parameters = action.parameters
            diagnostics: dict[str, Any] = {
                "reasons": report.reasons,
                "facts": _jsonable(candidate.feasibility),
                "parameters": _jsonable(parameters),
            }
            if "pickup_base_position_m" in parameters:
                robot_position = generator.probe.world_position(
                    generator.config.countermeasure_robot_path
                )
                pickup_position = np.asarray(parameters["pickup_base_position_m"])
                placement_position = np.asarray(parameters["placement_base_position_m"])
                excluded = (
                    () if action.target_prim_path is None else (str(action.target_prim_path),)
                )
                diagnostics.update(
                    {
                        "robot_position_m": robot_position.tolist(),
                        "to_pickup_blockers": _path_blockers(
                            generator.probe,
                            robot_position,
                            pickup_position,
                            excluded_paths=excluded,
                        ),
                        "to_placement_blockers": _path_blockers(
                            generator.probe,
                            pickup_position,
                            placement_position,
                            excluded_paths=excluded,
                        ),
                    }
                )
            raise RuntimeError(f"{label} is infeasible: {diagnostics}")
        action = candidate.action if action_override is None else action_override
        prediction = services.preview(action, belief)
        panel.update(f"Executing: {label}", completed, total_operations, prediction)
        result = _complete(services.execute(action))
        if result.status != ActionStatus.COMPLETED:
            raise RuntimeError(f"{label} failed: {result.public_details}")
        verification = _complete(services.verify(action))
        diagnosis = None
        if update_residual:
            diagnosis = services.diagnose(prediction, verification)
            belief = services.update(belief, diagnosis)
        completed += 1
        row = {
            "label": label,
            "action": _jsonable(action),
            "feasibility": _jsonable(report),
            "result": _jsonable(result.public_view()),
            "verification": _measurement_rows(verification),
            "diagnosis": _jsonable(diagnosis),
        }
        records.append(row)
        dashboard.set_workflow_view(services.workflow_view())
        panel.update(f"Completed: {label}", completed, total_operations, row)
        _hold(app, args.phase_hold_s)

    measurement_candidates = generator.generate_measurement_actions(belief)
    if len(measurement_candidates) < 4:
        raise RuntimeError("the scene did not generate all four measurement stations")
    panel.update("Moving measurement robot through four stations", completed, total_operations)
    for candidate in measurement_candidates:
        report = checker.evaluate(candidate, resources)
        if not report.feasible:
            raise RuntimeError(
                f"measurement move {candidate.action.action_id} is infeasible: {report.reasons}"
            )
        result = _complete(services.execute(candidate.action))
        if result.status != ActionStatus.COMPLETED:
            position, orientation = measurement_controller._base_pose()
            velocity = measurement_controller._numpy(
                measurement_controller._base.get_linear_velocities()
            )[0]
            raise RuntimeError(
                "measurement move failed: "
                f"details={result.public_details}, position_m={position.tolist()}, "
                f"orientation_wxyz={orientation.tolist()}, velocity_m_s={velocity.tolist()}, "
                f"target_m={candidate.action.target_pose_world[:3, 3].tolist()}"
            )
    initial_measurement = _complete(services.measure())
    belief = services.estimate(initial_measurement, None)
    completed += 1
    records.append(
        {
            "label": "four-station measurement",
            "station_actions": [item.action.action_id for item in measurement_candidates],
            "measurement": _measurement_rows(initial_measurement),
        }
    )
    panel.update("Completed: four-station measurement", completed, total_operations, records[-1])
    _hold(app, args.phase_hold_s)

    decon_candidate = generator.generate_decon_actions(belief)[0]
    decon_action = replace(
        decon_candidate.action,
        predicted_duration_s=args.decon_duration_s,
        parameters={
            **decon_candidate.action.parameters,
            "duration_s": args.decon_duration_s,
            "decon_media": args.decon_duration_s,
        },
    )
    precontact = countermeasure_controller.navigate_to(
        decon_action.parameters["pickup_base_position_m"]
    )
    if not precontact.success:
        raise RuntimeError(f"decontamination pre-positioning failed: {precontact.message}")
    _position_contact_tool(
        stage,
        app,
        _surface_center(stage, "/World/ContaminatedFloor"),
    )
    contact_probe = _probe_decon_contacts(decontaminator)
    print(json.dumps({"decon_contact_probe": contact_probe}), flush=True)
    # Pre-positioning above established and probed the physical contact pose.
    # Do not let the service navigate the parent base a second time.
    decon_parameters = dict(decon_action.parameters)
    decon_parameters.pop("pickup_base_position_m", None)
    decon_action = replace(decon_action, parameters=decon_parameters)
    execute_candidate(
        "contact decontamination",
        decon_candidate,
        action_override=decon_action,
    )

    generated_shields = generator.generate_shield_actions(belief)
    shield_candidates = [
        item for item in generated_shields if checker.evaluate(item, resources).feasible
    ]
    if not shield_candidates:
        diagnostics = [
            {
                "action_id": item.action.action_id,
                "target_m": item.action.target_pose_world[:3, 3].tolist(),
                "facts": _jsonable(item.feasibility),
                "reasons": checker.evaluate(item, resources).reasons,
                "to_pickup_blockers": _path_blockers(
                    generator.probe,
                    generator.probe.world_position(generator.config.countermeasure_robot_path),
                    np.asarray(item.action.parameters["pickup_base_position_m"]),
                    excluded_paths=(str(item.action.target_prim_path),),
                ),
                "to_placement_blockers": _path_blockers(
                    generator.probe,
                    np.asarray(item.action.parameters["pickup_base_position_m"]),
                    np.asarray(item.action.parameters["placement_base_position_m"]),
                    excluded_paths=(str(item.action.target_prim_path),),
                ),
            }
            for item in generated_shields
        ]
        raise RuntimeError(f"the scene generated no feasible shield placement: {diagnostics}")
    shield_candidate = shield_candidates[0]
    execute_candidate("shield placement", shield_candidate)

    current_shield = _world_position(stage, "/World/LeadShield")
    generated_corrections = generator.generate_shield_actions(belief)
    correction_candidates = [
        item for item in generated_corrections if checker.evaluate(item, resources).feasible
    ]
    if not correction_candidates:
        diagnostics = [
            {
                "action_id": item.action.action_id,
                "target_m": item.action.target_pose_world[:3, 3].tolist(),
                "facts": _jsonable(item.feasibility),
                "reasons": checker.evaluate(item, resources).reasons,
            }
            for item in generated_corrections
        ]
        raise RuntimeError(f"the scene generated no feasible shield correction: {diagnostics}")
    correction = max(
        correction_candidates,
        key=lambda item: float(
            np.linalg.norm(item.action.target_pose_world[:3, 3] - current_shield)
        ),
    )
    correction_action = replace(correction.action, action_type=ActionType.MOVE_SHIELD)
    execute_candidate("shield pose correction", correction, action_override=correction_action)

    object_candidates = generator.generate_move_remove_actions(belief)
    drum_move = next(
        item
        for item in object_candidates
        if item.action.action_type == ActionType.MOVE_OBJECT
        and item.action.target_prim_path == "/World/HiddenContaminatedDrum"
    )
    execute_candidate("contaminated object relocation", drum_move)

    object_candidates = generator.generate_move_remove_actions(belief)
    drum_remove = next(
        item
        for item in object_candidates
        if item.action.action_type == ActionType.REMOVE_OBJECT
        and item.action.target_prim_path == "/World/HiddenContaminatedDrum"
    )
    execute_candidate("contaminated object disposal", drum_remove)

    object_candidates = generator.generate_move_remove_actions(belief)
    obstacle_move = next(
        item
        for item in object_candidates
        if item.action.action_type == ActionType.MOVE_OBJECT
        and item.action.target_prim_path == "/World/MovableObstacle"
    )
    execute_candidate("non-contaminated obstacle relocation", obstacle_move)

    final_measurement = _complete(services.verify(obstacle_move.action))
    final_prediction = {
        "detector_path": protected,
        "predicted_rate_proxy_cps": initial_protected_rate,
    }
    final_diagnosis = services.diagnose(final_prediction, final_measurement)
    belief = services.update(belief, final_diagnosis)
    completed += 1
    records.append(
        {
            "label": "final verification and belief update",
            "measurement": _measurement_rows(final_measurement),
            "diagnosis": _jsonable(final_diagnosis),
            "belief_revision": _jsonable(belief.revision),
        }
    )
    dashboard.set_workflow_view(services.workflow_view())
    panel.update("Completed: final verification", completed, total_operations, records[-1])

    with np.load(activity_path, allow_pickle=False) as payload:
        floor_activity_after = float(np.sum(payload["activity_bq"]))
    final_robot_position = _world_position(stage, "/World/MeasurementRobot")
    final_shield_position = _world_position(stage, "/World/LeadShield")
    final_obstacle_position = _world_position(stage, "/World/MovableObstacle")
    drum = stage.GetPrimAtPath("/World/HiddenContaminatedDrum")
    drum_disposed = not drum or not drum.IsValid() or not drum.IsActive()
    if drum and drum.IsValid() and drum.IsActive():
        source_enabled = drum.GetAttribute("rad:source:enabled")
        disposed = drum.GetAttribute("rad:disposal:disposed")
        drum_disposed = (
            source_enabled
            and source_enabled.HasAuthoredValueOpinion()
            and not bool(source_enabled.Get())
            and disposed
            and disposed.HasAuthoredValueOpinion()
            and bool(disposed.Get())
        )
    final_protected_rate = simulation.expected_rates(detector_paths=[protected])[protected]

    invariants = {
        "all_operations_completed": completed == total_operations,
        "measurement_robot_moved": float(
            np.linalg.norm(final_robot_position - initial_robot_position)
        )
        > 0.25,
        "floor_activity_reduced": floor_activity_after < floor_activity_before,
        "shield_moved": float(np.linalg.norm(final_shield_position - initial_shield_position))
        > 0.25,
        "contaminated_drum_relocated_before_disposal": any(
            row["label"] == "contaminated object relocation" for row in records
        ),
        "contaminated_drum_disposed": drum_disposed,
        "obstacle_moved": float(np.linalg.norm(final_obstacle_position - initial_obstacle_position))
        > 0.25,
        "post_action_measurement_available": bool(final_measurement),
        "residual_available": final_diagnosis is not None,
        "native_transport_used": simulation.transport.statistics["native_trace_calls"] > 0,
    }
    failed = [name for name, passed in invariants.items() if not passed]
    if failed:
        raise RuntimeError(f"final GUI invariants failed: {failed}")

    return {
        "success": True,
        "stage": str(stage_path),
        "operations": records,
        "invariants": invariants,
        "initial": {
            "protected_rate_cps": initial_protected_rate,
            "floor_activity_bq": floor_activity_before,
            "measurement_robot_position_m": initial_robot_position,
            "shield_position_m": initial_shield_position,
            "drum_position_m": initial_drum_position,
            "obstacle_position_m": initial_obstacle_position,
        },
        "final": {
            "protected_rate_cps": final_protected_rate,
            "floor_activity_bq": floor_activity_after,
            "measurement_robot_position_m": final_robot_position,
            "shield_position_m": final_shield_position,
            "obstacle_position_m": final_obstacle_position,
            "drum_disposed": drum_disposed,
        },
        "transport_statistics": simulation.transport.statistics,
        "controller_trace": countermeasure_controller.trace,
        "measurement_controller_trace": measurement_controller.trace,
    }


def main() -> int:
    args = _arguments()
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": args.headless,
            "width": 1600,
            "height": 1000,
            "window_width": 1600,
            "window_height": 1000,
        }
    )
    panel = None
    payload: dict[str, Any]
    exit_code = 0
    try:
        panel = _ValidationPanel()
        payload = _run_validation(app, args, panel)
    except Exception as error:
        exit_code = 1
        payload = {
            "success": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        print(payload["traceback"], flush=True)
    _atomic_json(args.artifact, payload)
    print(
        json.dumps({"validation_artifact": str(args.artifact), **payload}, default=str), flush=True
    )
    if panel is not None:
        panel.finish(exit_code == 0, args.artifact)
        panel.update(
            "PASS - GUI remains open for inspection"
            if exit_code == 0
            else "FAIL - GUI remains open for inspection",
            8 if exit_code == 0 else 0,
            8,
            payload,
        )
    if args.keep_open:
        while app.is_running():
            app.update()
    app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
