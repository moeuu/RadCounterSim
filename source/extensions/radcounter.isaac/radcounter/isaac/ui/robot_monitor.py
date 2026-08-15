"""Low-cost robot tracking, minimap, camera, and viewport operation overlays."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import omni.kit.app
import omni.ui as ui
import omni.usd
from omni.ui import color as cl
from omni.ui import scene as sc
from pxr import Gf, Usd, UsdGeom

from radcounter.core.robot_monitoring import (
    MONITOR_UPDATE_HZ,
    MonitorRobot,
    action_label,
    action_route_points,
    action_target_point,
    padded_square_bounds,
    phase_label,
    progress_from_remaining,
    project_minimap_point,
    robots_for_selection,
)


def _rgba(value: Sequence[float]) -> Any:
    return cl(float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _bind_label(
    model: ui.AbstractValueModel,
    subscriptions: list[Any],
    **kwargs: object,
) -> ui.Label:
    label = ui.Label(model.get_value_as_string(), **kwargs)

    def update(changed: ui.AbstractValueModel) -> None:
        label.text = changed.get_value_as_string()

    subscriptions.append(model.subscribe_value_changed_fn(update))
    return label


class _RobotViewportManipulator(sc.Manipulator):
    """Draw operator guidance over the viewport, including through-wall beacons."""

    def __init__(self) -> None:
        super().__init__()
        self.robots: tuple[dict[str, object], ...] = ()
        self.active_robot_id: str | None = None
        self.route_m: tuple[tuple[float, float, float], ...] = ()
        self.target_m: tuple[float, float, float] | None = None
        self.target_label = ""
        self.operation_label = ""
        self.camera_position_m: tuple[float, float, float] | None = None

    def update(
        self,
        *,
        robots: tuple[dict[str, object], ...],
        active_robot_id: str | None,
        route_m: tuple[tuple[float, float, float], ...],
        target_m: tuple[float, float, float] | None,
        target_label: str,
        operation_label: str,
        camera_position_m: tuple[float, float, float] | None,
    ) -> None:
        self.robots = robots
        self.active_robot_id = active_robot_id
        self.route_m = route_m
        self.target_m = target_m
        self.target_label = target_label
        self.operation_label = operation_label
        self.camera_position_m = camera_position_m
        self.invalidate()

    @staticmethod
    def _outline(lower: Sequence[float], upper: Sequence[float], color: Any) -> None:
        low = tuple(map(float, lower))
        high = tuple(map(float, upper))
        corners = (
            (low[0], low[1], low[2]),
            (high[0], low[1], low[2]),
            (high[0], high[1], low[2]),
            (low[0], high[1], low[2]),
            (low[0], low[1], high[2]),
            (high[0], low[1], high[2]),
            (high[0], high[1], high[2]),
            (low[0], high[1], high[2]),
        )
        for first, second in (
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ):
            sc.Line(corners[first], corners[second], color=color, thickness=2.2)

    def on_build(self) -> None:
        if len(self.route_m) >= 2:
            elevated = [(x, y, z + 0.18) for x, y, z in self.route_m]
            sc.Curve(
                elevated,
                curve_type=sc.Curve.CurveType.LINEAR,
                colors=[cl(0.20, 0.78, 1.00, 0.78)],
                thicknesses=[3.0],
            )
        if self.target_m is not None:
            target = tuple(self.target_m)
            sc.Points([target], colors=[cl(1.0, 0.35, 0.28, 0.92)], sizes=[16.0])
            with (
                sc.Transform(
                    look_at=sc.Transform.LookAt.CAMERA,
                    transform=sc.Matrix44.get_translation_matrix(
                        target[0], target[1], target[2] + 0.45
                    ),
                ),
                sc.Transform(scale_to=sc.Space.NDC),
            ):
                sc.Label(
                    self.target_label or "Work target",
                    alignment=ui.Alignment.CENTER_BOTTOM,
                    color=cl(1.0, 0.55, 0.42, 0.95),
                    size=15,
                )
        for item in self.robots:
            position = tuple(item["position_m"])
            lower = tuple(item["lower_m"])
            upper = tuple(item["upper_m"])
            color = _rgba(item["color_rgba"])
            active = item["robot_id"] == self.active_robot_id
            self._outline(lower, upper, color)
            marker_position = (position[0], position[1], upper[2] + (0.28 if active else 0.16))
            sc.Points(
                [marker_position],
                colors=[color],
                sizes=[17.0 if active else 10.0],
            )
            distance = ""
            if self.camera_position_m is not None:
                distance_m = math.dist(position, self.camera_position_m)
                distance = f" · {distance_m:.1f} m"
            suffix = f"\n{self.operation_label}" if active and self.operation_label else ""
            with (
                sc.Transform(
                    look_at=sc.Transform.LookAt.CAMERA,
                    transform=sc.Matrix44.get_translation_matrix(*marker_position),
                ),
                sc.Transform(scale_to=sc.Space.NDC),
            ):
                sc.Label(
                    f"{item['display_name']}{distance}{suffix}",
                    alignment=ui.Alignment.CENTER_BOTTOM,
                    color=color,
                    size=17 if active else 13,
                )


class RobotMonitorOverlay:
    """Own the operator monitor without creating additional rendered viewports."""

    CAMERA_PATH = "/World/RadCounterMonitoring/OperatorCamera"

    def __init__(self, ext_id: str) -> None:
        self.ext_id = ext_id
        self.robots: tuple[MonitorRobot, ...] = ()
        self.active_robot_id: str | None = None
        self.active_action = "Idle"
        self.active_phase = "idle"
        self.progress = 0.0
        self.coverage_fraction: float | None = None
        self.route_m: tuple[tuple[float, float, float], ...] = ()
        self.target_m: tuple[float, float, float] | None = None
        self.target_path: str | None = None
        self.camera_mode = "free"
        self._camera_eye: np.ndarray | None = None
        self._camera_target: np.ndarray | None = None
        self._camera_op: Any | None = None
        self._authored_camera_matrix: np.ndarray | None = None
        self._authored_camera_world_matrix: np.ndarray | None = None
        self._skip_manual_check = False
        self._initial_remaining_m: float | None = None
        self._auto_work_view = False
        self._work_view_triggered = False
        self._last_tick_s = 0.0
        self._map_lower_xy = (-5.0, -5.0)
        self._map_upper_xy = (5.0, 5.0)
        self._structure_rectangles: tuple[tuple[tuple[float, float, float], ...], ...] = ()
        self._contamination_points: tuple[tuple[float, float, float], ...] = ()
        self._overlay_error: str | None = None
        self._on_robot_list_changed: Callable[[], None] | None = None
        self._subscriptions: list[Any] = []
        self._viewport_window: Any | None = None
        self._viewport_api: Any | None = None
        self._overlay_frame: Any | None = None
        self._viewport_scene: Any | None = None
        self._viewport_manipulator: _RobotViewportManipulator | None = None
        self._minimap_scene: Any | None = None
        self._minimap_dynamic: Any | None = None
        self._active_model = ui.SimpleStringModel("NO ACTIVE ROBOT")
        self._position_model = ui.SimpleStringModel("Position --")
        self._action_model = ui.SimpleStringModel("Idle")
        self._camera_model = ui.SimpleStringModel("FREE CAMERA")
        self._progress_model = ui.SimpleFloatModel(0.0)
        self._progress_text_model = ui.SimpleStringModel("0%")
        self._update_subscription = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_update, name=f"{ext_id}.robot_monitor")
        )
        self._ensure_overlay()

    def set_robot_list_changed_callback(self, callback: Callable[[], None]) -> None:
        self._on_robot_list_changed = callback

    def _ensure_overlay(self) -> bool:
        if self._overlay_frame is not None:
            return True
        try:
            from omni.kit.viewport.utility import get_active_viewport_window

            self._viewport_window = get_active_viewport_window()
            if self._viewport_window is None:
                return False
            self._viewport_api = self._viewport_window.viewport_api
            self._overlay_frame = self._viewport_window.get_frame(f"{self.ext_id}.robot_monitor")
            with self._overlay_frame, ui.ZStack():
                self._viewport_scene = sc.SceneView()
                with self._viewport_scene.scene:
                    self._viewport_manipulator = _RobotViewportManipulator()
                self._viewport_api.add_scene_view(self._viewport_scene)
                with ui.VStack():
                    ui.Spacer(height=12)
                    with ui.HStack(height=214):
                        ui.Spacer(width=14)
                        self._build_status_bar()
                        ui.Spacer()
                        self._build_minimap()
                        ui.Spacer(width=14)
                    ui.Spacer()
            self._overlay_error = None
            return True
        except Exception as exc:
            self._overlay_error = f"{type(exc).__name__}: {exc}"
            self._overlay_frame = None
            return False

    def _build_status_bar(self) -> None:
        with ui.ZStack(width=670, height=82):
            ui.Rectangle(
                style={
                    "background_color": 0xE620252A,
                    "border_color": 0xFF3B5668,
                    "border_width": 1,
                    "border_radius": 5,
                }
            )
            with ui.VStack(spacing=3):
                ui.Spacer(height=8)
                with ui.HStack(height=20):
                    ui.Spacer(width=12)
                    _bind_label(
                        self._active_model,
                        self._subscriptions,
                        width=260,
                        style={"font_size": 15, "color": 0xFFFFC66D},
                    )
                    _bind_label(
                        self._position_model,
                        self._subscriptions,
                        width=230,
                        style={"font_size": 13, "color": 0xFFDCE5EB},
                    )
                    _bind_label(
                        self._camera_model,
                        self._subscriptions,
                        alignment=ui.Alignment.RIGHT,
                        style={"font_size": 11, "color": 0xFF73C9F5},
                    )
                    ui.Spacer(width=12)
                with ui.HStack(height=22):
                    ui.Spacer(width=12)
                    _bind_label(
                        self._action_model,
                        self._subscriptions,
                        width=560,
                        style={"font_size": 13, "color": 0xFFF1F4F6},
                    )
                    _bind_label(
                        self._progress_text_model,
                        self._subscriptions,
                        alignment=ui.Alignment.RIGHT,
                        style={"font_size": 13, "color": 0xFF73C9F5},
                    )
                    ui.Spacer(width=12)
                with ui.HStack(height=8):
                    ui.Spacer(width=12)
                    ui.ProgressBar(
                        self._progress_model,
                        height=5,
                        style={"color": 0xFF4CC7F2, "background_color": 0xFF303941},
                    )
                    ui.Spacer(width=12)

    def _build_minimap(self) -> None:
        with ui.ZStack(width=252, height=202):
            ui.Rectangle(
                style={
                    "background_color": 0xEA151A1E,
                    "border_color": 0xFF3B5668,
                    "border_width": 1,
                    "border_radius": 5,
                }
            )
            with ui.VStack(spacing=2):
                ui.Spacer(height=7)
                ui.Label(
                    "BUILDING OVERVIEW",
                    height=18,
                    alignment=ui.Alignment.CENTER,
                    style={"font_size": 11, "color": 0xFF9BAAB4},
                )
                self._minimap_scene = sc.SceneView(
                    height=168,
                    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
                )
                self._minimap_scene.cache_draw_buffer = False
        self._rebuild_minimap_scene()

    def configure(self, selection: Any) -> None:
        self.robots = robots_for_selection(selection)
        self.active_robot_id = self.robots[0].robot_id if self.robots else None
        self.active_action = "Idle"
        self.active_phase = "idle"
        self.progress = 0.0
        self.coverage_fraction = None
        self.route_m = ()
        self.target_m = None
        self.target_path = None
        self.stop_follow(manual=False)
        self._compute_minimap_geometry(selection)
        self._ensure_overlay()
        self._rebuild_minimap_scene()
        self._update_models(())
        if self._on_robot_list_changed is not None:
            self._on_robot_list_changed()

    def _stage_bounds(self, stage: Any) -> tuple[np.ndarray, np.ndarray] | None:
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        for path in ("/World/Environment", "/World/RemoteDeconFacility", "/World"):
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue
            aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            lower = np.asarray(aligned.GetMin(), dtype=np.float64)
            upper = np.asarray(aligned.GetMax(), dtype=np.float64)
            if np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)):
                return lower, upper
        return None

    def _compute_minimap_geometry(self, selection: Any) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        bounds = self._stage_bounds(stage)
        if bounds is None:
            anchors = [
                anchor.translation_m
                for anchor in selection.environment_entry.spawn_anchors.values()
            ]
            if anchors:
                points = np.asarray(anchors, dtype=np.float64)
                lower, upper = points.min(axis=0), points.max(axis=0)
            else:
                lower, upper = np.asarray((-5.0, -5.0, 0.0)), np.asarray((5.0, 5.0, 2.0))
        else:
            lower, upper = bounds
        self._map_lower_xy, self._map_upper_xy = padded_square_bounds(lower[:2], upper[:2])

        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        rectangles: list[tuple[float, tuple[tuple[float, float, float], ...]]] = []
        contamination: list[tuple[float, float, float]] = []
        for prim in stage.Traverse():
            role = prim.GetAttribute("rad:role")
            role_value = str(role.Get() or "") if role else ""
            decon = prim.GetAttribute("rad:decon:enabled")
            if role_value == "contaminated_surface" or (decon and bool(decon.Get())):
                aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                center = (np.asarray(aligned.GetMin()) + np.asarray(aligned.GetMax())) * 0.5
                points = None
                if prim.IsA(UsdGeom.Mesh):
                    points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
                if points:
                    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
                        prim
                    )
                    stride = max(1, int(math.ceil(len(points) / 96)))
                    for point in points[::stride]:
                        world = transform.Transform(Gf.Vec3d(*point))
                        contamination.append(
                            project_minimap_point(
                                world,
                                self._map_lower_xy,
                                self._map_upper_xy,
                            )
                        )
                elif np.all(np.isfinite(center)):
                    contamination.append(
                        project_minimap_point(
                            center,
                            self._map_lower_xy,
                            self._map_upper_xy,
                        )
                    )
            if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube)):
                continue
            path = str(prim.GetPath())
            if not (
                path.startswith("/World/Environment")
                or path.startswith("/World/RemoteDeconFacility")
            ):
                continue
            aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            item_lower = np.asarray(aligned.GetMin(), dtype=np.float64)
            item_upper = np.asarray(aligned.GetMax(), dtype=np.float64)
            extent = item_upper - item_lower
            if not np.all(np.isfinite(extent)) or extent[0] < 0.15 or extent[1] < 0.15:
                continue
            corners = tuple(
                project_minimap_point(point, self._map_lower_xy, self._map_upper_xy)
                for point in (
                    (item_lower[0], item_lower[1], 0.0),
                    (item_upper[0], item_lower[1], 0.0),
                    (item_upper[0], item_upper[1], 0.0),
                    (item_lower[0], item_upper[1], 0.0),
                    (item_lower[0], item_lower[1], 0.0),
                )
            )
            rectangles.append((float(extent[0] * extent[1]), corners))
        rectangles.sort(key=lambda item: item[0], reverse=True)
        self._structure_rectangles = tuple(item[1] for item in rectangles[:90])
        self._contamination_points = tuple(contamination[:128])

    def _rebuild_minimap_scene(self) -> None:
        if self._minimap_scene is None:
            return
        self._minimap_scene.scene.clear()
        with self._minimap_scene.scene:
            for value in (-0.45, 0.0, 0.45):
                grid_color = cl(0.22, 0.28, 0.32, 0.55)
                sc.Line(
                    (-0.9, value, 0.0),
                    (0.9, value, 0.0),
                    color=grid_color,
                    thickness=1.0,
                )
                sc.Line(
                    (value, -0.9, 0.0),
                    (value, 0.9, 0.0),
                    color=grid_color,
                    thickness=1.0,
                )
            for rectangle in self._structure_rectangles:
                sc.Curve(
                    rectangle,
                    curve_type=sc.Curve.CurveType.LINEAR,
                    colors=[cl(0.40, 0.48, 0.53, 0.48)],
                    thicknesses=[1.0],
                )
            if self._contamination_points:
                sc.Points(
                    self._contamination_points,
                    colors=[cl(1.0, 0.28, 0.22, 0.78)] * len(self._contamination_points),
                    sizes=[8.0],
                )
            self._minimap_dynamic = sc.Transform()
        self._update_minimap_dynamic(())

    def _robot_by_id(self, robot_id: str | None) -> MonitorRobot | None:
        return next((robot for robot in self.robots if robot.robot_id == robot_id), None)

    def _robot_for_action(self, action_robot_id: object) -> MonitorRobot | None:
        value = str(action_robot_id)
        return next(
            (
                robot
                for robot in self.robots
                if value == robot.robot_id
                or value == robot.prim_path
                or value.startswith(robot.prim_path.rstrip("/") + "/")
            ),
            None,
        )

    def begin_action(self, action: Any) -> None:
        robot = self._robot_for_action(getattr(action, "robot_id", ""))
        self.begin_operation(
            robot_id=None if robot is None else robot.robot_id,
            operation=action_label(getattr(action, "action_type", "operation")),
            phase="navigating",
            route_m=action_route_points(action),
            target_m=action_target_point(action),
            target_path=getattr(action, "target_prim_path", None),
            auto_work_view=str(getattr(action, "action_type", "")) == "decontaminate",
        )

    def begin_operation(
        self,
        *,
        robot_id: str | None,
        operation: str,
        phase: str,
        route_m: Sequence[Sequence[float]] = (),
        target_m: Sequence[float] | None = None,
        target_path: str | None = None,
        auto_work_view: bool = False,
    ) -> None:
        if self._robot_by_id(robot_id) is not None:
            self.active_robot_id = robot_id
        self.active_action = operation
        self.active_phase = phase
        self.progress = 0.0
        self.coverage_fraction = None
        self._initial_remaining_m = None
        self._auto_work_view = auto_work_view
        self._work_view_triggered = False
        self.route_m = tuple(tuple(float(item) for item in point[:3]) for point in route_m)
        self.target_m = None if target_m is None else tuple(float(item) for item in target_m[:3])
        self.target_path = target_path
        self.follow_robot(self.active_robot_id)
        self._update_models(self._robot_snapshots())

    def finish_action(
        self,
        *,
        success: bool,
        public_details: Mapping[str, object] | None = None,
    ) -> None:
        self.active_phase = "complete" if success else "failed"
        self.progress = 1.0 if success else self.progress
        details = public_details or {}
        motion = details.get("motion_audit")
        if isinstance(motion, Mapping):
            coverage = motion.get("coverage_fraction")
            if isinstance(coverage, (int, float)):
                self.coverage_fraction = float(coverage)
        self._update_models(self._robot_snapshots())

    def update_countermeasure_progress(self, event: Mapping[str, object]) -> None:
        phase = event.get("phase")
        if isinstance(phase, str):
            self.active_phase = phase
            if (
                phase == "decontaminating"
                and self._auto_work_view
                and not self._work_view_triggered
            ):
                self._work_view_triggered = True
                self.work_view(self.active_robot_id)
        progress = event.get("progress")
        if isinstance(progress, (int, float)) and math.isfinite(float(progress)):
            self.progress = max(0.0, min(1.0, float(progress)))
        coverage = event.get("coverage_fraction")
        if isinstance(coverage, (int, float)):
            self.coverage_fraction = max(0.0, min(1.0, float(coverage)))
        route = event.get("path_m")
        if isinstance(route, Sequence) and not isinstance(route, (str, bytes)):
            points = []
            for value in route:
                if isinstance(value, Sequence) and len(value) >= 3:
                    points.append(tuple(float(item) for item in value[:3]))
            if points:
                self.route_m = tuple(points)
        target = event.get("target_m")
        if (
            isinstance(target, Sequence)
            and not isinstance(target, (str, bytes))
            and len(target) >= 3
        ):
            self.target_m = tuple(float(item) for item in target[:3])

    def update_navigation_progress(
        self,
        *,
        position_m: tuple[float, float, float],
        target_xy_m: tuple[float, float],
        remaining_m: float,
    ) -> None:
        measurement = next((item for item in self.robots if item.robot_id == "measurement"), None)
        if measurement is not None:
            self.active_robot_id = measurement.robot_id
        self.active_phase = "navigating"
        if self._initial_remaining_m is None or remaining_m > self._initial_remaining_m:
            self._initial_remaining_m = max(remaining_m, 1.0e-9)
        self.progress = progress_from_remaining(remaining_m, self._initial_remaining_m)
        self.target_m = (float(target_xy_m[0]), float(target_xy_m[1]), float(position_m[2]))

    def _set_active_robot(self, robot_id: str | None) -> MonitorRobot | None:
        robot = self._robot_by_id(robot_id)
        if robot is not None:
            self.active_robot_id = robot.robot_id
            if self._on_robot_list_changed is not None:
                self._on_robot_list_changed()
        return robot

    def follow_robot(self, robot_id: str | None) -> None:
        if self._set_active_robot(robot_id) is None:
            return
        self._activate_camera("follow")

    def onboard_robot(self, robot_id: str | None) -> None:
        if self._set_active_robot(robot_id) is None:
            return
        self._activate_camera("onboard")

    def work_view(self, robot_id: str | None) -> None:
        if self._set_active_robot(robot_id) is None:
            return
        self._activate_camera("work")

    def overview(self) -> None:
        self._activate_camera("overview")

    def _activate_camera(self, mode: str) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        camera = UsdGeom.Camera.Define(stage, self.CAMERA_PATH)
        camera.CreateFocalLengthAttr(24.0 if mode != "onboard" else 18.0)
        xformable = UsdGeom.Xformable(camera)
        xformable.ClearXformOpOrder()
        self._camera_op = xformable.AddTransformOp()
        self.camera_mode = mode
        self._camera_eye = None
        self._camera_target = None
        self._authored_camera_matrix = None
        self._authored_camera_world_matrix = None
        self._skip_manual_check = True
        self._ensure_overlay()
        if self._viewport_api is not None:
            self._viewport_api.set_active_camera(self.CAMERA_PATH)
        self._update_models(self._robot_snapshots())

    def stop_follow(self, *, manual: bool = True) -> None:
        self.camera_mode = "free"
        self._camera_op = None
        self._authored_camera_matrix = None
        self._authored_camera_world_matrix = None
        if manual:
            self._auto_work_view = False
        self._update_models(self._robot_snapshots())

    @staticmethod
    def _matrix_array(matrix: Any) -> np.ndarray:
        return np.asarray([[float(matrix[row][column]) for column in range(4)] for row in range(4)])

    @staticmethod
    def _world_transform(stage: Any, path: str) -> Any | None:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return None
        return UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)

    def _desired_camera_pose(
        self, snapshots: tuple[dict[str, object], ...]
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if self.camera_mode == "overview":
            lower = np.asarray(self._map_lower_xy)
            upper = np.asarray(self._map_upper_xy)
            center = (lower + upper) * 0.5
            span = float(np.max(upper - lower))
            return (
                np.asarray((center[0], center[1] - 0.15 * span, max(8.0, 0.95 * span))),
                np.asarray((center[0], center[1], 0.0)),
            )
        active = next(
            (item for item in snapshots if item["robot_id"] == self.active_robot_id), None
        )
        if active is None:
            return None
        position = np.asarray(active["position_m"], dtype=np.float64)
        forward = np.asarray(active["forward_m"], dtype=np.float64)
        forward[2] = 0.0
        length = float(np.linalg.norm(forward))
        forward = np.asarray((1.0, 0.0, 0.0)) if length < 1.0e-6 else forward / length
        side = np.asarray((-forward[1], forward[0], 0.0))
        if self.camera_mode == "onboard":
            eye = position + 0.38 * forward + np.asarray((0.0, 0.0, 0.82))
            return eye, eye + 5.0 * forward + np.asarray((0.0, 0.0, 0.05))
        if self.camera_mode == "work" and self.target_m is not None:
            target = np.asarray(self.target_m, dtype=np.float64)
            center = (position + target) * 0.5
            separation = max(2.2, float(np.linalg.norm(target[:2] - position[:2])))
            eye = center - 0.7 * separation * forward - 0.75 * separation * side
            eye[2] = max(position[2], target[2]) + max(2.2, 0.65 * separation)
            return eye, center + np.asarray((0.0, 0.0, 0.35))
        eye = position - 3.4 * forward - 1.45 * side + np.asarray((0.0, 0.0, 2.35))
        target = position + 0.75 * forward + np.asarray((0.0, 0.0, 0.65))
        return eye, target

    def _update_camera(
        self,
        snapshots: tuple[dict[str, object], ...],
        dt_s: float,
    ) -> None:
        if self.camera_mode == "free" or self._camera_op is None:
            return
        if self._viewport_api is None or str(self._viewport_api.camera_path) != self.CAMERA_PATH:
            self.stop_follow()
            return
        current_matrix = self._camera_op.Get()
        stage = omni.usd.get_context().get_stage()
        current_world = None if stage is None else self._world_transform(stage, self.CAMERA_PATH)
        if (
            not self._skip_manual_check
            and self._authored_camera_matrix is not None
            and current_matrix is not None
            and float(
                np.max(np.abs(self._matrix_array(current_matrix) - self._authored_camera_matrix))
            )
            > 1.0e-3
        ):
            self.stop_follow()
            return
        if (
            not self._skip_manual_check
            and self._authored_camera_world_matrix is not None
            and current_world is not None
            and float(
                np.max(
                    np.abs(self._matrix_array(current_world) - self._authored_camera_world_matrix)
                )
            )
            > 1.0e-3
        ):
            self.stop_follow()
            return
        desired = self._desired_camera_pose(snapshots)
        if desired is None:
            return
        desired_eye, desired_target = desired
        if self._camera_eye is None or self._camera_target is None:
            self._camera_eye = desired_eye.copy()
            self._camera_target = desired_target.copy()
        else:
            blend = 1.0 - math.exp(-4.5 * max(dt_s, 1.0 / MONITOR_UPDATE_HZ))
            self._camera_eye += blend * (desired_eye - self._camera_eye)
            self._camera_target += blend * (desired_target - self._camera_target)
        view = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*self._camera_eye),
            Gf.Vec3d(*self._camera_target),
            Gf.Vec3d(0.0, 0.0, 1.0),
        )
        matrix = view.GetInverse()
        self._camera_op.Set(matrix)
        self._authored_camera_matrix = self._matrix_array(matrix)
        stage = omni.usd.get_context().get_stage()
        authored_world = None if stage is None else self._world_transform(stage, self.CAMERA_PATH)
        self._authored_camera_world_matrix = (
            None if authored_world is None else self._matrix_array(authored_world)
        )
        self._skip_manual_check = False

    def _robot_snapshots(self) -> tuple[dict[str, object], ...]:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return ()
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        snapshots = []
        for robot in self.robots:
            prim = stage.GetPrimAtPath(robot.prim_path)
            if not prim or not prim.IsValid():
                continue
            transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
            position = np.asarray(transform.ExtractTranslation(), dtype=np.float64)
            forward = np.asarray(transform.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0)), dtype=np.float64)
            aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            lower = np.asarray(aligned.GetMin(), dtype=np.float64)
            upper = np.asarray(aligned.GetMax(), dtype=np.float64)
            if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
                lower = position - np.asarray((0.35, 0.35, 0.05))
                upper = position + np.asarray((0.35, 0.35, 0.75))
            snapshots.append(
                {
                    "robot_id": robot.robot_id,
                    "display_name": robot.display_name,
                    "prim_path": robot.prim_path,
                    "color_rgba": robot.color_rgba,
                    "position_m": tuple(map(float, position)),
                    "forward_m": tuple(map(float, forward)),
                    "lower_m": tuple(map(float, lower)),
                    "upper_m": tuple(map(float, upper)),
                }
            )
        return tuple(snapshots)

    def _camera_position(self, stage: Any) -> tuple[float, float, float] | None:
        if self._viewport_api is None:
            return None
        path = str(self._viewport_api.camera_path)
        transform = self._world_transform(stage, path)
        if transform is None:
            return None
        return tuple(map(float, transform.ExtractTranslation()))

    def _update_models(self, snapshots: tuple[dict[str, object], ...]) -> None:
        active = next(
            (item for item in snapshots if item["robot_id"] == self.active_robot_id), None
        )
        robot = self._robot_by_id(self.active_robot_id)
        name = robot.display_name if robot is not None else "NO ACTIVE ROBOT"
        self._active_model.set_value(f"ACTIVE ROBOT: {name}")
        if active is None:
            self._position_model.set_value("Position --")
        else:
            x, y, z = active["position_m"]
            self._position_model.set_value(f"Position X {x:.2f} · Y {y:.2f} · Z {z:.2f} m")
        phase = phase_label(self.active_phase)
        coverage = (
            ""
            if self.coverage_fraction is None
            else f" · processed {100.0 * self.coverage_fraction:.0f}%"
        )
        self._action_model.set_value(f"{self.active_action} · {phase}{coverage}")
        camera_names = {
            "free": "FREE CAMERA",
            "follow": "FOLLOW",
            "work": "WORK VIEW",
            "onboard": "ONBOARD",
            "overview": "OVERVIEW",
        }
        self._camera_model.set_value(camera_names.get(self.camera_mode, self.camera_mode.upper()))
        self._progress_model.set_value(float(self.progress))
        self._progress_text_model.set_value(f"{100.0 * self.progress:.0f}%")

    def _update_minimap_dynamic(self, snapshots: tuple[dict[str, object], ...]) -> None:
        if self._minimap_dynamic is None:
            return
        self._minimap_dynamic.clear()
        with self._minimap_dynamic:
            if len(self.route_m) >= 2:
                route = tuple(
                    project_minimap_point(point, self._map_lower_xy, self._map_upper_xy)
                    for point in self.route_m
                )
                sc.Curve(
                    route,
                    curve_type=sc.Curve.CurveType.LINEAR,
                    colors=[cl(0.20, 0.78, 1.00, 0.88)],
                    thicknesses=[2.5],
                )
            if self.target_m is not None:
                target = project_minimap_point(
                    self.target_m, self._map_lower_xy, self._map_upper_xy
                )
                sc.Points([target], colors=[cl(1.0, 0.35, 0.28, 0.95)], sizes=[11.0])
            for item in snapshots:
                position = project_minimap_point(
                    item["position_m"], self._map_lower_xy, self._map_upper_xy
                )
                color = _rgba(item["color_rgba"])
                active = item["robot_id"] == self.active_robot_id
                sc.Points([position], colors=[color], sizes=[14.0 if active else 9.0])
                forward = np.asarray(item["forward_m"], dtype=np.float64)
                heading_world = np.asarray(item["position_m"], dtype=np.float64) + 0.9 * forward
                heading = project_minimap_point(
                    heading_world, self._map_lower_xy, self._map_upper_xy
                )
                sc.Line(position, heading, color=color, thickness=2.5)
                with sc.Transform(
                    transform=sc.Matrix44.get_translation_matrix(
                        position[0], position[1] + 0.07, 0.0
                    )
                ):
                    sc.Label(
                        str(item["display_name"]),
                        alignment=ui.Alignment.CENTER_BOTTOM,
                        color=color,
                        size=11 if active else 9,
                    )

    def _on_update(self, _event: object) -> None:
        now = time.perf_counter()
        interval = 1.0 / MONITOR_UPDATE_HZ
        if now - self._last_tick_s < interval:
            return
        dt_s = interval if self._last_tick_s == 0.0 else now - self._last_tick_s
        self._last_tick_s = now
        if not self._ensure_overlay():
            return
        snapshots = self._robot_snapshots()
        self._update_camera(snapshots, dt_s)
        self._update_models(snapshots)
        self._update_minimap_dynamic(snapshots)
        stage = omni.usd.get_context().get_stage()
        camera_position = None if stage is None else self._camera_position(stage)
        if self._viewport_manipulator is not None:
            if "measure" in self.active_action.casefold():
                target_label = "Detector measurement position"
            elif self.target_path:
                target_label = self.target_path.rsplit("/", 1)[-1]
            else:
                target_label = "Work target"
            self._viewport_manipulator.update(
                robots=snapshots,
                active_robot_id=self.active_robot_id,
                route_m=self.route_m,
                target_m=self.target_m,
                target_label=target_label,
                operation_label=f"{self.active_action} · {phase_label(self.active_phase)}",
                camera_position_m=camera_position,
            )

    def audit(self) -> dict[str, object]:
        return {
            "enabled": self._overlay_frame is not None,
            "update_rate_hz": MONITOR_UPDATE_HZ,
            "active_robot_id": self.active_robot_id,
            "active_action": self.active_action,
            "phase": self.active_phase,
            "progress": self.progress,
            "camera_mode": self.camera_mode,
            "robot_count": len(self.robots),
            "robots": [
                {
                    "robot_id": robot.robot_id,
                    "display_name": robot.display_name,
                    "prim_path": robot.prim_path,
                }
                for robot in self.robots
            ],
            "minimap_structure_count": len(self._structure_rectangles),
            "contamination_marker_count": len(self._contamination_points),
            "planned_route_point_count": len(self.route_m),
            "overlay_error": self._overlay_error,
            "single_rendered_viewport": True,
        }

    def destroy(self) -> None:
        self._update_subscription = None
        if self._viewport_scene is not None and self._viewport_api is not None:
            self._viewport_api.remove_scene_view(self._viewport_scene)
            self._viewport_scene.scene.clear()
        if self._minimap_scene is not None:
            self._minimap_scene.scene.clear()
        self._subscriptions.clear()
        self._viewport_manipulator = None
        self._viewport_scene = None
        self._minimap_scene = None
        self._overlay_frame = None
        self._viewport_window = None
        self._viewport_api = None
