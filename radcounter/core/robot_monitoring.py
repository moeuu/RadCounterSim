"""Simulator-independent state helpers for operator robot monitoring."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from radcounter.core.system_profiles import ResolvedSystemSelection

MONITOR_UPDATE_HZ = 12.0

_ROBOT_COLORS = (
    (0.20, 0.78, 1.00, 0.92),
    (1.00, 0.72, 0.24, 0.90),
    (0.45, 0.90, 0.62, 0.90),
    (0.78, 0.56, 1.00, 0.90),
)

_REFERENCE_ROBOT_NAMES = {
    "irobot-packbot-fukushima": "PackBot",
    "flyability-elios3-rad": "Elios 3",
}

_ACTION_LABELS = {
    "measure": "測定地点へ移動",
    "decontaminate": "除染",
    "place_shield": "遮蔽体を配置",
    "move_shield": "遮蔽体を移動",
    "move_object": "物体を移動",
    "remove_object": "物体を撤去",
    "repair_action": "修復作業",
}

_PHASE_LABELS = {
    "idle": "待機中",
    "navigating": "移動中",
    "stowing_arm": "アーム収納中",
    "arm_stowed": "アーム収納完了",
    "approaching": "作業面へ接近中",
    "contact_confirmed": "接触確認済み",
    "decontaminating": "除染中",
    "grasping": "把持中",
    "releasing": "解放中",
    "returning_home": "開始位置へ帰還中",
    "complete": "完了",
    "failed": "失敗",
}


@dataclass(frozen=True, slots=True)
class MonitorRobot:
    robot_id: str
    display_name: str
    prim_path: str
    color_rgba: tuple[float, float, float, float]


def robots_for_selection(selection: ResolvedSystemSelection) -> tuple[MonitorRobot, ...]:
    """Return stable operator-facing robot identities for a catalog selection."""

    entries: list[tuple[str, str, str]] = []
    robot_set = selection.robot_set
    if robot_set.kind == "reference":
        entries.extend(
            (
                robot.robot_id,
                _REFERENCE_ROBOT_NAMES.get(
                    robot.reference_model_id,
                    robot.robot_id.replace("-", " ").title(),
                ),
                robot.prim_path,
            )
            for robot in robot_set.reference_robots
        )
    elif robot_set.kind == "fleet" and selection.robot_fleet is not None:
        entries.extend(
            (
                robot.id,
                robot.id.replace("-", " ").title(),
                str(robot.prim_path),
            )
            for robot in selection.robot_fleet.robots
            if robot.prim_path
        )
    elif robot_set.kind == "decommissioning":
        entries.extend(
            (
                ("countermeasure", "Ridgeback + Franka", "/World/CountermeasureRobot"),
                ("measurement", "Nova Carter", "/World/MeasurementRobot"),
            )
        )
    return tuple(
        MonitorRobot(robot_id, display_name, prim_path, _ROBOT_COLORS[index % len(_ROBOT_COLORS)])
        for index, (robot_id, display_name, prim_path) in enumerate(entries)
    )


def action_label(action_type: object) -> str:
    value = str(action_type)
    return _ACTION_LABELS.get(value, value.replace("_", " "))


def phase_label(phase: str) -> str:
    return _PHASE_LABELS.get(phase, phase.replace("_", " "))


def _point3(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None
    try:
        point = (
            float(value[0]),
            float(value[1]),
            float(value[2]) if len(value) >= 3 else 0.0,
        )
    except (TypeError, ValueError):
        return None
    return point if all(math.isfinite(item) for item in point) else None


def action_route_points(action: Any) -> tuple[tuple[float, float, float], ...]:
    """Combine planner route fields into one de-duplicated operator route."""

    parameters = getattr(action, "parameters", {})
    if not isinstance(parameters, Mapping):
        return ()
    route: list[tuple[float, float, float]] = []
    for key in ("base_route_m", "pickup_base_route_m", "placement_base_route_m"):
        values = parameters.get(key, ())
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for value in values:
            point = _point3(value)
            if point is not None and (not route or point != route[-1]):
                route.append(point)
    return tuple(route)


def action_target_point(action: Any) -> tuple[float, float, float] | None:
    pose = getattr(action, "target_pose_world", None)
    if pose is None:
        return None
    try:
        point = (float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3]))
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    return point if all(math.isfinite(item) for item in point) else None


def padded_square_bounds(
    lower_xy: Sequence[float],
    upper_xy: Sequence[float],
    *,
    padding_fraction: float = 0.08,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Expand a world XY range to a padded square suitable for a minimap."""

    lower = (float(lower_xy[0]), float(lower_xy[1]))
    upper = (float(upper_xy[0]), float(upper_xy[1]))
    if not all(math.isfinite(value) for value in (*lower, *upper)):
        raise ValueError("minimap bounds must be finite")
    width = max(upper[0] - lower[0], 1.0)
    height = max(upper[1] - lower[1], 1.0)
    span = max(width, height) * (1.0 + 2.0 * max(0.0, padding_fraction))
    center = ((lower[0] + upper[0]) * 0.5, (lower[1] + upper[1]) * 0.5)
    half = span * 0.5
    return (center[0] - half, center[1] - half), (center[0] + half, center[1] + half)


def project_minimap_point(
    point_m: Sequence[float],
    lower_xy: Sequence[float],
    upper_xy: Sequence[float],
) -> tuple[float, float, float]:
    """Project world XY into the SceneView's normalized top-down plane."""

    width = max(float(upper_xy[0]) - float(lower_xy[0]), 1.0e-9)
    height = max(float(upper_xy[1]) - float(lower_xy[1]), 1.0e-9)
    x = -0.9 + 1.8 * (float(point_m[0]) - float(lower_xy[0])) / width
    y = 0.9 - 1.8 * (float(point_m[1]) - float(lower_xy[1])) / height
    return (max(-0.9, min(0.9, x)), max(-0.9, min(0.9, y)), 0.0)


def progress_from_remaining(remaining_m: float, initial_remaining_m: float) -> float:
    if not math.isfinite(remaining_m) or not math.isfinite(initial_remaining_m):
        return 0.0
    if initial_remaining_m <= 1.0e-9:
        return 1.0
    return max(0.0, min(1.0, 1.0 - remaining_m / initial_remaining_m))
