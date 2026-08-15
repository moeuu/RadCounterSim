from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from radcounter.core.robot_monitoring import (
    action_route_points,
    action_target_point,
    padded_square_bounds,
    progress_from_remaining,
    project_minimap_point,
    robots_for_selection,
)
from radcounter.core.system_profiles import resolve_system_selection


def test_monitor_lists_reference_and_decommissioning_robots() -> None:
    reference = robots_for_selection(resolve_system_selection(profile_id="fukushima-packbot"))
    assert [(robot.robot_id, robot.display_name, robot.prim_path) for robot in reference] == [
        ("packbot", "PackBot", "/World/Robots/PackBot"),
        ("elios3", "Elios 3", "/World/Robots/Elios3"),
    ]
    decommissioning = robots_for_selection(resolve_system_selection(profile_id="vertical-slice"))
    assert [robot.robot_id for robot in decommissioning] == ["countermeasure", "measurement"]
    assert len({robot.color_rgba for robot in reference}) == 2


def test_monitor_combines_planner_routes_and_target() -> None:
    action = SimpleNamespace(
        parameters={
            "pickup_base_route_m": [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
            "placement_base_route_m": [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]],
        },
        target_pose_world=np.asarray(
            [
                [1.0, 0.0, 0.0, 4.0],
                [0.0, 1.0, 0.0, 5.0],
                [0.0, 0.0, 1.0, 6.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    assert action_route_points(action) == (
        (0.0, 0.0, 0.0),
        (1.0, 2.0, 0.0),
        (3.0, 4.0, 0.0),
    )
    assert action_target_point(action) == (4.0, 5.0, 6.0)


def test_minimap_projection_uses_padded_square_and_inverts_world_y() -> None:
    lower, upper = padded_square_bounds((-2.0, -1.0), (2.0, 1.0), padding_fraction=0.0)
    assert lower == (-2.0, -2.0)
    assert upper == (2.0, 2.0)
    assert project_minimap_point((0.0, 0.0, 0.0), lower, upper) == pytest.approx((0.0, 0.0, 0.0))
    assert project_minimap_point((2.0, 2.0, 0.0), lower, upper) == pytest.approx((0.9, -0.9, 0.0))


def test_progress_from_remaining_is_bounded() -> None:
    assert progress_from_remaining(7.5, 10.0) == pytest.approx(0.25)
    assert progress_from_remaining(-1.0, 10.0) == 1.0
    assert progress_from_remaining(12.0, 10.0) == 0.0


def test_isaac_monitor_declares_all_operator_views_and_single_viewport_policy() -> None:
    root = Path(__file__).resolve().parents[2]
    monitor = (
        root / "source/extensions/radcounter.isaac/radcounter/isaac/ui/robot_monitor.py"
    ).read_text(encoding="utf-8")
    dashboard = (
        root / "source/extensions/radcounter.isaac/radcounter/isaac/ui/dashboard.py"
    ).read_text(encoding="utf-8")

    assert "ACTIVE ROBOT" in monitor
    assert "BUILDING OVERVIEW" in monitor
    assert 'self._activate_camera("follow")' in monitor
    assert 'self._activate_camera("onboard")' in monitor
    assert 'self._activate_camera("work")' in monitor
    assert 'phase == "decontaminating"' in monitor
    assert "and not self._work_view_triggered" in monitor
    assert '"single_rendered_viewport": True' in monitor
    assert "MONITOR_UPDATE_HZ = 12.0" in (root / "radcounter/core/robot_monitoring.py").read_text(
        encoding="utf-8"
    )
    assert '"View"' in dashboard
    assert "self._robot_monitor.follow_robot" in dashboard
    assert '"Onboard"' in dashboard
    assert "self._robot_monitor.onboard_robot" in dashboard
    assert "update_countermeasure_progress" in dashboard
    assert "with ui.ScrollingFrame(" in dashboard
    runner = (root / "scripts/run_gui_validation.py").read_text(encoding="utf-8")
    assert '"--robot-monitor-smoke-test"' in runner
    assert "def _exercise_robot_monitor" in runner
    assert 'manual_cancel["camera_mode"] != "free"' in runner
    assert "do not inject an unselected" in runner
