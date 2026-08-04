import numpy as np

from radcounter.core.surface_decontamination import (
    DecontaminationTool,
    SurfaceSourceGrid,
)


def _grid() -> SurfaceSourceGrid:
    return SurfaceSourceGrid(
        cells_x=4,
        cells_y=4,
        size_x_m=2.0,
        size_y_m=2.0,
        center_world_m=(0.0, 0.0, 0.0),
        activity_bq_per_cell=100.0,
    )


def _tool() -> DecontaminationTool:
    return DecontaminationTool(
        length_m=0.9,
        width_m=0.9,
        rate_constant_s_inv=1.0,
        max_contact_distance_m=0.1,
        max_surface_speed_m_s=2.0,
    )


def test_contact_reduces_only_cells_inside_footprint() -> None:
    grid = _grid()
    step = grid.apply_tool(
        _tool(),
        tool_center_world_m=(0.5, 0.5, 0.02),
        tool_yaw_rad=0.0,
        surface_speed_m_s=0.0,
        dt_s=1.0,
    )
    assert step.removed_activity_bq > 0.0
    assert len(step.contacted_cells) == 4
    changed = np.flatnonzero(grid.activity_bq < grid.initial_activity_bq)
    np.testing.assert_array_equal(changed, step.contacted_cells)


def test_no_contact_preserves_activity() -> None:
    grid = _grid()
    step = grid.apply_tool(
        _tool(),
        tool_center_world_m=(0.0, 0.0, 0.5),
        tool_yaw_rad=0.0,
        surface_speed_m_s=0.0,
        dt_s=1.0,
    )
    assert step.contacted_cells == ()
    assert grid.remaining_fraction == 1.0


def test_activity_is_monotonic_and_visual_color_tracks_cleaning() -> None:
    grid = _grid()
    previous = grid.total_activity_bq
    for _ in range(5):
        grid.apply_tool(
            _tool(),
            tool_center_world_m=(0.0, 0.0, 0.02),
            tool_yaw_rad=0.0,
            surface_speed_m_s=0.2,
            dt_s=0.5,
        )
        assert grid.total_activity_bq <= previous
        previous = grid.total_activity_bq
    colors = grid.color_rgb()
    treated = grid.cumulative_exposure_s > 0.0
    assert colors[treated, 1].mean() > colors[~treated, 1].mean()
