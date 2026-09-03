import numpy as np

from radcounter.core.surface_decontamination import (
    DecontaminationTool,
    SurfaceSourceGrid,
    effective_contact_exposure_s,
    irregular_deposition_field,
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


def test_minimum_verified_dwell_delays_treatment_exposure() -> None:
    grid = _grid()
    tool = DecontaminationTool(
        length_m=0.9,
        width_m=0.9,
        rate_constant_s_inv=1.0,
        max_contact_distance_m=0.1,
        max_surface_speed_m_s=2.0,
        minimum_contact_dwell_s=0.4,
    )
    pending = grid.apply_tool(
        tool,
        tool_center_world_m=(0.5, 0.5, 0.02),
        tool_yaw_rad=0.0,
        surface_speed_m_s=0.0,
        dt_s=0.3,
    )
    qualified = grid.apply_tool(
        tool,
        tool_center_world_m=(0.5, 0.5, 0.02),
        tool_yaw_rad=0.0,
        surface_speed_m_s=0.0,
        dt_s=0.3,
    )
    assert pending.removed_activity_bq == 0.0
    assert pending.effective_dwell_s == 0.0
    assert qualified.removed_activity_bq > 0.0
    assert np.isclose(qualified.effective_dwell_s, 0.2)


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


def test_vertical_wall_grid_uses_surface_basis_for_contact() -> None:
    grid = SurfaceSourceGrid(
        cells_x=4,
        cells_y=4,
        size_x_m=2.0,
        size_y_m=2.0,
        center_world_m=(0.0, 1.0, 5.0),
        activity_bq_per_cell=100.0,
        surface_u_world=(1.0, 0.0, 0.0),
        surface_v_world=(0.0, 0.0, 1.0),
    )

    step = grid.apply_tool(
        _tool(),
        tool_center_world_m=(0.5, 0.98, 5.5),
        tool_yaw_rad=0.0,
        surface_speed_m_s=0.0,
        dt_s=1.0,
    )

    assert len(step.contacted_cells) == 4
    assert step.removed_activity_bq > 0.0
    assert np.allclose(grid.surface_normal_world, (0.0, -1.0, 0.0))


def test_canonical_irregular_field_retains_reference_mask_invariants() -> None:
    field = irregular_deposition_field().reshape(28, 48)
    active = field > 0.0

    assert int(np.count_nonzero(active)) == 497
    assert int(active.size) == 1_344
    assert field[active].std() > 60_000.0
    assert not active[13, 21]  # clean hole inside the main deposition

    seen: set[tuple[int, int]] = set()
    component_sizes: list[int] = []
    for row, column in np.argwhere(active):
        start = (int(row), int(column))
        if start in seen:
            continue
        pending = [start]
        seen.add(start)
        size = 0
        while pending:
            current_row, current_column = pending.pop()
            size += 1
            for row_delta in (-1, 0, 1):
                for column_delta in (-1, 0, 1):
                    if row_delta == column_delta == 0:
                        continue
                    neighbor = (
                        current_row + row_delta,
                        current_column + column_delta,
                    )
                    if (
                        0 <= neighbor[0] < active.shape[0]
                        and 0 <= neighbor[1] < active.shape[1]
                        and active[neighbor]
                        and neighbor not in seen
                    ):
                        seen.add(neighbor)
                        pending.append(neighbor)
        component_sizes.append(size)

    assert sorted(component_sizes, reverse=True) == [477, 13, 7]


def test_contact_exposure_is_one_speed_adjusted_tick_not_ray_count() -> None:
    exposure = effective_contact_exposure_s(0.5, 0.1, 0.3)
    assert np.isclose(exposure, 1.0 / 3.0)
    assert effective_contact_exposure_s(0.5, 0.31, 0.3) == 0.0
