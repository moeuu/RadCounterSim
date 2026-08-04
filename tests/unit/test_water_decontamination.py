import numpy as np

from radcounter.core.surface_decontamination import SurfaceSourceGrid
from radcounter.core.water_decontamination import (
    WaterDecontaminationState,
    WaterJetSpec,
    WaterSurfaceDecontaminator,
)


def _grid() -> SurfaceSourceGrid:
    return SurfaceSourceGrid(
        cells_x=6,
        cells_y=4,
        size_x_m=1.2,
        size_y_m=0.8,
        center_world_m=(0.0, 0.0, 0.0),
        activity_bq_per_cell=1000.0,
    )


def _spec(**overrides) -> WaterJetSpec:
    values = dict(
        flow_rate_l_min=6.0,
        pressure_mpa=5.0,
        reference_pressure_mpa=5.0,
        spray_cone_angle_deg=30.0,
        minimum_footprint_radius_m=0.16,
        min_standoff_m=0.2,
        max_standoff_m=1.0,
        max_incidence_angle_deg=55.0,
        removal_coefficient_m2_per_l=0.8,
        max_surface_speed_m_s=1.0,
        activity_capture_fraction=0.7,
        water_recovery_fraction=0.75,
        runoff_redeposition_fraction=0.5,
        surface_water_retention_fraction=0.1,
    )
    values.update(overrides)
    return WaterJetSpec(**values)


def test_water_jet_reduces_surface_and_conserves_activity() -> None:
    grid = _grid()
    state = WaterDecontaminationState(
        supply_remaining_l=10.0,
        wastewater_capacity_l=10.0,
    )
    process = WaterSurfaceDecontaminator(grid, state)
    step = process.apply(
        _spec(),
        nozzle_world_m=(0.0, 0.0, 0.5),
        jet_direction_world=(0.0, 0.0, -1.0),
        surface_speed_m_s=0.2,
        dt_s=1.0,
    )
    assert step.removed_activity_bq > 0.0
    assert step.captured_activity_bq > 0.0
    assert step.redeposited_activity_bq > 0.0
    assert state.applied_water_l == 0.1
    assert abs(process.mass_balance_error_bq) < 1e-8
    assert abs(process.water_balance_error_l) < 1e-12
    assert grid.treated_fraction > 0.0


def test_collection_capacity_limits_water_application() -> None:
    grid = _grid()
    state = WaterDecontaminationState(
        supply_remaining_l=10.0,
        wastewater_capacity_l=0.015,
    )
    process = WaterSurfaceDecontaminator(grid, state)
    step = process.apply(
        _spec(water_recovery_fraction=0.75),
        nozzle_world_m=(0.0, 0.0, 0.5),
        jet_direction_world=(0.0, 0.0, -1.0),
        surface_speed_m_s=0.0,
        dt_s=1.0,
    )
    assert np.isclose(step.applied_water_l, 0.02)
    assert np.isclose(state.wastewater_volume_l, 0.015)


def test_bad_standoff_uses_no_water_and_changes_no_activity() -> None:
    grid = _grid()
    state = WaterDecontaminationState(10.0, 10.0)
    process = WaterSurfaceDecontaminator(grid, state)
    step = process.apply(
        _spec(),
        nozzle_world_m=(0.0, 0.0, 2.0),
        jet_direction_world=(0.0, 0.0, -1.0),
        surface_speed_m_s=0.0,
        dt_s=1.0,
    )
    assert step.blocked_reason is not None
    assert state.applied_water_l == 0.0
    assert grid.remaining_fraction == 1.0


def test_wetness_adds_blue_component_to_visualization() -> None:
    grid = _grid()
    state = WaterDecontaminationState(10.0, 10.0)
    process = WaterSurfaceDecontaminator(grid, state)
    before = process.visual_color_rgb()
    process.apply(
        _spec(),
        nozzle_world_m=(0.0, 0.0, 0.5),
        jet_direction_world=(0.0, 0.0, -1.0),
        surface_speed_m_s=0.0,
        dt_s=1.0,
    )
    after = process.visual_color_rgb()
    assert np.any(after[:, 2] > before[:, 2])
