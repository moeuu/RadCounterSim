import numpy as np
import pytest

from radcounter.core.scene import SurfaceActivityMap
from radcounter.core.surface_decontamination import SurfaceSourceGrid
from radcounter.core.treatment import WaterJetTreatment
from radcounter.core.water_decontamination import (
    TriangleSurfaceGeometry,
    TriangleWaterSurfaceDecontaminator,
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
        max_surface_speed_m_s=1.0,
        water_recovery_fraction=0.75,
        surface_water_retention_fraction=0.1,
    )
    values.update(overrides)
    return WaterJetSpec(**values)


def _treatment() -> WaterJetTreatment:
    return WaterJetTreatment(
        removal_coefficient_m2_per_l=0.8,
        washability_mean=1.0,
        washability_std=0.0,
        activity_capture_fraction=0.7,
        runoff_redeposition_fraction=0.5,
    )


def test_water_jet_reduces_surface_and_conserves_activity() -> None:
    grid = _grid()
    state = WaterDecontaminationState(
        supply_remaining_l=10.0,
        wastewater_capacity_l=10.0,
    )
    process = WaterSurfaceDecontaminator(grid, state, _treatment())
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
    process = WaterSurfaceDecontaminator(grid, state, _treatment())
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
    process = WaterSurfaceDecontaminator(grid, state, _treatment())
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
    process = WaterSurfaceDecontaminator(grid, state, _treatment())
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


def test_water_jet_treats_vertical_wall_surface() -> None:
    grid = SurfaceSourceGrid(
        cells_x=6,
        cells_y=4,
        size_x_m=1.2,
        size_y_m=0.8,
        center_world_m=(0.0, 1.0, 5.0),
        activity_bq_per_cell=1000.0,
        surface_u_world=(1.0, 0.0, 0.0),
        surface_v_world=(0.0, 0.0, 1.0),
    )
    state = WaterDecontaminationState(10.0, 10.0)
    process = WaterSurfaceDecontaminator(
        grid,
        state,
        _treatment(),
        runoff_direction_world_xy=(0.0, -1.0),
    )

    step = process.apply(
        _spec(),
        nozzle_world_m=(0.0, 0.5, 5.0),
        jet_direction_world=(0.0, 1.0, 0.0),
        surface_speed_m_s=0.0,
        dt_s=1.0,
    )

    assert step.blocked_reason is None
    assert step.removed_activity_bq > 0.0
    assert step.impact_world_m is not None
    assert np.allclose(step.impact_world_m, (0.0, 1.0, 5.0))


def _triangle_geometry() -> TriangleSurfaceGeometry:
    vertices = np.asarray(
        (
            (-0.8, -0.2, 0.0),
            (-0.4, -0.2, 0.0),
            (-0.6, 0.2, 0.0),
            (0.4, -0.2, 0.0),
            (0.8, -0.2, 0.0),
            (0.6, 0.2, 0.0),
        )
    )
    return TriangleSurfaceGeometry(
        vertices,
        np.asarray(((0, 1, 2), (3, 4, 5))),
        np.arange(2),
    )


def _triangle_activity() -> SurfaceActivityMap:
    return SurfaceActivityMap(
        triangle_indices=np.arange(2),
        activity_bq=np.asarray((1000.0, 1000.0)),
        cumulative_treatment_exposure=np.zeros(2),
        last_treated_step=np.full(2, -1),
        verified_contact_dwell_s=np.zeros(2),
    )


def test_triangle_water_process_uses_visible_faces_and_conserves_activity() -> None:
    activity = _triangle_activity()
    state = WaterDecontaminationState(10.0, 10.0)
    process = TriangleWaterSurfaceDecontaminator(
        activity,
        _triangle_geometry(),
        state,
        _treatment(),
        runoff_direction_world=(1.0, 0.0, 0.0),
    )
    step = process.apply(
        _spec(minimum_footprint_radius_m=0.12, spray_cone_angle_deg=5.0),
        nozzle_world_m=(-0.6, 0.0, 0.5),
        jet_direction_world=(0.0, 0.0, -1.0),
        surface_speed_m_s=0.0,
        dt_s=1.0,
        simulation_step=7,
    )
    assert step.contacted_cells == (0,)
    assert step.removed_activity_bq > 0.0
    assert activity.activity_bq[0] < 1000.0
    assert activity.activity_bq[1] == pytest.approx(1000.0)
    assert activity.last_treated_step[0] == 7
    assert abs(process.mass_balance_error_bq) < 1e-8
    assert abs(process.water_balance_error_l) < 1e-12


def test_triangle_water_process_does_not_treat_a_hidden_regular_proxy() -> None:
    process = TriangleWaterSurfaceDecontaminator(
        _triangle_activity(),
        _triangle_geometry(),
        WaterDecontaminationState(10.0, 10.0),
        _treatment(),
        runoff_direction_world=(1.0, 0.0, 0.0),
    )
    step = process.apply(
        _spec(),
        nozzle_world_m=(0.0, 0.0, 0.5),
        jet_direction_world=(0.0, 0.0, -1.0),
        surface_speed_m_s=0.0,
        dt_s=1.0,
        simulation_step=1,
    )
    assert step.blocked_reason == "nozzle ray misses visible activity mesh"
    assert np.all(process.activity_map.activity_bq == 1000.0)


def test_triangle_water_geometry_rejects_partial_activity_mapping() -> None:
    with pytest.raises(ValueError, match="every and only every visible face"):
        TriangleSurfaceGeometry(
            _triangle_geometry().vertices_world_m,
            _triangle_geometry().face_vertex_indices,
            np.asarray((0,)),
        )
