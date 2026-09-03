from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import radcounter
from radcounter.core.actions import ResourceState
from radcounter.core.models.actions import ActionStatus, ActionType, CountermeasureAction

ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "source/extensions/radcounter.isaac"
if str(EXTENSION) not in sys.path:
    sys.path.insert(0, str(EXTENSION))
extension_namespace = str(EXTENSION / "radcounter")
if extension_namespace not in radcounter.__path__:
    radcounter.__path__.append(extension_namespace)

from radcounter.isaac.workflow import IsaacWorkflowServices  # noqa: E402


class _Simulation:
    configuration = SimpleNamespace(duration_s=1.0)

    def refresh_scene_state(self) -> None:
        pass

    def synchronize(self) -> tuple[str, ...]:
        return ("/Surface",)


class _WaterState(SimpleNamespace):
    @property
    def wastewater_capacity_remaining_l(self) -> float:
        return self.wastewater_capacity_l - self.wastewater_volume_l


class _WaterTreatment:
    treatment_method = "water_jet"
    triangle_indices = (0, 1)
    minimum_tool_dwell_s = 0.0
    treatment_model = SimpleNamespace(
        model_id="water-test",
        status="synthetic_validation_only",
        numeric_sha256="a" * 64,
    )

    def __init__(self) -> None:
        self.state = _WaterState(
            supply_remaining_l=2.0,
            wastewater_capacity_l=2.0,
            wastewater_volume_l=0.0,
            applied_water_l=0.0,
            recovered_water_l=0.0,
            retained_surface_water_l=0.0,
            discharged_water_l=0.0,
            captured_activity_bq=0.0,
            redeposited_activity_bq=0.0,
            discharged_activity_bq=0.0,
        )
        self.process = SimpleNamespace(water_balance_error_l=0.0)

    def tick(self, dt_s: float, step: int) -> object:
        del dt_s, step
        self.state.applied_water_l += 0.4
        self.state.recovered_water_l += 0.3
        self.state.wastewater_volume_l += 0.3
        return SimpleNamespace(
            removed_activity_bq=20.0,
            accepted_contacts=2,
            dwell_qualified_triangle_indices=(0, 1),
            dwell_pending_triangle_indices=(),
            activity_balance_error_bq=0.0,
        )

    def flush(self) -> str:
        return "b" * 64


def _estimator(measurements: object, previous: object) -> object:
    del measurements
    return previous


def _action() -> CountermeasureAction:
    return CountermeasureAction(
        action_id="water-surface",
        action_type=ActionType.DECONTAMINATE,
        robot_id="robot",
        target_prim_path="/Surface",
        parameters={
            "surface_path": "/Surface",
            "treatment_method": "water_jet",
            "duration_s": 0.1,
            "clean_water_l": 1.0,
            "wastewater_l": 0.75,
        },
        predicted_duration_s=0.1,
    )


def test_workflow_dispatches_water_separately_and_consumes_actual_volume() -> None:
    water = _WaterTreatment()
    resources = ResourceState(
        remaining_work_time_s=10.0,
        remaining_robot_runtime_s={"robot": 10.0},
        remaining_clean_water_l=3.0,
        remaining_wastewater_capacity_l=3.0,
        remaining_countermeasure_count=3,
    )
    services = IsaacWorkflowServices(
        _Simulation(),
        SimpleNamespace(),
        _estimator,
        decontaminators={("/Surface", "water_jet"): water},
        resources=resources,
        simulation_time_s=lambda: 0.0,
        physics_dt_s=0.1,
    )
    result = asyncio.run(services.execute(_action()))
    assert result.status is ActionStatus.COMPLETED
    assert result.public_details["treatment_method"] == "water_jet"
    assert result.public_details["applied_water_l"] == 0.4
    assert result.public_details["recovered_water_l"] == 0.3
    assert resources.remaining_clean_water_l == 2.6
    assert resources.remaining_wastewater_capacity_l == 2.7


def test_workflow_never_falls_back_from_water_to_dry_implementation() -> None:
    dry = SimpleNamespace(treatment_method="dry_contact")
    services = IsaacWorkflowServices(
        _Simulation(),
        SimpleNamespace(),
        _estimator,
        decontaminators={"/Surface": dry},
        simulation_time_s=lambda: 0.0,
    )
    result = asyncio.run(services.execute(_action()))
    assert result.status is ActionStatus.FAILED
    assert "no water_jet decontaminator" in result.public_details["message"]
