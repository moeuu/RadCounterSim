import ast
from pathlib import Path

import numpy as np

from radcounter.core.actions import ActionType, CountermeasureAction, ResourceState
from radcounter.core.models import BeliefState, RevisionState, TruthState
from radcounter.core.planning import (
    ActionCandidate,
    ActionCandidateGenerator,
    ActionMetrics,
    ActionOutcomeSamples,
    ClosedLoopResidualPlanner,
    DeterministicFeasibilityChecker,
    FeasibilityFacts,
    GreedyDoseReductionPlanner,
    NearestSourcePlanner,
    ObjectiveWeights,
    OpenLoopPlanner,
    OraclePlanner,
    OraclePlanningContext,
    PlanningContext,
    RandomPlanner,
    RiskAwareCountermeasurePlanner,
)


def _belief() -> BeliefState:
    return BeliefState(("source",), np.array([1.0]), np.eye(1), RevisionState(), {"robot_s": 100.0})


def _candidate(
    action_id: str,
    action_type: ActionType,
    *,
    task_dose: float,
    peak: float,
    distance: float,
    information: float = 0.0,
    feasible: bool = True,
) -> ActionCandidate:
    return ActionCandidate(
        CountermeasureAction(
            action_id,
            action_type,
            "robot",
            predicted_duration_s=2.0,
            resource_cost={"robot_s": 2.0},
        ),
        ActionMetrics(task_dose, peak, 1.0, 2.0, 1.0, 0.1, information, distance),
        FeasibilityFacts(collision_free=feasible),
    )


def _context() -> PlanningContext:
    candidates = (
        _candidate(
            "measure", ActionType.MEASURE, task_dose=9.0, peak=9.0, distance=3.0, information=5.0
        ),
        _candidate("shield", ActionType.PLACE_SHIELD, task_dose=2.0, peak=3.0, distance=8.0),
        _candidate("near", ActionType.DECONTAMINATE, task_dose=5.0, peak=5.0, distance=1.0),
    )
    return PlanningContext(
        _belief(), candidates, ResourceState({"robot_s": 10.0}), ObjectiveWeights()
    )


def test_candidate_generator_returns_all_required_groups() -> None:
    generator = ActionCandidateGenerator(_context().candidates)
    assert [item.action.action_id for item in generator.generate_measurement_actions()] == [
        "measure"
    ]
    assert [item.action.action_id for item in generator.generate_decon_actions()] == ["near"]
    assert [item.action.action_id for item in generator.generate_shield_actions()] == ["shield"]
    assert len(generator.generate_all()) == 3


def test_feasibility_reports_geometry_and_resource_reasons() -> None:
    candidate = _candidate(
        "blocked", ActionType.PLACE_SHIELD, task_dose=1.0, peak=1.0, distance=1.0, feasible=False
    )
    resources = ResourceState({"robot_s": 0.0}, remaining_countermeasure_count=0)
    report = DeterministicFeasibilityChecker().evaluate(candidate, resources)
    assert not report.feasible
    assert "collision" in report.reasons
    assert "generic_resource_shortage" in report.reasons
    assert "countermeasure_count_exhausted" in report.reasons


def test_objective_matches_specification_signs() -> None:
    metrics = ActionMetrics(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0)
    assert ObjectiveWeights().score(metrics) == 14.0


def test_baseline_planners_choose_their_defined_targets() -> None:
    context = _context()
    assert OpenLoopPlanner(("near", "shield")).plan(context).selected.action.action_id == "near"
    assert GreedyDoseReductionPlanner().plan(context).selected.action.action_id == "shield"
    assert NearestSourcePlanner().plan(context).selected.action.action_id == "near"
    first = RandomPlanner(np.random.default_rng(7)).plan(context)
    second = RandomPlanner(np.random.default_rng(7)).plan(context)
    assert first.selected.action.action_id == second.selected.action.action_id
    assert ClosedLoopResidualPlanner().plan(context).selected.action.action_id == "shield"


def test_oracle_uses_separate_truth_context() -> None:
    public = _context()
    context = OraclePlanningContext(
        public,
        TruthState(),
        {"measure": 4.0, "shield": 10.0, "near": 1.0},
    )
    decision = OraclePlanner().plan(context)
    assert decision is not None
    assert decision.selected.action.action_id == "near"


def test_non_oracle_planners_do_not_reference_truth_state() -> None:
    root = Path(__file__).resolve().parents[2] / "radcounter/core/planning"
    for filename in ("models.py", "candidates.py", "planners.py"):
        tree = ast.parse((root / filename).read_text(encoding="utf-8"))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert "TruthState" not in names


def test_water_treatment_requires_clean_water_and_wastewater_capacity() -> None:
    candidate = ActionCandidate(
        CountermeasureAction(
            "wash",
            ActionType.DECONTAMINATE,
            "robot",
            parameters={
                "treatment_method": "water_jet",
                "clean_water_l": 8.0,
                "wastewater_l": 6.0,
            },
        ),
        ActionMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
    )
    resources = ResourceState(
        remaining_clean_water_l=7.0,
        remaining_wastewater_capacity_l=5.0,
    )
    report = DeterministicFeasibilityChecker().evaluate(candidate, resources)
    assert not report.feasible
    assert "clean_water_exhausted" in report.reasons
    assert "wastewater_capacity_exhausted" in report.reasons
    clone = resources.clone()
    assert clone.remaining_clean_water_l == 7.0
    assert clone.remaining_wastewater_capacity_l == 5.0


def test_water_treatment_fails_closed_without_explicit_resource_prediction() -> None:
    candidate = ActionCandidate(
        CountermeasureAction(
            "wash",
            ActionType.DECONTAMINATE,
            "robot",
            parameters={"treatment_method": "water_jet"},
        ),
        ActionMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
    )
    report = DeterministicFeasibilityChecker().evaluate(candidate, ResourceState())
    assert not report.feasible
    assert report.reasons == ("water_resource_requirements_missing",)


def test_risk_aware_planner_rejects_action_with_bad_upper_tail() -> None:
    safe = _candidate("safe", ActionType.PLACE_SHIELD, task_dose=4.0, peak=0.0, distance=1.0)
    risky = ActionCandidate(
        _candidate("risky", ActionType.PLACE_SHIELD, task_dose=2.0, peak=0.0, distance=1.0).action,
        ActionMetrics(2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        outcome_samples=ActionOutcomeSamples(
            task_path_dose_sv=(1.0, 1.0, 1.0, 20.0),
            peak_dose_rate_sv_h=(0.0,) * 4,
            residual_source_uncertainty=(0.0,) * 4,
            action_time_s=(0.0,) * 4,
            resource_cost=(0.0,) * 4,
            robot_execution_risk=(0.0,) * 4,
            information_gain=(0.0,) * 4,
        ),
    )
    context = PlanningContext(
        _belief(),
        (safe, risky),
        ResourceState({"robot_s": 10.0}),
        ObjectiveWeights(peak=0.0, uncertainty=0.0, time=0.0, resource=0.0, risk=0.0),
    )
    decision = RiskAwareCountermeasurePlanner(tail_probability=0.25, risk_aversion=1.0).plan(
        context
    )
    assert decision is not None
    assert decision.selected.action.action_id == "safe"
