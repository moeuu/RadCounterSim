"""Planner inputs, feasibility facts, objective values, and decisions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from radcounter.core.actions.resources import ResourceState
from radcounter.core.estimation.residual import ResidualDiagnosis
from radcounter.core.models.actions import CountermeasureAction
from radcounter.core.models.state import BeliefState


@dataclass(frozen=True)
class ActionMetrics:
    """Predicted terms in the resource-constrained action objective."""

    expected_task_path_dose_sv: float
    expected_peak_dose_rate_sv_h: float
    residual_source_uncertainty: float
    action_time_s: float
    resource_cost: float
    robot_execution_risk: float
    expected_information_gain: float
    distance_to_target_m: float = float("inf")

    def __post_init__(self) -> None:
        values = (
            self.expected_task_path_dose_sv,
            self.expected_peak_dose_rate_sv_h,
            self.residual_source_uncertainty,
            self.action_time_s,
            self.resource_cost,
            self.robot_execution_risk,
            self.expected_information_gain,
            self.distance_to_target_m,
        )
        if any(value < 0 for value in values):
            raise ValueError("action metric values must be nonnegative")


@dataclass(frozen=True)
class ObjectiveWeights:
    """Nonnegative weights for the smaller-is-better objective."""

    dose: float = 1.0
    peak: float = 1.0
    uncertainty: float = 1.0
    time: float = 1.0
    resource: float = 1.0
    risk: float = 1.0
    information: float = 1.0

    def __post_init__(self) -> None:
        if any(value < 0 for value in vars(self).values()):
            raise ValueError("objective weights must be nonnegative")

    def score(self, metrics: ActionMetrics) -> float:
        """Compute the specification objective; lower is better."""

        return (
            self.dose * metrics.expected_task_path_dose_sv
            + self.peak * metrics.expected_peak_dose_rate_sv_h
            + self.uncertainty * metrics.residual_source_uncertainty
            + self.time * metrics.action_time_s
            + self.resource * metrics.resource_cost
            + self.risk * metrics.robot_execution_risk
            - self.information * metrics.expected_information_gain
        )


@dataclass(frozen=True)
class FeasibilityFacts:
    """Deterministic geometric/operational feasibility inputs."""

    mobile_path_available: bool = True
    manipulator_reachable: bool = True
    collision_free: bool = True
    grasp_frame_available: bool = True
    placement_stable: bool = True
    disposal_capacity_available: bool = True
    robot_available: bool = True


@dataclass(frozen=True)
class ActionOutcomeSamples:
    """Matched public prediction samples for uncertainty-aware action scoring."""

    task_path_dose_sv: tuple[float, ...]
    peak_dose_rate_sv_h: tuple[float, ...]
    residual_source_uncertainty: tuple[float, ...]
    action_time_s: tuple[float, ...]
    resource_cost: tuple[float, ...]
    robot_execution_risk: tuple[float, ...]
    information_gain: tuple[float, ...]

    def __post_init__(self) -> None:
        series = tuple(vars(self).values())
        lengths = {len(values) for values in series}
        if lengths == {0} or len(lengths) != 1:
            raise ValueError("outcome sample arrays must have one common nonzero length")
        if any(value < 0 for values in series for value in values):
            raise ValueError("outcome samples must be nonnegative")

    def objective_samples(self, weights: ObjectiveWeights) -> tuple[float, ...]:
        return tuple(
            weights.dose * dose
            + weights.peak * peak
            + weights.uncertainty * uncertainty
            + weights.time * duration
            + weights.resource * resource
            + weights.risk * risk
            - weights.information * information
            for dose, peak, uncertainty, duration, resource, risk, information in zip(
                self.task_path_dose_sv,
                self.peak_dose_rate_sv_h,
                self.residual_source_uncertainty,
                self.action_time_s,
                self.resource_cost,
                self.robot_execution_risk,
                self.information_gain,
                strict=True,
            )
        )


@dataclass(frozen=True)
class ActionCandidate:
    """One action with public predicted metrics and feasibility facts."""

    action: CountermeasureAction
    metrics: ActionMetrics
    feasibility: FeasibilityFacts = FeasibilityFacts()
    tags: frozenset[str] = frozenset()
    outcome_samples: ActionOutcomeSamples | None = None


@dataclass(frozen=True)
class FeasibilityReport:
    feasible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PlanningContext:
    """Truth-free context accepted by all non-oracle planners."""

    belief: BeliefState
    candidates: tuple[ActionCandidate, ...]
    resources: ResourceState
    weights: ObjectiveWeights = ObjectiveWeights()
    residual_diagnosis: ResidualDiagnosis | None = None


@dataclass(frozen=True)
class PlanningDecision:
    """Selected action plus auditable candidate scores."""

    selected: ActionCandidate
    planner_id: str
    selected_score: float
    score_by_action_id: Mapping[str, float]
    rejected_reasons_by_action_id: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


class Planner(Protocol):
    planner_id: str

    def plan(self, context: PlanningContext) -> PlanningDecision | None: ...
