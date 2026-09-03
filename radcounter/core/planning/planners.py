"""Truth-free baseline and proposed planners."""

from __future__ import annotations

import numpy as np

from radcounter.core.models.actions import ActionType
from radcounter.core.planning.candidates import DeterministicFeasibilityChecker
from radcounter.core.planning.models import (
    ActionCandidate,
    PlanningContext,
    PlanningDecision,
)


def _feasible(
    context: PlanningContext,
) -> tuple[list[ActionCandidate], dict[str, tuple[str, ...]]]:
    checker = DeterministicFeasibilityChecker()
    candidates: list[ActionCandidate] = []
    rejected: dict[str, tuple[str, ...]] = {}
    for candidate in context.candidates:
        report = checker.evaluate(candidate, context.resources)
        if report.feasible:
            candidates.append(candidate)
        else:
            rejected[candidate.action.action_id] = report.reasons
    return candidates, rejected


def _decision(
    planner_id: str,
    selected: ActionCandidate,
    scores: dict[str, float],
    rejected: dict[str, tuple[str, ...]],
) -> PlanningDecision:
    return PlanningDecision(
        selected,
        planner_id,
        scores[selected.action.action_id],
        scores,
        rejected,
    )


class OpenLoopPlanner:
    """Execute a fixed action sequence without residual-driven replanning."""

    planner_id = "open_loop"

    def __init__(self, action_sequence: tuple[str, ...]) -> None:
        self._sequence = action_sequence
        self._cursor = 0

    def plan(self, context: PlanningContext) -> PlanningDecision | None:
        candidates, rejected = _feasible(context)
        by_id = {candidate.action.action_id: candidate for candidate in candidates}
        while self._cursor < len(self._sequence):
            action_id = self._sequence[self._cursor]
            self._cursor += 1
            candidate = by_id.get(action_id)
            if candidate is not None:
                score = context.weights.score(candidate.metrics)
                return _decision(self.planner_id, candidate, {action_id: score}, rejected)
        return None


class GreedyDoseReductionPlanner:
    """Minimize predicted task-path plus peak dose only."""

    planner_id = "greedy_dose_reduction"

    def plan(self, context: PlanningContext) -> PlanningDecision | None:
        candidates, rejected = _feasible(context)
        if not candidates:
            return None
        scores = {
            candidate.action.action_id: candidate.metrics.expected_task_path_dose_sv
            + candidate.metrics.expected_peak_dose_rate_sv_h
            for candidate in candidates
        }
        selected = min(candidates, key=lambda candidate: scores[candidate.action.action_id])
        return _decision(self.planner_id, selected, scores, rejected)


class NearestSourcePlanner:
    """Choose the nearest feasible target irrespective of dose benefit."""

    planner_id = "nearest_source"

    def plan(self, context: PlanningContext) -> PlanningDecision | None:
        candidates, rejected = _feasible(context)
        if not candidates:
            return None
        scores = {
            candidate.action.action_id: candidate.metrics.distance_to_target_m
            for candidate in candidates
        }
        selected = min(candidates, key=lambda candidate: scores[candidate.action.action_id])
        return _decision(self.planner_id, selected, scores, rejected)


class RandomPlanner:
    """Select uniformly from feasible actions using an injected RNG."""

    planner_id = "random"

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def plan(self, context: PlanningContext) -> PlanningDecision | None:
        candidates, rejected = _feasible(context)
        if not candidates:
            return None
        selected = candidates[int(self._rng.integers(0, len(candidates)))]
        scores = {candidate.action.action_id: 0.0 for candidate in candidates}
        return _decision(self.planner_id, selected, scores, rejected)


class RiskAwareCountermeasurePlanner:
    """Rank feasible actions by mean cost and the upper-tail outcome penalty."""

    planner_id = "risk_aware_countermeasure"

    def __init__(self, *, tail_probability: float = 0.1, risk_aversion: float = 1.0) -> None:
        if not 0.0 < tail_probability <= 1.0:
            raise ValueError("tail_probability must be in (0, 1]")
        if risk_aversion < 0.0:
            raise ValueError("risk_aversion must be nonnegative")
        self._tail_probability = tail_probability
        self._risk_aversion = risk_aversion

    def _score(self, candidate: ActionCandidate, context: PlanningContext) -> float:
        if candidate.outcome_samples is None:
            return context.weights.score(candidate.metrics)
        samples = np.asarray(
            candidate.outcome_samples.objective_samples(context.weights),
            dtype=np.float64,
        )
        expected = float(np.mean(samples))
        tail_count = max(1, int(np.ceil(self._tail_probability * len(samples))))
        upper_tail_mean = float(np.mean(np.sort(samples)[-tail_count:]))
        return expected + self._risk_aversion * (upper_tail_mean - expected)

    def plan(self, context: PlanningContext) -> PlanningDecision | None:
        candidates, rejected = _feasible(context)
        if not candidates:
            return None
        scores = {
            candidate.action.action_id: self._score(candidate, context) for candidate in candidates
        }
        selected = min(
            candidates,
            key=lambda candidate: (scores[candidate.action.action_id], candidate.action.action_id),
        )
        return _decision(self.planner_id, selected, scores, rejected)


class ClosedLoopResidualPlanner:
    """Full objective with ambiguity-aware preference for information actions."""

    planner_id = "closed_loop_residual"

    def __init__(self, *, low_confidence_threshold: float = 0.7) -> None:
        if not 0 <= low_confidence_threshold <= 1:
            raise ValueError("confidence threshold must be in [0,1]")
        self._low_confidence_threshold = low_confidence_threshold

    def plan(self, context: PlanningContext) -> PlanningDecision | None:
        candidates, rejected = _feasible(context)
        if not candidates:
            return None
        confidence = (
            1.0 if context.residual_diagnosis is None else context.residual_diagnosis.confidence
        )
        scores: dict[str, float] = {}
        for candidate in candidates:
            score = context.weights.score(candidate.metrics)
            if confidence < self._low_confidence_threshold:
                if candidate.action.action_type in {ActionType.MEASURE, ActionType.REPAIR_ACTION}:
                    score -= (
                        context.weights.information * candidate.metrics.expected_information_gain
                    )
                else:
                    score += (
                        context.weights.risk
                        * (self._low_confidence_threshold - confidence)
                        * (1.0 + candidate.metrics.robot_execution_risk)
                    )
            scores[candidate.action.action_id] = score
        selected = min(candidates, key=lambda candidate: scores[candidate.action.action_id])
        return _decision(self.planner_id, selected, scores, rejected)
