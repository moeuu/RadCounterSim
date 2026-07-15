"""Evaluation-only planner with explicit simulator-truth access."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from radcounter.core.models.state import TruthState
from radcounter.core.planning.candidates import DeterministicFeasibilityChecker
from radcounter.core.planning.models import PlanningContext, PlanningDecision


@dataclass(frozen=True)
class OraclePlanningContext:
    """Separate context that cannot be passed to non-oracle planners by type."""

    public: PlanningContext
    truth_state: TruthState
    true_outcome_score_by_action_id: Mapping[str, float]


class OraclePlanner:
    """Choose the best true simulated outcome for experiment comparison only."""

    planner_id = "oracle"

    def plan(self, context: OraclePlanningContext) -> PlanningDecision | None:
        _ = context.truth_state
        checker = DeterministicFeasibilityChecker()
        feasible = []
        rejected: dict[str, tuple[str, ...]] = {}
        for candidate in context.public.candidates:
            report = checker.evaluate(candidate, context.public.resources)
            if report.feasible:
                feasible.append(candidate)
            else:
                rejected[candidate.action.action_id] = report.reasons
        scores = {
            candidate.action.action_id: float(
                context.true_outcome_score_by_action_id[candidate.action.action_id]
            )
            for candidate in feasible
            if candidate.action.action_id in context.true_outcome_score_by_action_id
        }
        if not scores:
            return None
        selected = min(
            feasible, key=lambda candidate: scores.get(candidate.action.action_id, float("inf"))
        )
        return PlanningDecision(
            selected,
            self.planner_id,
            scores[selected.action.action_id],
            scores,
            rejected,
        )
