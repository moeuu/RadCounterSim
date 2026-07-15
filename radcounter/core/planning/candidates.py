"""Candidate grouping and deterministic feasibility checks."""

from __future__ import annotations

from collections.abc import Iterable

from radcounter.core.actions.resources import ResourceState
from radcounter.core.models.actions import ActionType
from radcounter.core.planning.models import (
    ActionCandidate,
    FeasibilityReport,
)


class ActionCandidateGenerator:
    """Expose typed candidate groups from public templates/predictors."""

    def __init__(self, candidates: Iterable[ActionCandidate]) -> None:
        self._candidates = tuple(candidates)
        action_ids = [candidate.action.action_id for candidate in self._candidates]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("candidate action IDs must be unique")

    def _of_type(self, *action_types: ActionType) -> tuple[ActionCandidate, ...]:
        allowed = set(action_types)
        return tuple(
            candidate for candidate in self._candidates if candidate.action.action_type in allowed
        )

    def generate_measurement_actions(self) -> tuple[ActionCandidate, ...]:
        return self._of_type(ActionType.MEASURE)

    def generate_decon_actions(self) -> tuple[ActionCandidate, ...]:
        return self._of_type(ActionType.DECONTAMINATE)

    def generate_shield_actions(self) -> tuple[ActionCandidate, ...]:
        return self._of_type(ActionType.PLACE_SHIELD, ActionType.MOVE_SHIELD)

    def generate_move_remove_actions(self) -> tuple[ActionCandidate, ...]:
        return self._of_type(ActionType.MOVE_OBJECT, ActionType.REMOVE_OBJECT)

    def generate_repair_actions(self) -> tuple[ActionCandidate, ...]:
        return self._of_type(ActionType.REPAIR_ACTION)

    def generate_all(self) -> tuple[ActionCandidate, ...]:
        return self._candidates


class DeterministicFeasibilityChecker:
    """Check geometry, manipulation, robot, and mission budgets."""

    def evaluate(self, candidate: ActionCandidate, resources: ResourceState) -> FeasibilityReport:
        action = candidate.action
        facts = candidate.feasibility
        reasons: list[str] = []
        checks = (
            (facts.mobile_path_available, "mobile_path_unavailable"),
            (facts.manipulator_reachable, "manipulator_unreachable"),
            (facts.collision_free, "collision"),
            (facts.grasp_frame_available, "grasp_frame_unavailable"),
            (facts.placement_stable, "placement_unstable"),
            (facts.disposal_capacity_available, "disposal_capacity_unavailable"),
            (facts.robot_available, "robot_unavailable"),
            (resources.can_afford(action.resource_cost), "generic_resource_shortage"),
        )
        reasons.extend(reason for passed, reason in checks if not passed)
        robot_remaining = resources.remaining_robot_runtime_s.get(action.robot_id, float("inf"))
        if action.predicted_duration_s > robot_remaining:
            reasons.append("robot_runtime_exhausted")
        if action.action_type == ActionType.MEASURE:
            if action.predicted_duration_s > resources.remaining_measurement_time_s:
                reasons.append("measurement_time_exhausted")
        else:
            if action.predicted_duration_s > resources.remaining_work_time_s:
                reasons.append("work_time_exhausted")
            if resources.remaining_countermeasure_count < 1:
                reasons.append("countermeasure_count_exhausted")
        if action.action_type in {ActionType.PLACE_SHIELD, ActionType.MOVE_SHIELD}:
            shield_type = str(action.parameters.get("shield_type", "default"))
            requested = int(action.parameters.get("shield_units", 1))
            available = resources.remaining_shield_units.get(shield_type, 2**31 - 1)
            if requested > available:
                reasons.append("shield_units_exhausted")
        if action.action_type == ActionType.DECONTAMINATE:
            requested_media = float(action.parameters.get("decon_media", 0.0))
            if requested_media > resources.remaining_decon_media:
                reasons.append("decon_media_exhausted")
        return FeasibilityReport(not reasons, tuple(reasons))
