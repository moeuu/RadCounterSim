import asyncio
import json

import numpy as np

from radcounter.core.actions import (
    ActionResult,
    ActionStatus,
    ActionType,
    CountermeasureAction,
    ResourceState,
)
from radcounter.core.models import BeliefState, RevisionState
from radcounter.core.planning import (
    ActionCandidate,
    ActionMetrics,
    ClosedLoopResidualPlanner,
    PlanningContext,
)
from radcounter.core.workflow import (
    ClosedLoopCoordinator,
    SnapshotStore,
    TaskMetrics,
    TerminationConfig,
    WorkflowState,
)


class FakeServices:
    def __init__(self, *, provide_action: bool = True, resources_exhausted: bool = False) -> None:
        self.provide_action = provide_action
        self._resources_exhausted = resources_exhausted
        self.revision = RevisionState()

    async def initialize(self) -> None:
        return None

    async def measure(self) -> object:
        return np.array([10.0])

    def estimate(self, measurement: object, previous: BeliefState | None) -> BeliefState:
        del measurement, previous
        return BeliefState(("s",), np.array([10.0]), np.eye(1), self.revision.copy())

    def planning_context(self, belief: BeliefState, diagnosis: object | None) -> PlanningContext:
        del diagnosis
        candidates = ()
        if self.provide_action:
            action = CountermeasureAction(
                "shield", ActionType.PLACE_SHIELD, "robot", predicted_duration_s=1.0
            )
            candidates = (
                ActionCandidate(
                    action,
                    ActionMetrics(0.5, 0.5, 0.1, 1.0, 0.0, 0.0, 0.0, 1.0),
                ),
            )
        return PlanningContext(belief, candidates, ResourceState())

    def preview(self, action: CountermeasureAction, belief: BeliefState) -> object:
        del action, belief
        return np.array([5.0])

    async def execute(self, action: CountermeasureAction) -> ActionResult:
        before = self.revision.copy()
        self.revision.bump_geometry()
        return ActionResult(
            action.action_id,
            ActionStatus.COMPLETED,
            0.0,
            1.0,
            {},
            None,
            before,
            self.revision.copy(),
        )

    async def verify(self, action: CountermeasureAction) -> object:
        del action
        return np.array([5.0])

    def diagnose(self, predicted: object, observed: object) -> object:
        return {"predicted": predicted, "observed": observed}

    def update(self, belief: BeliefState, diagnosis: object) -> BeliefState:
        del diagnosis
        return BeliefState(
            belief.basis_ids,
            belief.source_strength_bq,
            belief.covariance,
            self.revision.copy(),
        )

    def evaluate_task(self, belief: BeliefState) -> TaskMetrics:
        del belief
        return TaskMetrics(0.5, 0.5)

    def resources_exhausted(self) -> bool:
        return self._resources_exhausted


def test_full_closed_loop_reaches_dose_target_and_saves_snapshots(tmp_path) -> None:
    store = SnapshotStore(tmp_path / "snapshots.jsonl")
    coordinator = ClosedLoopCoordinator(
        FakeServices(),
        ClosedLoopResidualPlanner(),
        TerminationConfig(3, 1.0, 1.0),
        snapshot_store=store,
    )
    result = asyncio.run(coordinator.run_episode())
    assert result.terminal_state == WorkflowState.COMPLETE
    assert result.termination_reason == "dose_targets_reached"
    assert result.completed_cycles == 1
    states = [snapshot.state for snapshot in result.snapshots]
    assert states == [
        WorkflowState.INITIALIZE,
        WorkflowState.MEASURE,
        WorkflowState.ESTIMATE,
        WorkflowState.PLAN,
        WorkflowState.PREDICT,
        WorkflowState.EXECUTE,
        WorkflowState.VERIFY,
        WorkflowState.DIAGNOSE,
        WorkflowState.UPDATE,
        WorkflowState.COMPLETE,
    ]
    records = [json.loads(line) for line in (tmp_path / "snapshots.jsonl").read_text().splitlines()]
    assert len(records) == len(states)


def test_no_valid_action_terminates_without_execution() -> None:
    coordinator = ClosedLoopCoordinator(
        FakeServices(provide_action=False),
        ClosedLoopResidualPlanner(),
        TerminationConfig(3, 0.0, 0.0),
    )
    result = asyncio.run(coordinator.run_episode())
    assert result.termination_reason == "no_valid_action"
    assert result.completed_cycles == 0


def test_pause_resume_and_stop_are_controlled() -> None:
    coordinator = ClosedLoopCoordinator(
        FakeServices(),
        ClosedLoopResidualPlanner(),
        TerminationConfig(3, 0.0, 0.0),
    )
    coordinator.pause()
    assert asyncio.run(coordinator.step()) == WorkflowState.INITIALIZE
    coordinator.resume()
    assert asyncio.run(coordinator.step()) == WorkflowState.MEASURE
    coordinator.stop()
    assert asyncio.run(coordinator.step()) == WorkflowState.COMPLETE
    assert coordinator.termination_reason == "stop_requested"


def test_resource_exhaustion_is_recorded_after_update() -> None:
    coordinator = ClosedLoopCoordinator(
        FakeServices(resources_exhausted=True),
        ClosedLoopResidualPlanner(),
        TerminationConfig(3, 0.0, 0.0),
    )
    result = asyncio.run(coordinator.run_episode())
    assert result.termination_reason == "resources_exhausted"
