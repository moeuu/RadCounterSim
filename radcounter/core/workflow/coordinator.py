"""Pauseable closed-loop state machine with immutable snapshots."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from radcounter.core.models.actions import ActionResult, ActionStatus, CountermeasureAction
from radcounter.core.models.state import BeliefState
from radcounter.core.planning.models import Planner, PlanningContext, PlanningDecision


class WorkflowState(StrEnum):
    INITIALIZE = "initialize"
    MEASURE = "measure"
    ESTIMATE = "estimate"
    PLAN = "plan"
    PREDICT = "predict"
    EXECUTE = "execute"
    VERIFY = "verify"
    DIAGNOSE = "diagnose"
    UPDATE = "update"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True)
class TaskMetrics:
    task_path_dose_sv: float
    peak_dose_rate_sv_h: float
    safety_violation: bool = False


@dataclass(frozen=True)
class TerminationConfig:
    maximum_cycles: int
    task_path_dose_threshold_sv: float
    peak_dose_rate_threshold_sv_h: float

    def __post_init__(self) -> None:
        if self.maximum_cycles < 1:
            raise ValueError("maximum_cycles must be positive")
        if min(self.task_path_dose_threshold_sv, self.peak_dose_rate_threshold_sv_h) < 0:
            raise ValueError("dose thresholds must be nonnegative")


@dataclass(frozen=True)
class WorkflowSnapshot:
    sequence: int
    state: WorkflowState
    event: str
    payload_json: str


class SnapshotStore:
    """Append immutable snapshots in memory and optionally as JSONL."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = None if path is None else Path(path)
        self._snapshots: list[WorkflowSnapshot] = []
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, state: WorkflowState, event: str, payload: dict[str, Any]) -> None:
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        snapshot = WorkflowSnapshot(len(self._snapshots), state, event, payload_json)
        self._snapshots.append(snapshot)
        if self._path is not None:
            record = {
                "sequence": snapshot.sequence,
                "state": snapshot.state,
                "event": snapshot.event,
                "payload": json.loads(snapshot.payload_json),
            }
            with self._path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")

    @property
    def snapshots(self) -> tuple[WorkflowSnapshot, ...]:
        return tuple(self._snapshots)


class WorkflowServices(Protocol):
    """Application adapters invoked by the truth-free coordinator."""

    async def initialize(self) -> None: ...

    async def measure(self) -> object: ...

    def estimate(self, measurement: object, previous: BeliefState | None) -> BeliefState: ...

    def planning_context(
        self, belief: BeliefState, diagnosis: object | None
    ) -> PlanningContext: ...

    def preview(self, action: CountermeasureAction, belief: BeliefState) -> object: ...

    async def execute(self, action: CountermeasureAction) -> ActionResult: ...

    async def verify(self, action: CountermeasureAction) -> object: ...

    def diagnose(self, predicted: object, observed: object) -> object: ...

    def update(self, belief: BeliefState, diagnosis: object) -> BeliefState: ...

    def evaluate_task(self, belief: BeliefState) -> TaskMetrics: ...

    def resources_exhausted(self) -> bool: ...


@dataclass(frozen=True)
class EpisodeResult:
    terminal_state: WorkflowState
    termination_reason: str
    completed_cycles: int
    snapshots: tuple[WorkflowSnapshot, ...]
    belief: BeliefState | None


class ClosedLoopCoordinator:
    """Run MEASURE->...->UPDATE without accessing simulator truth."""

    def __init__(
        self,
        services: WorkflowServices,
        planner: Planner,
        termination: TerminationConfig,
        *,
        snapshot_store: SnapshotStore | None = None,
    ) -> None:
        self.services = services
        self.planner = planner
        self.termination = termination
        self.store = snapshot_store if snapshot_store is not None else SnapshotStore()
        self.state = WorkflowState.INITIALIZE
        self.belief: BeliefState | None = None
        self.measurement: object | None = None
        self.decision: PlanningDecision | None = None
        self.predicted: object | None = None
        self.observed: object | None = None
        self.diagnosis: object | None = None
        self.completed_cycles = 0
        self.termination_reason = ""
        self._paused = False
        self._stop_requested = False
        self.store.append(self.state, "created", {})

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._stop_requested = True

    def _transition(self, state: WorkflowState, event: str, **payload: Any) -> WorkflowState:
        self.state = state
        self.store.append(state, event, payload)
        return state

    async def step(self) -> WorkflowState:
        """Advance one transition, or retain state while paused."""

        if self.state in {WorkflowState.COMPLETE, WorkflowState.FAILED}:
            return self.state
        if self._paused:
            return self.state
        if self._stop_requested:
            self.termination_reason = "stop_requested"
            return self._transition(WorkflowState.COMPLETE, "stopped")
        try:
            if self.state == WorkflowState.INITIALIZE:
                await self.services.initialize()
                return self._transition(WorkflowState.MEASURE, "initialized")
            if self.state == WorkflowState.MEASURE:
                self.measurement = await self.services.measure()
                return self._transition(WorkflowState.ESTIMATE, "measurement_completed")
            if self.state == WorkflowState.ESTIMATE:
                assert self.measurement is not None
                self.belief = self.services.estimate(self.measurement, self.belief)
                return self._transition(WorkflowState.PLAN, "estimate_completed")
            if self.state == WorkflowState.PLAN:
                assert self.belief is not None
                context = self.services.planning_context(self.belief, self.diagnosis)
                self.decision = self.planner.plan(context)
                if self.decision is None:
                    self.termination_reason = "no_valid_action"
                    return self._transition(WorkflowState.COMPLETE, "no_valid_action")
                return self._transition(
                    WorkflowState.PREDICT,
                    "action_selected",
                    action_id=self.decision.selected.action.action_id,
                    planner=self.decision.planner_id,
                )
            if self.state == WorkflowState.PREDICT:
                assert self.belief is not None and self.decision is not None
                self.predicted = self.services.preview(self.decision.selected.action, self.belief)
                return self._transition(WorkflowState.EXECUTE, "post_action_predicted")
            if self.state == WorkflowState.EXECUTE:
                assert self.decision is not None
                result = await self.services.execute(self.decision.selected.action)
                if result.status not in {ActionStatus.COMPLETED, ActionStatus.PARTIAL}:
                    self.termination_reason = f"action_{result.status}"
                    return self._transition(
                        WorkflowState.FAILED,
                        "action_failed",
                        action_id=result.action_id,
                        status=result.status,
                    )
                return self._transition(WorkflowState.VERIFY, "action_completed")
            if self.state == WorkflowState.VERIFY:
                assert self.decision is not None
                self.observed = await self.services.verify(self.decision.selected.action)
                return self._transition(WorkflowState.DIAGNOSE, "verification_completed")
            if self.state == WorkflowState.DIAGNOSE:
                self.diagnosis = self.services.diagnose(self.predicted, self.observed)
                return self._transition(WorkflowState.UPDATE, "diagnosis_completed")
            if self.state == WorkflowState.UPDATE:
                assert self.belief is not None and self.diagnosis is not None
                self.belief = self.services.update(self.belief, self.diagnosis)
                self.completed_cycles += 1
                metrics = self.services.evaluate_task(self.belief)
                reason = self._termination_reason(metrics)
                if reason:
                    self.termination_reason = reason
                    return self._transition(
                        WorkflowState.COMPLETE,
                        "termination_condition",
                        reason=reason,
                        completed_cycles=self.completed_cycles,
                    )
                return self._transition(WorkflowState.PLAN, "belief_updated")
        except Exception as exc:
            self.termination_reason = f"exception:{type(exc).__name__}"
            return self._transition(
                WorkflowState.FAILED,
                "exception",
                exception_type=type(exc).__name__,
                message=str(exc),
            )
        raise RuntimeError(f"unhandled workflow state: {self.state}")

    def _termination_reason(self, metrics: TaskMetrics) -> str:
        if metrics.safety_violation:
            return "safety_violation"
        if (
            metrics.task_path_dose_sv <= self.termination.task_path_dose_threshold_sv
            and metrics.peak_dose_rate_sv_h <= self.termination.peak_dose_rate_threshold_sv_h
        ):
            return "dose_targets_reached"
        if self.services.resources_exhausted():
            return "resources_exhausted"
        if self.completed_cycles >= self.termination.maximum_cycles:
            return "maximum_cycles"
        return ""

    async def run_episode(self) -> EpisodeResult:
        """Run until COMPLETE/FAILED; paused coordinators yield cooperatively."""

        while self.state not in {WorkflowState.COMPLETE, WorkflowState.FAILED}:
            if self._paused:
                await asyncio.sleep(0)
                continue
            await self.step()
        return EpisodeResult(
            self.state,
            self.termination_reason,
            self.completed_cycles,
            self.store.snapshots,
            self.belief,
        )
