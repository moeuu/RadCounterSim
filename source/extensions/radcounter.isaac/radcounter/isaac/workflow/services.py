"""Live Isaac implementation of the Core WorkflowServices protocol."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from radcounter.core.actions.resources import ResourceState
from radcounter.core.models.actions import (
    ActionResult,
    ActionStatus,
    ActionType,
    CountermeasureAction,
)
from radcounter.core.models.state import BeliefState, RevisionState
from radcounter.core.planning.models import ObjectiveWeights, PlanningContext
from radcounter.core.workflow.coordinator import TaskMetrics

from ..planning.scene_candidates import IsaacActionCandidateGenerator


@dataclass(frozen=True, slots=True)
class PublicMeasurement:
    """Estimator-safe detector observation with no expected/Truth fields."""

    detector_path: str
    detector_id: str
    duration_s: float
    counts: int
    measured_rate_cps: float

    def as_dict(self) -> dict[str, object]:
        return {
            "detector_path": self.detector_path,
            "detector_id": self.detector_id,
            "duration_s": self.duration_s,
            "counts": self.counts,
            "measured_rate_cps": self.measured_rate_cps,
        }


@dataclass(frozen=True, slots=True)
class WorkflowResidual:
    detector_paths: tuple[str, ...]
    predicted_rate_cps: np.ndarray
    observed_rate_cps: np.ndarray
    residual_rate_cps: np.ndarray
    normalized_residual: np.ndarray
    confidence: float

    def as_dict(self) -> dict[str, object]:
        return {
            "detector_paths": list(self.detector_paths),
            "predicted_rate_cps": self.predicted_rate_cps.tolist(),
            "observed_rate_cps": self.observed_rate_cps.tolist(),
            "residual_rate_cps": self.residual_rate_cps.tolist(),
            "normalized_residual": self.normalized_residual.tolist(),
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class _ExecutionReport:
    success: bool
    message: str
    public_details: Mapping[str, object]


def _motion_audit(report: object) -> Mapping[str, object]:
    if is_dataclass(report):
        return asdict(report)
    return {
        name: getattr(report, name)
        for name in (
            "state",
            "steps",
            "message",
            "phases",
            "grasp_distance_m",
            "placement_error_m",
        )
        if hasattr(report, name)
    }


EstimatorCallback = Callable[[tuple[PublicMeasurement, ...], BeliefState | None], BeliefState]
PreviewCallback = Callable[[CountermeasureAction, BeliefState], object]
DiagnosisCallback = Callable[[object, tuple[PublicMeasurement, ...]], object]
UpdateCallback = Callable[[BeliefState, object, RevisionState], BeliefState]
TaskEvaluator = Callable[[BeliefState], TaskMetrics]
SimulationClock = Callable[[], float]


class IsaacWorkflowServices:
    """Execute Core workflow transitions without exposing simulator Truth."""

    def __init__(
        self,
        radiation_simulation: Any,
        candidate_generator: IsaacActionCandidateGenerator,
        estimator: EstimatorCallback,
        *,
        controller: Any | None = None,
        measurement_controller: Any | None = None,
        decontaminators: Mapping[object, Any] | None = None,
        resources: ResourceState | None = None,
        weights: ObjectiveWeights | None = None,
        previewer: PreviewCallback | None = None,
        diagnoser: DiagnosisCallback | None = None,
        updater: UpdateCallback | None = None,
        task_evaluator: TaskEvaluator | None = None,
        reestimate_after_verification: bool = False,
        simulation_time_s: SimulationClock,
        physics_dt_s: float = 1.0 / 60.0,
        artifact_path: str | Path | None = None,
    ) -> None:
        if physics_dt_s <= 0:
            raise ValueError("physics_dt_s must be positive")
        self.simulation = radiation_simulation
        self.candidate_generator = candidate_generator
        self.estimator_callback = estimator
        self.controller = controller
        self.measurement_controller = measurement_controller
        self.decontaminators: dict[tuple[str, str], Any] = {}
        for key, decontaminator in dict(decontaminators or {}).items():
            if isinstance(key, tuple) and len(key) == 2:
                surface_path, method = map(str, key)
            else:
                surface_path = str(key)
                method = str(getattr(decontaminator, "treatment_method", "dry_contact"))
            if method not in {"dry_contact", "water_jet"}:
                raise ValueError(f"unsupported registered treatment method: {method!r}")
            normalized = (surface_path, method)
            if normalized in self.decontaminators:
                raise ValueError(f"duplicate decontaminator registration: {normalized}")
            self.decontaminators[normalized] = decontaminator
        self.resources = resources if resources is not None else ResourceState()
        self.weights = weights or ObjectiveWeights()
        self.previewer = previewer
        self.diagnoser = diagnoser
        self.updater = updater
        self.task_evaluator = task_evaluator
        self.reestimate_after_verification = bool(reestimate_after_verification)
        self.simulation_time_s = simulation_time_s
        self.physics_dt_s = physics_dt_s
        self.artifact_path = None if artifact_path is None else Path(artifact_path)
        self.revision = RevisionState()
        self.last_measurement: tuple[PublicMeasurement, ...] = ()
        self.last_verification: tuple[PublicMeasurement, ...] = ()
        self.last_prediction: object | None = None
        self.last_diagnosis: object | None = None
        self.last_action: CountermeasureAction | None = None
        self.last_belief: BeliefState | None = None
        self.last_candidates = ()

    @staticmethod
    def _public(records: tuple[Any, ...]) -> tuple[PublicMeasurement, ...]:
        return tuple(
            PublicMeasurement(
                detector_path=record.detector_path,
                detector_id=record.detector_id,
                duration_s=float(record.duration_s),
                counts=int(record.counts),
                measured_rate_cps=float(record.measured_rate_cps),
            )
            for record in records
        )

    def _sim_time(self) -> float:
        value = float(self.simulation_time_s())
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError("simulation clock must return a finite nonnegative time")
        return value

    async def initialize(self) -> None:
        changed = self.simulation.synchronize()
        if changed:
            self.revision.bump_geometry()

    def _consume_measurement_time(self, duration_s: float) -> None:
        if math.isfinite(self.resources.remaining_measurement_time_s):
            self.resources.remaining_measurement_time_s = max(
                0.0,
                self.resources.remaining_measurement_time_s - duration_s,
            )

    async def measure(self) -> tuple[PublicMeasurement, ...]:
        duration = self.simulation.configuration.duration_s
        records = tuple(self.simulation.measure(duration_s=duration))
        self.last_measurement = self._public(records)
        self._consume_measurement_time(duration)
        return self.last_measurement

    def estimate(
        self,
        measurement: object,
        previous: BeliefState | None,
    ) -> BeliefState:
        if not isinstance(measurement, tuple) or not all(
            isinstance(item, PublicMeasurement) for item in measurement
        ):
            raise TypeError("Isaac estimator input must be PublicMeasurement records")
        belief = self.estimator_callback(measurement, previous)
        if not isinstance(belief, BeliefState):
            raise TypeError("the injected estimator must return BeliefState")
        self.last_belief = belief
        return belief

    def planning_context(
        self,
        belief: BeliefState,
        diagnosis: object | None,
    ) -> PlanningContext:
        self.last_candidates = self.candidate_generator.generate_all(belief, diagnosis)
        residual = diagnosis if hasattr(diagnosis, "confidence") else None
        return PlanningContext(
            belief,
            self.last_candidates,
            self.resources.clone(),
            self.weights,
            residual,
        )

    def preview(self, action: CountermeasureAction, belief: BeliefState) -> object:
        self.last_prediction = (
            self.previewer(action, belief)
            if self.previewer is not None
            else self.candidate_generator.preview(action, belief)
        )
        return self.last_prediction

    def _collateral_positions(
        self, excluded_object_path: str | None = None
    ) -> dict[str, np.ndarray]:
        """Snapshot non-target movable props for post-motion safety auditing."""

        positions: dict[str, np.ndarray] = {}
        stage = getattr(self.controller, "stage", None)
        if stage is None:
            return positions
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if excluded_object_path and (
                path == excluded_object_path
                or path.startswith(excluded_object_path.rstrip("/") + "/")
            ):
                continue
            movable = prim.GetAttribute("rad:manipulation:movable")
            if not movable or not movable.HasAuthoredValueOpinion() or not bool(movable.Get()):
                continue
            positions[path] = self.candidate_generator.probe.world_position(prim)
        return positions

    def _collateral_motion_audit(
        self,
        before: Mapping[str, np.ndarray],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        audit: list[dict[str, object]] = []
        violations: list[dict[str, object]] = []
        for path, initial in before.items():
            final = self.candidate_generator.probe.world_position(path)
            displacement = float(np.linalg.norm(final - initial))
            row: dict[str, object] = {
                "object_path": path,
                "displacement_m": displacement,
            }
            audit.append(row)
            if displacement > 0.05:
                violations.append(row)
        return audit, violations

    def _pick_and_place(self, action: CountermeasureAction) -> _ExecutionReport:
        if self.controller is None:
            return _ExecutionReport(False, "countermeasure controller is not configured", {})
        parameters = action.parameters
        object_path = str(parameters.get("object_path", action.target_prim_path or ""))
        collateral_before = self._collateral_positions(object_path)
        pickup = np.asarray(parameters["pickup_base_position_m"], dtype=np.float64)
        placement = np.asarray(parameters["placement_base_position_m"], dtype=np.float64)
        pickup_yaw = parameters.get("pickup_base_yaw_rad")
        placement_yaw = parameters.get("placement_base_yaw_rad")
        pickup_route = parameters.get("pickup_base_route_m")
        placement_route = parameters.get("placement_base_route_m")
        plan_audit = {
            "pickup_base_position_m": pickup.tolist(),
            "placement_base_position_m": placement.tolist(),
            "pickup_base_route_m": pickup_route,
            "placement_base_route_m": placement_route,
            "pickup_base_yaw_rad": pickup_yaw,
            "placement_base_yaw_rad": placement_yaw,
            "target_root_position_m": (
                None
                if action.target_pose_world is None
                else action.target_pose_world[:3, 3].tolist()
            ),
        }
        report = self.controller.execute_pick_and_place(
            object_path,
            pickup,
            placement,
            pickup_base_yaw_rad=None if pickup_yaw is None else float(pickup_yaw),
            placement_base_yaw_rad=(None if placement_yaw is None else float(placement_yaw)),
            pickup_base_route_m=pickup_route,
            placement_base_route_m=placement_route,
            target_root_position_m=(
                None if action.target_pose_world is None else action.target_pose_world[:3, 3]
            ),
            placement_settle_tolerance_m=float(
                parameters.get("placement_settle_tolerance_m", 0.15)
            ),
        )
        released_into_disposal = (
            action.action_type == ActionType.REMOVE_OBJECT
            and "release" in tuple(getattr(report, "phases", ()))
        )
        if action.action_type == ActionType.REMOVE_OBJECT and (
            report.success or released_into_disposal
        ):
            disposal = self.controller.remove_to_disposal_zone(
                object_path,
                str(parameters["disposal_zone_path"]),
            )
            if disposal.success:
                return _ExecutionReport(
                    True,
                    disposal.message,
                    {
                        "object_path": object_path,
                        "steps": report.steps + disposal.steps,
                        "motion_audit": _motion_audit(report),
                        "disposal_audit": _motion_audit(disposal),
                    },
                )
            if report.success:
                return _ExecutionReport(
                    False,
                    disposal.message,
                    {
                        "steps": report.steps + disposal.steps,
                        "motion_audit": _motion_audit(report),
                        "disposal_audit": _motion_audit(disposal),
                    },
                )
        if not report.success:
            collateral_audit, collateral_violations = self._collateral_motion_audit(
                collateral_before
            )
            message = report.message
            if collateral_violations:
                paths = ", ".join(str(row["object_path"]) for row in collateral_violations)
                message = f"{message}; unintended object motion detected: {paths}"
            return _ExecutionReport(
                False,
                message,
                {
                    "steps": report.steps,
                    "motion_audit": _motion_audit(report),
                    "collateral_motion_audit": collateral_audit,
                    "plan_audit": plan_audit,
                },
            )
        collateral_audit, collateral_violations = self._collateral_motion_audit(collateral_before)
        if collateral_violations:
            paths = ", ".join(str(row["object_path"]) for row in collateral_violations)
            return _ExecutionReport(
                False,
                f"unintended object motion detected: {paths}",
                {
                    "object_path": object_path,
                    "steps": report.steps,
                    "motion_audit": _motion_audit(report),
                    "collateral_motion_audit": collateral_audit,
                    "plan_audit": plan_audit,
                },
            )
        return _ExecutionReport(
            True,
            report.message,
            {
                "object_path": object_path,
                "steps": report.steps,
                "physical_state": str(report.state),
                "motion_audit": _motion_audit(report),
                "collateral_motion_audit": collateral_audit,
                "plan_audit": plan_audit,
            },
        )

    def _execute_measurement_move(self, action: CountermeasureAction) -> _ExecutionReport:
        if self.measurement_controller is None:
            return _ExecutionReport(False, "measurement robot controller is not configured", {})
        assert action.target_pose_world is not None
        route = action.parameters.get("base_route_m")
        report = (
            self.measurement_controller.navigate_route(route)
            if route and hasattr(self.measurement_controller, "navigate_route")
            else self.measurement_controller.navigate_to(action.target_pose_world[:3, 3])
        )
        return _ExecutionReport(
            report.success,
            report.message,
            {
                "steps": report.steps,
                "detector_path": action.target_prim_path,
                "base_route_m": route,
                "motion_audit": _motion_audit(report),
            },
        )

    def _execute_decontamination(self, action: CountermeasureAction) -> _ExecutionReport:
        surface_path = str(action.parameters.get("surface_path", action.target_prim_path or ""))
        method = str(action.parameters.get("treatment_method", "dry_contact"))
        if method not in {"dry_contact", "water_jet"}:
            return _ExecutionReport(False, f"unsupported treatment method {method!r}", {})
        decontaminator = self.decontaminators.get((surface_path, method))
        if decontaminator is None:
            return _ExecutionReport(
                False,
                f"no {method} decontaminator for {surface_path}",
                {},
            )
        if getattr(decontaminator, "treatment_method", None) != method:
            return _ExecutionReport(False, "registered treatment implementation mismatch", {})
        if method == "water_jet":
            if "clean_water_l" not in action.parameters or "wastewater_l" not in action.parameters:
                return _ExecutionReport(False, "water resource requirements are missing", {})
            requested_water = float(action.parameters["clean_water_l"])
            requested_wastewater = float(action.parameters["wastewater_l"])
            if (
                not math.isfinite(requested_water)
                or not math.isfinite(requested_wastewater)
                or requested_water < 0.0
                or requested_wastewater < 0.0
                or requested_water > self.resources.remaining_clean_water_l
                or requested_wastewater > self.resources.remaining_wastewater_capacity_l
            ):
                return _ExecutionReport(False, "water resources are invalid or exhausted", {})
            state = decontaminator.state
            state.supply_remaining_l = min(
                state.supply_remaining_l,
                requested_water,
                self.resources.remaining_clean_water_l,
            )
            state.wastewater_capacity_l = state.wastewater_volume_l + min(
                state.wastewater_capacity_remaining_l,
                requested_wastewater,
                self.resources.remaining_wastewater_capacity_l,
            )
        collateral_before = self._collateral_positions()
        stow = None
        if self.controller is not None and hasattr(self.controller, "stow_arm"):
            stow = self.controller.stow_arm()
            if not stow.success:
                return _ExecutionReport(
                    False,
                    stow.message,
                    {"steps": stow.steps, "stow_audit": _motion_audit(stow)},
                )
        navigation = None
        if self.controller is not None and "pickup_base_position_m" in action.parameters:
            route = action.parameters.get("pickup_base_route_m")
            navigation = (
                self.controller.navigate_route(
                    route,
                    action.parameters.get("pickup_base_yaw_rad"),
                )
                if route and hasattr(self.controller, "navigate_route")
                else self.controller.navigate_to(action.parameters["pickup_base_position_m"])
            )
            if not navigation.success:
                return _ExecutionReport(
                    False,
                    navigation.message,
                    {
                        "steps": navigation.steps + (0 if stow is None else stow.steps),
                        "stow_audit": None if stow is None else _motion_audit(stow),
                        "navigation_audit": _motion_audit(navigation),
                        "base_route_m": route,
                    },
                )
        duration = float(action.parameters.get("duration_s", action.predicted_duration_s))
        if self.controller is not None and hasattr(
            self.controller, "execute_surface_decontamination"
        ):
            report = self.controller.execute_surface_decontamination(
                decontaminator,
                duration,
            )
            digest = decontaminator.flush()
            self.simulation.refresh_scene_state()
            collateral_audit, collateral_violations = self._collateral_motion_audit(
                collateral_before
            )
            success = bool(report.success) and not collateral_violations
            treatment_details: dict[str, object] = {
                "surface_path": surface_path,
                "treatment_method": method,
                "accepted_contacts": int(report.accepted_contacts),
                "removed_activity_bq": float(report.removed_activity_bq),
                "activity_balance_error_bq": float(report.activity_balance_error_bq),
                "treated_area_coverage_fraction": float(report.coverage_fraction),
                "treatment_model_id": decontaminator.treatment_model.model_id,
                "treatment_data_status": decontaminator.treatment_model.status,
                "treatment_numeric_sha256": (decontaminator.treatment_model.numeric_sha256),
                "minimum_tool_dwell_s": float(report.minimum_tool_dwell_s),
                "dwell_qualified_triangle_indices": list(report.dwell_qualified_triangle_indices),
                "dwell_pending_triangle_indices": list(report.dwell_pending_triangle_indices),
                "activity_map_sha256": digest,
                "stow_audit": None if stow is None else _motion_audit(stow),
                "navigation_audit": (None if navigation is None else _motion_audit(navigation)),
                "motion_audit": _motion_audit(report),
                "collateral_motion_audit": collateral_audit,
            }
            if method == "water_jet":
                state = decontaminator.state
                treatment_details.update(
                    {
                        "water_balance_error_l": float(
                            decontaminator.process.water_balance_error_l
                        ),
                        "applied_water_l": float(state.applied_water_l),
                        "recovered_water_l": float(state.recovered_water_l),
                        "retained_surface_water_l": float(state.retained_surface_water_l),
                        "discharged_water_l": float(state.discharged_water_l),
                        "captured_activity_bq": float(state.captured_activity_bq),
                        "redeposited_activity_bq": float(state.redeposited_activity_bq),
                        "discharged_activity_bq": float(state.discharged_activity_bq),
                    }
                )
            return _ExecutionReport(
                success,
                f"articulated {method} decontamination completed"
                if success
                else (
                    "unintended object motion detected during decontamination"
                    if collateral_violations
                    else "articulated contact decontamination failed"
                ),
                treatment_details,
            )
        ticks = max(1, int(math.ceil(duration / self.physics_dt_s)))
        removed = 0.0
        contacts = 0
        dwell_qualified: set[int] = set()
        dwell_pending: set[int] = set()
        activity_balance_error_bq = 0.0
        for step in range(1, ticks + 1):
            treatment = decontaminator.tick(self.physics_dt_s, step)
            removed += float(treatment.removed_activity_bq)
            contacts += int(treatment.accepted_contacts)
            dwell_qualified.update(treatment.dwell_qualified_triangle_indices)
            dwell_pending.update(treatment.dwell_pending_triangle_indices)
            dwell_pending.difference_update(dwell_qualified)
            activity_balance_error_bq = float(treatment.activity_balance_error_bq)
            if self.controller is not None:
                self.controller.stepper.step(render=False)
        digest = decontaminator.flush()
        self.simulation.refresh_scene_state()
        treatment_details = {
            "surface_path": surface_path,
            "treatment_method": method,
            "ticks": ticks,
            "accepted_contacts": contacts,
            "removed_activity_bq": removed,
            "activity_balance_error_bq": activity_balance_error_bq,
            "treated_area_coverage_fraction": len(dwell_qualified)
            / max(len(decontaminator.triangle_indices), 1),
            "treatment_model_id": decontaminator.treatment_model.model_id,
            "treatment_data_status": decontaminator.treatment_model.status,
            "treatment_numeric_sha256": decontaminator.treatment_model.numeric_sha256,
            "minimum_tool_dwell_s": decontaminator.minimum_tool_dwell_s,
            "dwell_qualified_triangle_indices": sorted(dwell_qualified),
            "dwell_pending_triangle_indices": sorted(dwell_pending),
            "activity_map_sha256": digest,
        }
        if method == "water_jet":
            state = decontaminator.state
            treatment_details.update(
                {
                    "water_balance_error_l": float(decontaminator.process.water_balance_error_l),
                    "applied_water_l": float(state.applied_water_l),
                    "recovered_water_l": float(state.recovered_water_l),
                    "retained_surface_water_l": float(state.retained_surface_water_l),
                    "discharged_water_l": float(state.discharged_water_l),
                    "captured_activity_bq": float(state.captured_activity_bq),
                    "redeposited_activity_bq": float(state.redeposited_activity_bq),
                    "discharged_activity_bq": float(state.discharged_activity_bq),
                }
            )
        return _ExecutionReport(
            contacts > 0 and removed > 0.0,
            (
                "contact decontamination completed"
                if removed > 0.0
                else (
                    "minimum verified dwell was not reached"
                    if contacts
                    else "no valid tool contact"
                )
            ),
            treatment_details,
        )

    def _physical_execute(self, action: CountermeasureAction) -> _ExecutionReport:
        if action.action_type == ActionType.MEASURE:
            return self._execute_measurement_move(action)
        if action.action_type == ActionType.DECONTAMINATE:
            return self._execute_decontamination(action)
        if action.action_type in {
            ActionType.PLACE_SHIELD,
            ActionType.MOVE_SHIELD,
            ActionType.MOVE_OBJECT,
            ActionType.REMOVE_OBJECT,
            ActionType.REPAIR_ACTION,
        }:
            return self._pick_and_place(action)
        return _ExecutionReport(False, f"unsupported action type: {action.action_type}", {})

    def _update_shield_deployment_metadata(
        self, action: CountermeasureAction
    ) -> Mapping[str, object]:
        """Commit public shield state only after physical placement succeeds."""

        if action.action_type not in {ActionType.PLACE_SHIELD, ActionType.MOVE_SHIELD}:
            return {}
        shield_path = str(action.target_prim_path or "")
        if not shield_path:
            raise RuntimeError("a successful shield action has no target prim path")
        placement_fraction = action.parameters.get("placement_fraction")
        if not isinstance(placement_fraction, (int, float)):
            raise RuntimeError("a successful shield action has no placement fraction")
        placement_fraction = float(placement_fraction)
        if not math.isfinite(placement_fraction) or not 0.0 <= placement_fraction <= 1.0:
            raise RuntimeError("shield placement fraction must be finite and in [0, 1]")

        stage = getattr(self.candidate_generator, "stage", None)
        if stage is None:
            stage = getattr(self.controller, "stage", None)
        if stage is None:
            raise RuntimeError("shield deployment metadata requires a live USD stage")
        shield = stage.GetPrimAtPath(shield_path)
        if not shield or not shield.IsValid():
            raise RuntimeError(f"shield prim is unavailable after placement: {shield_path}")

        deployed_attribute = shield.GetAttribute("rad:shield:deployed")
        fraction_attribute = shield.GetAttribute("rad:shield:placementFraction")
        if (
            not deployed_attribute
            or not deployed_attribute.IsValid()
            or not fraction_attribute
            or not fraction_attribute.IsValid()
        ):
            from pxr import Sdf

        if not deployed_attribute or not deployed_attribute.IsValid():
            deployed_attribute = shield.CreateAttribute(
                "rad:shield:deployed", Sdf.ValueTypeNames.Bool, custom=True
            )
        deployed_attribute.Set(True)
        if not fraction_attribute or not fraction_attribute.IsValid():
            fraction_attribute = shield.CreateAttribute(
                "rad:shield:placementFraction", Sdf.ValueTypeNames.Double, custom=True
            )
        fraction_attribute.Set(placement_fraction)
        return {
            "deployment_state": "deployed",
            "placement_fraction": placement_fraction,
        }

    def _consume(
        self,
        action: CountermeasureAction,
        public_details: Mapping[str, object] | None = None,
    ) -> None:
        self.resources.consume(action.resource_cost)
        runtime = self.resources.remaining_robot_runtime_s.get(action.robot_id)
        if runtime is not None:
            self.resources.remaining_robot_runtime_s[action.robot_id] = max(
                0.0, runtime - action.predicted_duration_s
            )
        if action.action_type == ActionType.MEASURE:
            return
        if math.isfinite(self.resources.remaining_work_time_s):
            self.resources.remaining_work_time_s = max(
                0.0,
                self.resources.remaining_work_time_s - action.predicted_duration_s,
            )
        self.resources.remaining_countermeasure_count = max(
            0, self.resources.remaining_countermeasure_count - 1
        )
        # Deployment consumes one inventory unit. Repositioning an already
        # deployed physical panel consumes time, but never another panel.
        if action.action_type == ActionType.PLACE_SHIELD:
            shield_type = str(action.parameters.get("shield_type", "default"))
            if shield_type in self.resources.remaining_shield_units:
                units = int(action.parameters.get("shield_units", 1))
                self.resources.remaining_shield_units[shield_type] = max(
                    0, self.resources.remaining_shield_units[shield_type] - units
                )
        if action.action_type == ActionType.DECONTAMINATE and math.isfinite(
            self.resources.remaining_decon_media
        ):
            used = float(action.parameters.get("decon_media", 0.0))
            self.resources.remaining_decon_media = max(
                0.0, self.resources.remaining_decon_media - used
            )
        if (
            action.action_type == ActionType.DECONTAMINATE
            and action.parameters.get("treatment_method", "dry_contact") == "water_jet"
        ):
            details = public_details or {}
            clean_water_l = float(
                details.get("applied_water_l", action.parameters["clean_water_l"])
            )
            wastewater_l = float(
                details.get("recovered_water_l", action.parameters["wastewater_l"])
            )
            if math.isfinite(self.resources.remaining_clean_water_l):
                self.resources.remaining_clean_water_l = max(
                    0.0,
                    self.resources.remaining_clean_water_l - clean_water_l,
                )
            if math.isfinite(self.resources.remaining_wastewater_capacity_l):
                self.resources.remaining_wastewater_capacity_l = max(
                    0.0,
                    self.resources.remaining_wastewater_capacity_l - wastewater_l,
                )

    def _bump_revision(self, action: CountermeasureAction) -> None:
        if action.action_type == ActionType.DECONTAMINATE:
            self.revision.bump_source_activity()
        elif action.action_type == ActionType.REMOVE_OBJECT:
            self.revision.bump_geometry()
            self.revision.bump_source_activity()
        elif action.action_type == ActionType.MOVE_OBJECT:
            self.revision.bump_source_pose(geometry_changed=True)
        elif action.action_type in {
            ActionType.PLACE_SHIELD,
            ActionType.MOVE_SHIELD,
            ActionType.REPAIR_ACTION,
        }:
            self.revision.bump_geometry()

    async def execute(self, action: CountermeasureAction) -> ActionResult:
        before = self.revision.copy()
        started = self._sim_time()
        self.last_action = action
        try:
            report = self._physical_execute(action)
        except Exception as exc:
            report = _ExecutionReport(False, f"{type(exc).__name__}: {exc}", {})
        if report.success:
            shield_state = self._update_shield_deployment_metadata(action)
            self._consume(action, report.public_details)
            self._bump_revision(action)
            changed = self.simulation.synchronize()
            status = ActionStatus.COMPLETED
            details = {
                **report.public_details,
                **shield_state,
                "changed_paths": list(changed),
            }
        else:
            status = ActionStatus.FAILED
            details = dict(report.public_details)
        details["message"] = report.message
        completed = self._sim_time()
        if completed < started:
            raise RuntimeError("simulation clock moved backwards during physical execution")
        return ActionResult(
            action_id=action.action_id,
            status=status,
            started_sim_s=started,
            completed_sim_s=completed,
            public_details=details,
            truth_details=None,
            before_revision=before,
            after_revision=self.revision.copy(),
        )

    async def verify(self, action: CountermeasureAction) -> tuple[PublicMeasurement, ...]:
        del action
        self.simulation.synchronize()
        duration = self.simulation.configuration.duration_s
        records = tuple(self.simulation.measure(duration_s=duration))
        self._consume_measurement_time(duration)
        self.last_verification = self._public(records)
        return self.last_verification

    def diagnose(self, predicted: object, observed: object) -> object:
        if not isinstance(observed, tuple) or not all(
            isinstance(item, PublicMeasurement) for item in observed
        ):
            raise TypeError("verification must contain PublicMeasurement records")
        if self.diagnoser is not None:
            self.last_diagnosis = self.diagnoser(predicted, observed)
            self.write_workflow_artifact()
            return self.last_diagnosis
        prediction = predicted if isinstance(predicted, Mapping) else {}
        detector_path = str(prediction.get("detector_path", ""))
        predicted_rate = float(prediction.get("predicted_rate_proxy_cps", 0.0))
        selected = [item for item in observed if item.detector_path == detector_path]
        if not selected:
            selected = list(observed)
        paths = tuple(item.detector_path for item in selected)
        observed_rate = np.asarray([item.measured_rate_cps for item in selected], dtype=np.float64)
        predicted_rates = np.full(len(selected), predicted_rate, dtype=np.float64)
        residual = observed_rate - predicted_rates
        durations_s = np.asarray([item.duration_s for item in selected], dtype=np.float64)
        predicted_count_variance = np.maximum(predicted_rates * durations_s, 1.0)
        normalized_residual = residual * durations_s / np.sqrt(predicted_count_variance)
        scale = float(np.linalg.norm(observed_rate) + np.linalg.norm(predicted_rates) + 1.0e-12)
        confidence = float(np.clip(1.0 - np.linalg.norm(residual) / scale, 0.0, 1.0))
        self.last_diagnosis = WorkflowResidual(
            paths,
            predicted_rates,
            observed_rate,
            residual,
            normalized_residual,
            confidence,
        )
        self.write_workflow_artifact()
        return self.last_diagnosis

    def update(self, belief: BeliefState, diagnosis: object) -> BeliefState:
        if self.updater is not None:
            updated = self.updater(belief, diagnosis, self.revision.copy())
            if not isinstance(updated, BeliefState):
                raise TypeError("the injected updater must return BeliefState")
            self.last_belief = updated
            self.write_workflow_artifact()
            return updated
        if self.reestimate_after_verification:
            if not self.last_verification:
                raise RuntimeError("belief update requires a completed verification measurement")
            estimate = self.estimator_callback(self.last_verification, belief)
            if not isinstance(estimate, BeliefState):
                raise TypeError("the estimator callback must return BeliefState")
            updated = BeliefState(
                estimate.basis_ids,
                estimate.source_strength_bq.copy(),
                estimate.covariance.copy(),
                self.revision.copy(),
                dict(estimate.remaining_resources),
                dict(estimate.action_effect_parameters),
            )
            self.last_belief = updated
            self.write_workflow_artifact()
            return updated
        updated = BeliefState(
            belief.basis_ids,
            belief.source_strength_bq.copy(),
            belief.covariance.copy(),
            self.revision.copy(),
            dict(belief.remaining_resources),
            dict(belief.action_effect_parameters),
        )
        self.last_belief = updated
        self.write_workflow_artifact()
        return updated

    def evaluate_task(self, belief: BeliefState) -> TaskMetrics:
        if self.task_evaluator is not None:
            return self.task_evaluator(belief)
        path_dose, peak = self.candidate_generator.evaluate_task(belief)
        return TaskMetrics(path_dose, peak)

    def resources_exhausted(self) -> bool:
        no_countermeasures = self.resources.remaining_countermeasure_count <= 0
        no_work_time = self.resources.remaining_work_time_s <= 0
        no_robot_runtime = bool(self.resources.remaining_robot_runtime_s) and all(
            value <= 0 for value in self.resources.remaining_robot_runtime_s.values()
        )
        return no_countermeasures or no_work_time or no_robot_runtime

    def workflow_view(self) -> dict[str, object]:
        diagnosis = self.last_diagnosis
        if hasattr(diagnosis, "as_dict"):
            diagnosis = diagnosis.as_dict()
        estimate: dict[str, object] | None = None
        if self.last_belief is not None:
            estimate = {
                "basis_ids": list(self.last_belief.basis_ids),
                "source_strength_bq": self.last_belief.source_strength_bq.tolist(),
                "covariance_diagonal": np.diag(self.last_belief.covariance).tolist(),
                "activity_standard_deviation_bq": np.sqrt(
                    np.maximum(np.diag(self.last_belief.covariance), 0.0)
                ).tolist(),
            }
            try:
                positions = [
                    self.candidate_generator.probe.world_position(path).tolist()
                    for path in self.last_belief.basis_ids
                ]
            except (AttributeError, TypeError, ValueError):
                positions = []
            if len(positions) == len(self.last_belief.basis_ids):
                estimate["positions_world_m"] = positions
        return {
            "measurement": [item.as_dict() for item in self.last_measurement],
            "estimate": estimate,
            "selected_action": (
                None
                if self.last_action is None
                else {
                    "action_id": self.last_action.action_id,
                    "action_type": str(self.last_action.action_type),
                    "target": self.last_action.target_prim_path,
                }
            ),
            "prediction": self.last_prediction,
            "verification": [item.as_dict() for item in self.last_verification],
            "residual": diagnosis,
            "revision": vars(self.revision),
            "resources": {
                "available": dict(self.resources.available),
                "consumed": dict(self.resources.consumed),
                "measurement_time_s": self.resources.remaining_measurement_time_s,
                "work_time_s": self.resources.remaining_work_time_s,
                "robot_runtime_s": dict(self.resources.remaining_robot_runtime_s),
                "shield_units": dict(self.resources.remaining_shield_units),
                "decon_media": self.resources.remaining_decon_media,
                "clean_water_l": self.resources.remaining_clean_water_l,
                "wastewater_capacity_l": self.resources.remaining_wastewater_capacity_l,
                "countermeasure_count": self.resources.remaining_countermeasure_count,
            },
        }

    def write_workflow_artifact(self, path: str | Path | None = None) -> Path | None:
        destination = self.artifact_path if path is None else Path(path)
        if destination is None:
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(self.workflow_view(), stream, indent=2, sort_keys=True, default=str)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        except Exception:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name)
            raise
        return destination
