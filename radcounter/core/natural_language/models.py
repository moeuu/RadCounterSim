"""Strict command contracts between a language model and the simulator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CommandName(StrEnum):
    """Only operations that the natural-language layer may request."""

    LOAD_DEFAULT_SCENE = "load_default_scene"
    PLAY = "play"
    PAUSE = "pause"
    STEP = "step"
    RESET = "reset"
    INITIALIZE_RADIATION = "initialize_radiation"
    SYNC_SCENE = "sync_scene"
    MEASURE = "measure"
    RENDER_DOSE_MAP = "render_dose_map"
    EXPORT_MEASUREMENT = "export_measurement"
    EXECUTE_CANDIDATE = "execute_candidate"
    RETURN_MEASUREMENT_ROBOT = "return_measurement_robot"
    SHOW_STATUS = "show_status"


class CompletionCriterion(StrEnum):
    """Public-result predicates allowed to stop a bounded repeated step."""

    DECONTAMINATION_REMOVED_FRACTION_AT_LEAST = (
        "decontamination_removed_fraction_at_least"
    )
    DECONTAMINATION_REMAINING_FRACTION_AT_MOST = (
        "decontamination_remaining_fraction_at_most"
    )
    DECONTAMINATION_COVERAGE_FRACTION_AT_LEAST = (
        "decontamination_coverage_fraction_at_least"
    )
    MEASURED_RATE_CPS_AT_MOST = "measured_rate_cps_at_most"
    SHIELD_PLACEMENT_ERROR_M_AT_MOST = "shield_placement_error_m_at_most"


class StepCompletionCondition(BaseModel):
    """One allowlisted stop-early condition evaluated from public action results."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    criterion: CompletionCriterion
    threshold: float = Field(ge=0.0, le=1.0e12)

    @model_validator(mode="after")
    def validate_threshold_units(self) -> StepCompletionCondition:
        if self.criterion in {
            CompletionCriterion.DECONTAMINATION_REMOVED_FRACTION_AT_LEAST,
            CompletionCriterion.DECONTAMINATION_REMAINING_FRACTION_AT_MOST,
            CompletionCriterion.DECONTAMINATION_COVERAGE_FRACTION_AT_LEAST,
        } and self.threshold > 1.0:
            raise ValueError("fraction completion thresholds must be between zero and one")
        return self


class AvailableAction(BaseModel):
    """One scene-derived physical action exposed to the language model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: str = Field(min_length=1, max_length=160)
    action_type: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=240)
    target: str | None = Field(default=None, max_length=320)
    target_label: str | None = Field(default=None, max_length=240)
    robot_id: str | None = Field(default=None, max_length=320)
    execution_mode: str | None = Field(default=None, max_length=160)
    placement_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    predicted_duration_s: float | None = Field(default=None, ge=0.0, le=3600.0)
    feasible: bool = True


class CommandContext(BaseModel):
    """Small, public-only snapshot supplied to the interpreter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_state: str = Field(default="unknown", max_length=120)
    stage_name: str | None = Field(default=None, max_length=240)
    available_actions: tuple[AvailableAction, ...] = Field(default=(), max_length=96)
    capabilities: tuple[CommandName, ...] = tuple(CommandName)


class CommandStep(BaseModel):
    """One allowlisted operation in a bounded sequential workflow."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    command: CommandName
    candidate_id: str | None = Field(default=None, max_length=160)
    duration_s: float | None = Field(default=None, ge=0.05, le=120.0)
    max_attempts: int = Field(default=1, ge=1, le=5)
    until: StepCompletionCondition | None = None

    @model_validator(mode="before")
    @classmethod
    def discard_irrelevant_optional_arguments(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if normalized.get("command") != CommandName.EXECUTE_CANDIDATE:
            normalized.pop("candidate_id", None)
        if normalized.get("command") != CommandName.MEASURE:
            normalized.pop("duration_s", None)
        if normalized.get("command") not in {
            CommandName.EXECUTE_CANDIDATE,
            CommandName.MEASURE,
        }:
            normalized.pop("max_attempts", None)
            normalized.pop("until", None)
        elif normalized.get("until") is not None and "max_attempts" not in normalized:
            # A stop condition without a bound is unsafe.  Give small local
            # models a conservative host-owned bound instead of an unbounded loop.
            normalized["max_attempts"] = 3
        return normalized

    @model_validator(mode="after")
    def validate_arguments(self) -> CommandStep:
        if self.command == CommandName.EXECUTE_CANDIDATE and not self.candidate_id:
            raise ValueError("execute_candidate requires candidate_id")
        return self


class CommandPlan(BaseModel):
    """Schema-constrained output produced by the local language model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: Literal["ja", "en", "mixed", "unknown"] = "unknown"
    summary: str = Field(min_length=1, max_length=600)
    steps: tuple[CommandStep, ...] = Field(min_length=1, max_length=24)


class PlanValidationError(ValueError):
    """Raised when a syntactically valid model plan is unsafe for current state."""


@dataclass(frozen=True, slots=True)
class ValidatedCommandPlan:
    """Plan approved by deterministic host-side policy."""

    plan: CommandPlan
    requires_confirmation: bool
    warnings: tuple[str, ...] = ()


_CONFIRMATION_COMMANDS = frozenset(
    {
        CommandName.LOAD_DEFAULT_SCENE,
        CommandName.RESET,
        CommandName.EXECUTE_CANDIDATE,
        CommandName.RETURN_MEASUREMENT_ROBOT,
    }
)


def validate_command_plan(
    plan: CommandPlan,
    context: CommandContext,
) -> ValidatedCommandPlan:
    """Resolve model output against current capabilities and candidate identifiers."""

    capabilities = set(context.capabilities)
    candidates = {action.action_id: action for action in context.available_actions}
    warnings: list[str] = []
    execution_count = sum(step.max_attempts for step in plan.steps)
    if execution_count > 48:
        raise PlanValidationError(
            f"workflow expands to {execution_count} executions; the safety limit is 48"
        )
    for index, step in enumerate(plan.steps, start=1):
        if step.command not in capabilities:
            raise PlanValidationError(
                f"step {index} requests unavailable command: {step.command.value}"
            )
        if step.command == CommandName.RETURN_MEASUREMENT_ROBOT:
            warnings.append("Physical action: return measurement robot to its starting position")
            continue
        if step.command == CommandName.MEASURE:
            if step.until is not None and (
                step.until.criterion != CompletionCriterion.MEASURED_RATE_CPS_AT_MOST
            ):
                raise PlanValidationError(
                    f"step {index} uses a completion condition that does not apply to measurement"
                )
            if step.max_attempts > 1:
                warnings.append(
                    f"Bounded repeated measurement: up to {step.max_attempts} attempts"
                )
            continue
        if step.command != CommandName.EXECUTE_CANDIDATE:
            continue
        assert step.candidate_id is not None
        candidate = candidates.get(step.candidate_id)
        if candidate is None:
            raise PlanValidationError(
                f"step {index} references an action not present in the current scene: "
                f"{step.candidate_id}"
            )
        if not candidate.feasible:
            raise PlanValidationError(
                f"step {index} references an infeasible action: {step.candidate_id}"
            )
        shield_retry = (
            candidate.action_type in {"place_shield", "move_shield"}
            and step.until is not None
            and step.until.criterion
            == CompletionCriterion.SHIELD_PLACEMENT_ERROR_M_AT_MOST
        )
        if (
            step.max_attempts > 1
            and candidate.action_type != "decontaminate"
            and not shield_retry
        ):
            raise PlanValidationError(
                f"step {index} repeats {candidate.action_type}; repeat one decontamination "
                "candidate or use explicit ordered steps for physical repositioning"
            )
        if step.until is not None:
            decon_criteria = {
                CompletionCriterion.DECONTAMINATION_REMOVED_FRACTION_AT_LEAST,
                CompletionCriterion.DECONTAMINATION_REMAINING_FRACTION_AT_MOST,
                CompletionCriterion.DECONTAMINATION_COVERAGE_FRACTION_AT_LEAST,
            }
            shield_criteria = {
                CompletionCriterion.SHIELD_PLACEMENT_ERROR_M_AT_MOST,
            }
            if (
                step.until.criterion in decon_criteria
                and candidate.action_type != "decontaminate"
            ) or (
                step.until.criterion in shield_criteria
                and candidate.action_type not in {"place_shield", "move_shield"}
            ) or step.until.criterion == CompletionCriterion.MEASURED_RATE_CPS_AT_MOST:
                raise PlanValidationError(
                    f"step {index} completion condition does not apply to "
                    f"{candidate.action_type}"
                )
        if step.max_attempts > 1:
            warnings.append(
                f"Bounded repeated physical action: {candidate.label} "
                f"(up to {step.max_attempts} attempts)"
            )
        warnings.append(f"Physical action: {candidate.label}")
    return ValidatedCommandPlan(
        plan=plan,
        requires_confirmation=any(
            step.command in _CONFIRMATION_COMMANDS for step in plan.steps
        ),
        warnings=tuple(warnings),
    )
