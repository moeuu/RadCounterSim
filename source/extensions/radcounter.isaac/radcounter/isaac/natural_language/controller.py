"""Application controller joining local inference to allowlisted host commands."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from radcounter.core.natural_language import (
    CommandContext,
    CommandInterpreter,
    CommandStep,
    CompletionCriterion,
    LlamaCppRuntime,
    OpenAICompatibleCommandInterpreter,
    ValidatedCommandPlan,
    validate_command_plan,
)


class NaturalLanguageCommandHost(Protocol):
    def natural_language_context(self) -> CommandContext: ...

    async def execute_natural_language_step(
        self, step: CommandStep
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class CommandSubmission:
    validated: ValidatedCommandPlan
    executed: bool
    results: tuple[Mapping[str, object], ...] = ()


class WorkflowExecutionError(RuntimeError):
    """A bounded workflow failed after one or more auditable operations."""

    def __init__(
        self,
        message: str,
        *,
        results: tuple[Mapping[str, object], ...],
        step_index: int,
        attempt: int,
    ) -> None:
        super().__init__(message)
        self.results = results
        self.step_index = step_index
        self.attempt = attempt


class NaturalLanguageCommandController:
    """Interpret, confirm, execute, and audit one bounded instruction at a time."""

    def __init__(
        self,
        host: NaturalLanguageCommandHost,
        *,
        runtime: LlamaCppRuntime | None = None,
        interpreter: CommandInterpreter | None = None,
        audit_path: str | Path | None = None,
    ) -> None:
        self.host = host
        self.runtime = runtime if runtime is not None else LlamaCppRuntime()
        self.interpreter = interpreter
        self.audit_path = None if audit_path is None else Path(audit_path)
        self.pending: ValidatedCommandPlan | None = None
        self._pending_instruction = ""
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    async def _ensure_interpreter(self) -> CommandInterpreter:
        if self.interpreter is not None:
            return self.interpreter
        endpoint = await asyncio.to_thread(self.runtime.start)
        self.interpreter = OpenAICompatibleCommandInterpreter(
            endpoint,
            model=self.runtime.config.model_alias,
        )
        return self.interpreter

    async def submit(self, instruction: str) -> CommandSubmission:
        if self._busy:
            raise RuntimeError("another natural-language command is already running")
        self._busy = True
        try:
            interpreter = await self._ensure_interpreter()
            context = self.host.natural_language_context()
            plan = await interpreter.interpret(instruction, context)
            validated = validate_command_plan(plan, context)
            if validated.requires_confirmation:
                self.pending = validated
                self._pending_instruction = instruction
                self._write_audit(instruction, validated, "awaiting_confirmation", ())
                return CommandSubmission(validated, False)
            results = await self._execute(validated)
            self._write_audit(instruction, validated, "completed", results)
            return CommandSubmission(validated, True, results)
        except Exception as error:
            self._write_failure(instruction, error)
            raise
        finally:
            self._busy = False

    async def confirm(self) -> CommandSubmission:
        if self._busy:
            raise RuntimeError("another natural-language command is already running")
        if self.pending is None:
            raise RuntimeError("there is no command awaiting confirmation")
        self._busy = True
        validated = self.pending
        instruction = self._pending_instruction
        self.pending = None
        self._pending_instruction = ""
        try:
            # Revalidate against the live scene immediately before execution.
            validated = validate_command_plan(validated.plan, self.host.natural_language_context())
            results = await self._execute(validated)
            self._write_audit(instruction, validated, "completed", results)
            return CommandSubmission(validated, True, results)
        except Exception as error:
            self._write_failure(instruction, error)
            raise
        finally:
            self._busy = False

    def cancel(self) -> None:
        if self.pending is not None:
            self._write_audit(
                self._pending_instruction,
                self.pending,
                "cancelled",
                (),
            )
        self.pending = None
        self._pending_instruction = ""

    async def _execute(
        self, validated: ValidatedCommandPlan
    ) -> tuple[Mapping[str, object], ...]:
        results: list[Mapping[str, object]] = []
        for step_index, step in enumerate(validated.plan.steps, start=1):
            initial_activity_bq: float | None = None
            for attempt in range(1, step.max_attempts + 1):
                # Earlier physical steps and earlier attempts may change
                # candidates, feasibility, action type, or capabilities.
                live_step_plan = validated.plan.model_copy(update={"steps": (step,)})
                try:
                    validate_command_plan(
                        live_step_plan, self.host.natural_language_context()
                    )
                    raw_result = await self.host.execute_natural_language_step(step)
                except Exception as error:
                    raise WorkflowExecutionError(
                        f"workflow step {step_index} attempt {attempt} failed: "
                        f"{type(error).__name__}: {error}",
                        results=tuple(results),
                        step_index=step_index,
                        attempt=attempt,
                    ) from error
                result = dict(raw_result)
                result["workflow_step"] = step_index
                result["attempt"] = attempt
                result["max_attempts"] = step.max_attempts
                if step.until is not None:
                    value, initial_activity_bq = self._completion_value(
                        step.until.criterion,
                        result,
                        initial_activity_bq=initial_activity_bq,
                    )
                    condition_met = self._condition_met(
                        step.until.criterion,
                        value,
                        step.until.threshold,
                    )
                    result["completion_condition"] = {
                        "criterion": step.until.criterion.value,
                        "threshold": step.until.threshold,
                        "observed": value,
                        "met": condition_met,
                    }
                    results.append(result)
                    if condition_met:
                        break
                else:
                    results.append(result)
        return tuple(results)

    @staticmethod
    def _public_metric(result: Mapping[str, object], *path: str) -> float:
        value: object = result
        for component in path:
            if not isinstance(value, Mapping) or component not in value:
                dotted = ".".join(path)
                raise RuntimeError(
                    f"public result does not provide completion metric: {dotted}"
                )
            value = value[component]
        if not isinstance(value, (int, float)):
            raise RuntimeError(
                f"public completion metric is not numeric: {'.'.join(path)}"
            )
        return float(value)

    @classmethod
    def _completion_value(
        cls,
        criterion: CompletionCriterion,
        result: Mapping[str, object],
        *,
        initial_activity_bq: float | None,
    ) -> tuple[float, float | None]:
        if criterion == CompletionCriterion.DECONTAMINATION_REMOVED_FRACTION_AT_LEAST:
            before = cls._public_metric(
                result, "public_details", "motion_audit", "activity_before_bq"
            )
            after = cls._public_metric(
                result, "public_details", "motion_audit", "activity_after_bq"
            )
            baseline = before if initial_activity_bq is None else initial_activity_bq
            removed_fraction = 1.0 - after / baseline if baseline > 0.0 else 1.0
            return (max(0.0, min(1.0, removed_fraction)), baseline)
        if criterion == CompletionCriterion.DECONTAMINATION_COVERAGE_FRACTION_AT_LEAST:
            return (
                cls._public_metric(
                    result, "public_details", "motion_audit", "coverage_fraction"
                ),
                initial_activity_bq,
            )
        if criterion == CompletionCriterion.DECONTAMINATION_REMAINING_FRACTION_AT_MOST:
            before = cls._public_metric(
                result, "public_details", "motion_audit", "activity_before_bq"
            )
            after = cls._public_metric(
                result, "public_details", "motion_audit", "activity_after_bq"
            )
            baseline = before if initial_activity_bq is None else initial_activity_bq
            return (after / baseline if baseline > 0.0 else 0.0, baseline)
        if criterion == CompletionCriterion.MEASURED_RATE_CPS_AT_MOST:
            return (
                cls._public_metric(result, "maximum_measured_rate_cps"),
                initial_activity_bq,
            )
        if criterion == CompletionCriterion.SHIELD_PLACEMENT_ERROR_M_AT_MOST:
            return (
                cls._public_metric(
                    result, "public_details", "motion_audit", "placement_error_m"
                ),
                initial_activity_bq,
            )
        raise RuntimeError(f"unsupported completion criterion: {criterion}")

    @staticmethod
    def _condition_met(
        criterion: CompletionCriterion,
        observed: float,
        threshold: float,
    ) -> bool:
        at_least = {
            CompletionCriterion.DECONTAMINATION_REMOVED_FRACTION_AT_LEAST,
            CompletionCriterion.DECONTAMINATION_COVERAGE_FRACTION_AT_LEAST,
        }
        return observed >= threshold if criterion in at_least else observed <= threshold

    def _append_audit(self, payload: Mapping[str, object]) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
            stream.write("\n")

    def _write_audit(
        self,
        instruction: str,
        validated: ValidatedCommandPlan,
        status: str,
        results: tuple[Mapping[str, object], ...],
    ) -> None:
        self._append_audit(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "instruction": instruction,
                "plan": validated.plan.model_dump(mode="json"),
                "requires_confirmation": validated.requires_confirmation,
                "warnings": list(validated.warnings),
                "status": status,
                "results": list(results),
            }
        )

    def _write_failure(self, instruction: str, error: Exception) -> None:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "instruction": instruction,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if isinstance(error, WorkflowExecutionError):
            payload.update(
                {
                    "failed_step": error.step_index,
                    "failed_attempt": error.attempt,
                    "partial_results": list(error.results),
                }
            )
        self._append_audit(payload)

    def shutdown(self) -> None:
        self.cancel()
        self.runtime.stop()
