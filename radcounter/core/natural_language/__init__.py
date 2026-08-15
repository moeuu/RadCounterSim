"""Safe natural-language command interpretation for RadCounterSim."""

from .client import (
    CommandInterpretationError,
    CommandInterpreter,
    OpenAICompatibleCommandInterpreter,
    normalize_operator_instruction,
)
from .models import (
    AvailableAction,
    CommandContext,
    CommandName,
    CommandPlan,
    CommandStep,
    CompletionCriterion,
    PlanValidationError,
    StepCompletionCondition,
    ValidatedCommandPlan,
    validate_command_plan,
)
from .runtime import (
    LlamaCppRuntime,
    LlamaCppRuntimeConfig,
    RuntimeAssetMissingError,
    RuntimeStatus,
)

__all__ = [
    "AvailableAction",
    "CompletionCriterion",
    "CommandContext",
    "CommandInterpretationError",
    "CommandInterpreter",
    "CommandName",
    "CommandPlan",
    "CommandStep",
    "LlamaCppRuntime",
    "LlamaCppRuntimeConfig",
    "OpenAICompatibleCommandInterpreter",
    "PlanValidationError",
    "RuntimeAssetMissingError",
    "RuntimeStatus",
    "StepCompletionCondition",
    "ValidatedCommandPlan",
    "normalize_operator_instruction",
    "validate_command_plan",
]
