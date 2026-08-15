"""OpenAI-compatible client used with a private local llama.cpp server."""

from __future__ import annotations

import asyncio
import copy
import json
import re
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from pydantic import ValidationError

from .models import AvailableAction, CommandContext, CommandPlan


class CommandInterpretationError(RuntimeError):
    """Raised when local inference fails or returns an invalid command plan."""


class CommandInterpreter(Protocol):
    async def interpret(self, instruction: str, context: CommandContext) -> CommandPlan: ...


_SYSTEM_PROMPT = """You translate English operator instructions into a
RadCounterSim command plan. Return only JSON matching the supplied schema. Never invent
action IDs, robot IDs, paths, commands, or coordinates. Use execute_candidate only with
an exact available_actions.action_id. Prefer the shortest plan that completes the request.
Write the summary in English. If the request is ambiguous or impossible
using the listed capabilities, return a show_status step and explain the limitation in summary.
Never claim an operation was executed; you only propose it.
An execute_candidate whose action_type is measure only navigates to a measurement station.
If the operator also asks to measure, count, or take a reading, add a separate measure step.
An available place_shield action already contains host-derived pickup, grasp, route, and
placement coordinates. When the operator asks to pick up and place the lead shield between
the source and protected area, select a feasible place_shield action instead of rejecting it
for not exposing those coordinates in the public context.
Available shield labels and placement_fraction values identify host-planned positions along
the source-to-protected-area line. Preserve every explicitly requested placement and
repositioning percentage as an ordered execute_candidate step. A later candidate for the same
physical shield is a host-controlled move, not a request for model-generated coordinates.
Each decontaminate action owns its approach, full irregular-surface serpentine raster, and
retreat. If the operator requests multiple passes or says "until" with a maximum number of
passes, use max_attempts (never more than five) and an applicable until criterion. Fractions in
completion thresholds are JSON fractions: 70 percent is 0.70. The host evaluates conditions
only from public contact, coverage, activity, placement-error, or measured-rate results.
For "all", "each", or multiple explicitly named targets, emit one ordered candidate step for
every distinct feasible target instead of silently choosing only the first.
Use return_measurement_robot when the operator asks the measurement robot to return to its
starting, initial, original, or home position. Multi-step requests are supported: preserve the
operator's order and emit as many steps as needed, up to twenty-four. The host confirms the complete
workflow once and revalidates the live scene after every step before continuing. Do not invent
conditional results. Bounded until conditions stop a repeated step early; they never authorize
an unbounded loop or an action outside the listed capabilities.
"""


def _inlined_command_schema() -> dict[str, object]:
    """Inline Pydantic references for reliable llama.cpp grammar generation."""

    schema = copy.deepcopy(CommandPlan.model_json_schema())
    definitions = schema.pop("$defs", {})

    def resolve(value: object) -> object:
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if not isinstance(value, dict):
            return value
        reference = value.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            name = reference.rsplit("/", 1)[-1]
            target = definitions.get(name)
            if target is None:
                raise RuntimeError(f"unresolved command schema reference: {reference}")
            merged = copy.deepcopy(target)
            merged.update({key: item for key, item in value.items() if key != "$ref"})
            return resolve(merged)
        return {key: resolve(item) for key, item in value.items()}

    resolved = resolve(schema)
    assert isinstance(resolved, dict)
    return resolved


def _instruction_language(instruction: str) -> str:
    """Return the only language accepted by the English operator interface."""

    del instruction
    return "en"


def normalize_operator_instruction(instruction: str) -> str:
    """Remove accidental triple-or-more repetitions from one-line UI input.

    Some IME/paste interactions can append the complete composition more than once.
    Two repetitions are left intact because they can be an intentional request; three
    or more exact consecutive copies are treated as input duplication.
    """

    normalized = re.sub(r"\s{2,}>\s*", " ", instruction.strip())
    # Paste/IME glitches can repeat a long instruction with a slightly different
    # final copy (for example, after a stripped Markdown quote marker). Detect
    # three-or-more recurrences by a stable leading anchor before exact matching.
    if len(normalized) >= 96:
        anchor = normalized[:48]
        starts = [match.start() for match in re.finditer(re.escape(anchor), normalized)]
        if len(starts) >= 3 and starts[0] == 0:
            return normalized[: starts[1]].strip()
    for unit_length in range(4, len(normalized) // 3 + 1):
        if len(normalized) % unit_length:
            continue
        repetition_count = len(normalized) // unit_length
        if repetition_count >= 3 and normalized == normalized[:unit_length] * repetition_count:
            return normalized[:unit_length].strip()
    return normalized


def _requested_measurement(instruction: str) -> bool:
    lowered = instruction.casefold()
    return any(
        token in lowered
        for token in (
            "measure",
            "measurement",
            "take a reading",
            "count radiation",
        )
    )


def _requested_status(instruction: str) -> bool:
    lowered = instruction.casefold()
    return any(
        token in lowered
        for token in (
            "show status",
            "show the current status",
            "display status",
            "display the current status",
            "report status",
        )
    )


def _requested_duration_s(instruction: str) -> float | None:
    match = re.search(
        r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s\b)",
        instruction,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    value = float(match.group(1))
    return value if 0.05 <= value <= 120.0 else None


def _requested_return_to_start(instruction: str) -> bool:
    lowered = instruction.casefold()
    english_return = bool(
        re.search(r"\b(?:return|go|come|send|move)\b.*\b(?:back|home)\b", lowered)
        or re.search(
            r"\breturn\b.*\b(?:starting|start|initial|original|home)\b.*\b(?:position|point|location)?\b",
            lowered,
        )
    )
    return english_return


def _requested_shield_placement(instruction: str) -> bool:
    lowered = instruction.casefold()
    mentions_shield = "shield" in lowered
    requests_manipulation = any(
        token in lowered
        for token in (
            "pick up",
            "pickup",
            "place",
            "position",
            "manipulator",
            "gripper",
        )
    )
    return mentions_shield and requests_manipulation


def _requested_decontamination(instruction: str) -> bool:
    lowered = instruction.casefold()
    return any(token in lowered for token in ("decontaminate", "decontamination"))


def _requested_max_attempts(instruction: str) -> int | None:
    """Return an explicit bounded pass/attempt count, never a time value."""

    patterns = (
        r"\b(?:up\s+to|max(?:imum)?(?:\s+of)?)\s*([1-5])\s*(?:times?|passes?|attempts?)\b",
        r"\b([1-5])\s*(?:times?|passes?|attempts?)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, instruction, flags=re.IGNORECASE)
        if match is not None:
            return int(match.group(1))
    lowered = instruction.casefold()
    if any(token in lowered for token in ("again", "one more pass")):
        return 2
    return None


def _percent_after_labels(instruction: str, labels: tuple[str, ...]) -> float | None:
    label_pattern = "|".join(re.escape(label) for label in labels)
    patterns = (
        rf"(?:{label_pattern})[^\d%]{{0,28}}(\d{{1,3}}(?:\.\d+)?)\s*%",
        rf"(\d{{1,3}}(?:\.\d+)?)\s*%[^,.]{{0,28}}(?:{label_pattern})",
    )
    for pattern in patterns:
        match = re.search(pattern, instruction, flags=re.IGNORECASE)
        if match is not None:
            value = float(match.group(1)) / 100.0
            if 0.0 <= value <= 1.0:
                return value
    return None


def _requested_decontamination_condition(
    instruction: str,
) -> tuple[str, float] | None:
    removed = _percent_after_labels(
        instruction,
        ("removal fraction", "removed fraction"),
    )
    if removed is not None:
        return "decontamination_removed_fraction_at_least", removed
    remaining = _percent_after_labels(
        instruction,
        ("remaining fraction", "residual fraction"),
    )
    if remaining is not None:
        return "decontamination_remaining_fraction_at_most", remaining
    coverage = _percent_after_labels(
        instruction,
        ("coverage", "surface coverage"),
    )
    if coverage is not None:
        return "decontamination_coverage_fraction_at_least", coverage
    return None


def _requested_measurement_condition(instruction: str) -> float | None:
    match = re.search(
        r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:cps|counts?\s*/\s*s)\s*"
        r"(?:or\s+less|or\s+lower|at\s+most|below)?",
        instruction,
        flags=re.IGNORECASE,
    )
    return None if match is None else float(match.group(1))


def _requests_all_targets(instruction: str, action_type: str) -> bool:
    lowered = instruction.casefold()
    all_tokens = ("all ", "each ", "every ")
    if not any(token in lowered for token in all_tokens):
        return False
    type_tokens = {
        "decontaminate": ("decontamin", "surface"),
        "measure": ("measurement", "station"),
        "shield": ("shield", "panel"),
    }[action_type]
    return any(token in lowered for token in type_tokens)


def _requests_multiple_shields(instruction: str) -> bool:
    lowered = instruction.casefold()
    return any(
        token in lowered
        for token in (
            "all shields",
            "both shields",
            "multiple shields",
            "two shields",
            "two panels",
        )
    )


def _requested_shield_fractions(
    instruction: str,
    context: CommandContext,
) -> tuple[float, ...]:
    available = sorted(
        {
            round(float(action.placement_fraction), 6)
            for action in context.available_actions
            if action.action_type in {"place_shield", "move_shield"}
            and action.placement_fraction is not None
        }
    )
    if not available:
        return ()
    requested: list[float] = []
    for match in re.finditer(r"(\d{1,3}(?:\.\d+)?)\s*%", instruction):
        fraction = float(match.group(1)) / 100.0
        closest = min(available, key=lambda item: abs(item - fraction))
        if abs(closest - fraction) <= 0.005:
            requested.append(closest)
    return tuple(requested)


def _requested_object_manipulation(instruction: str) -> bool:
    """Distinguish moving through a facility from moving a physical object."""

    lowered = instruction.casefold()
    return any(
        token in lowered
        for token in (
            "drum",
            "move the obstacle",
            "move an object",
            "remove the object",
            "dispose",
            "relocate the object",
        )
    )


def _normalize_plan(
    plan: CommandPlan,
    instruction: str,
    context: CommandContext,
) -> CommandPlan:
    """Repair harmless small-model omissions using explicit operator wording."""

    language = _instruction_language(instruction)
    by_id = {action.action_id: action for action in context.available_actions}
    from .models import (
        CommandName,
        CommandStep,
        CompletionCriterion,
        StepCompletionCondition,
    )

    steps: list[CommandStep] = []
    for step in plan.steps:
        candidate_id = step.candidate_id
        if (
            step.command == CommandName.EXECUTE_CANDIDATE
            and candidate_id not in by_id
            and candidate_id in context.capabilities
        ):
            try:
                built_in = CommandName(candidate_id)
            except ValueError:
                pass
            else:
                if built_in != CommandName.EXECUTE_CANDIDATE:
                    step = CommandStep(command=built_in, duration_s=step.duration_s)
        steps.append(step)
    repaired_shield_placement = False
    if _requested_shield_placement(instruction):
        feasible_shields = sorted(
            (
                action
                for action in context.available_actions
                if action.feasible and action.action_type in {"place_shield", "move_shield"}
            ),
            key=lambda action: action.action_id,
        )
        # Scene geometry can invalidate one of several equivalent shield-line
        # candidates. Small local models commonly select the first identifier;
        # substitute the first host-verified safe candidate before validation.
        if feasible_shields:
            safe_shield_id = feasible_shields[0].action_id
            steps = [
                step.model_copy(update={"candidate_id": safe_shield_id})
                if step.command == CommandName.EXECUTE_CANDIDATE
                and step.candidate_id in by_id
                and by_id[step.candidate_id].action_type in {"place_shield", "move_shield"}
                and not by_id[step.candidate_id].feasible
                else step
                for step in steps
            ]
        selected_shield = any(
            step.command == CommandName.EXECUTE_CANDIDATE
            and step.candidate_id in by_id
            and by_id[step.candidate_id].action_type in {"place_shield", "move_shield"}
            for step in steps
        )
        if not selected_shield and feasible_shields:
            steps = [step for step in steps if step.command != CommandName.SHOW_STATUS]
            steps.insert(
                0,
                CommandStep(
                    command=CommandName.EXECUTE_CANDIDATE,
                    candidate_id=feasible_shields[0].action_id,
                ),
            )
            repaired_shield_placement = True
    if _requested_decontamination(instruction):
        feasible_decon = sorted(
            (
                action
                for action in context.available_actions
                if action.feasible and action.action_type == "decontaminate"
            ),
            key=lambda action: action.action_id,
        )
        selected_decon = any(
            step.command == CommandName.EXECUTE_CANDIDATE
            and step.candidate_id in by_id
            and by_id[step.candidate_id].action_type == "decontaminate"
            and by_id[step.candidate_id].feasible
            for step in steps
        )
        if not selected_decon and feasible_decon:
            steps = [
                step
                for step in steps
                if not (
                    step.command == CommandName.SHOW_STATUS
                    or (
                        step.command == CommandName.EXECUTE_CANDIDATE
                        and step.candidate_id in by_id
                        and by_id[step.candidate_id].action_type == "decontaminate"
                    )
                )
            ]
            steps.insert(
                0,
                CommandStep(
                    command=CommandName.EXECUTE_CANDIDATE,
                    candidate_id=feasible_decon[0].action_id,
                ),
            )
    if not _requested_object_manipulation(instruction):
        # A small local model can interpret "move through the corridor" as a
        # request to move an unrelated prop. Candidate routes already contain
        # robot navigation, so unmentioned object manipulation is never needed.
        object_action_types = {"move_object", "remove_object"}
        steps = [
            step
            for step in steps
            if not (
                step.command == CommandName.EXECUTE_CANDIDATE
                and step.candidate_id in by_id
                and by_id[step.candidate_id].action_type in object_action_types
            )
        ]
    requested_duration_s = _requested_duration_s(instruction)
    if requested_duration_s is not None:
        steps = [
            step.model_copy(update={"duration_s": requested_duration_s})
            if step.command == CommandName.MEASURE and step.duration_s is None
            else step
            for step in steps
        ]
    if _requested_measurement(instruction):
        selected_measurement_navigation = any(
            step.command == CommandName.EXECUTE_CANDIDATE
            and step.candidate_id in by_id
            and by_id[step.candidate_id].action_type == "measure"
            for step in steps
        )
        available_measurements = [
            action
            for action in context.available_actions
            if action.feasible and action.action_type == "measure"
        ]
        lowered = instruction.casefold()
        wants_protected = any(
            token in lowered
            for token in (
                "protected area",
                "protected zone",
                "protected measurement station",
            )
        )
        wants_remote = any(token in lowered for token in ("remote room", "decontamination room"))
        if (
            not selected_measurement_navigation
            and available_measurements
            and (wants_protected or wants_remote)
        ):
            preferred_token = "protected" if wants_protected else "remote"
            available_measurements.sort(
                key=lambda action: (
                    preferred_token
                    not in " ".join(
                        (
                            action.action_id,
                            action.target or "",
                            action.target_label or "",
                        )
                    ).casefold()
                )
            )
            measurement_navigation = CommandStep(
                command=CommandName.EXECUTE_CANDIDATE,
                candidate_id=available_measurements[0].action_id,
            )
            measurement_index = next(
                (index for index, step in enumerate(steps) if step.command == CommandName.MEASURE),
                len(steps),
            )
            steps.insert(measurement_index, measurement_navigation)
    navigates_for_measurement = any(
        step.command.value == "execute_candidate"
        and step.candidate_id in by_id
        and by_id[step.candidate_id].action_type == "measure"
        for step in steps
    )
    has_measurement = any(step.command.value == "measure" for step in steps)
    executes_candidate = any(step.command == CommandName.EXECUTE_CANDIDATE for step in steps)
    if executes_candidate and _requested_measurement(instruction) and not has_measurement:
        steps.append(
            CommandStep(
                command=CommandName.MEASURE,
                duration_s=requested_duration_s,
            )
        )
    verification_action_types = {
        by_id[step.candidate_id].action_type
        for step in steps
        if step.command == CommandName.EXECUTE_CANDIDATE and step.candidate_id in by_id
    }
    verification_builtins_only = all(
        step.command
        in {
            CommandName.EXECUTE_CANDIDATE,
            CommandName.MEASURE,
            CommandName.RETURN_MEASUREMENT_ROBOT,
            CommandName.SHOW_STATUS,
        }
        for step in steps
    )
    repeated_workflow_requested = any(
        token in instruction.casefold()
        for token in (
            "again",
            "repeat",
            "re-measure",
            "remeasure",
            "until",
        )
    )
    if (
        _requested_decontamination(instruction)
        and _requested_measurement(instruction)
        and verification_builtins_only
        and verification_action_types <= {"decontaminate", "measure"}
        and not repeated_workflow_requested
    ):
        # For this common verification workflow, collapse small-model
        # repetitions into the three physical semantics the operator asked for:
        # decontaminate, navigate to the requested detector, then integrate.
        decon_steps = [
            step
            for step in steps
            if step.command == CommandName.EXECUTE_CANDIDATE
            and step.candidate_id in by_id
            and by_id[step.candidate_id].action_type == "decontaminate"
            and by_id[step.candidate_id].feasible
        ]
        measurement_steps = [
            step
            for step in steps
            if step.command == CommandName.EXECUTE_CANDIDATE
            and step.candidate_id in by_id
            and by_id[step.candidate_id].action_type == "measure"
            and by_id[step.candidate_id].feasible
        ]
        wants_protected = any(
            token in instruction.casefold()
            for token in (
                "protected area",
                "protected zone",
                "protected measurement station",
            )
        )
        wants_remote_room = any(
            token in instruction.casefold()
            for token in (
                "remote room",
                "decontamination room",
            )
        )
        available_measurements = [
            action
            for action in context.available_actions
            if action.feasible and action.action_type == "measure"
        ]
        if wants_protected:
            measurement_steps.sort(
                key=lambda step: "protected" not in str(step.candidate_id).casefold()
            )
            available_measurements.sort(
                key=lambda action: "protected" not in action.action_id.casefold()
            )
        elif wants_remote_room:
            measurement_steps.sort(
                key=lambda step: "remote" not in str(step.candidate_id).casefold()
            )
            available_measurements.sort(
                key=lambda action: "remote" not in action.action_id.casefold()
            )
        preferred_available = available_measurements[0] if available_measurements else None
        if (
            measurement_steps
            and preferred_available is not None
            and (
                (wants_protected and "protected" not in str(measurement_steps[0].candidate_id))
                or (wants_remote_room and "remote" not in str(measurement_steps[0].candidate_id))
            )
        ):
            measurement_steps = [
                CommandStep(
                    command=CommandName.EXECUTE_CANDIDATE,
                    candidate_id=preferred_available.action_id,
                )
            ]
        if not measurement_steps and available_measurements:
            measurement_steps = [
                CommandStep(
                    command=CommandName.EXECUTE_CANDIDATE,
                    candidate_id=available_measurements[0].action_id,
                )
            ]
        measurement_step = next(
            (step for step in steps if step.command == CommandName.MEASURE),
            CommandStep(
                command=CommandName.MEASURE,
                duration_s=requested_duration_s,
            ),
        )
        return_step = next(
            (step for step in steps if step.command == CommandName.RETURN_MEASUREMENT_ROBOT),
            None,
        )
        status_step = next(
            (step for step in steps if step.command == CommandName.SHOW_STATUS),
            None,
        )
        steps = [
            *decon_steps[:1],
            *measurement_steps[:1],
            measurement_step,
            *(() if return_step is None else (return_step,)),
            *(() if status_step is None else (status_step,)),
        ]
    if navigates_for_measurement and _requested_return_to_start(instruction):
        # Small models sometimes translate "return to the starting position" as
        # another measurement-station candidate. Once measurement has happened,
        # such a candidate is not a valid representation of home; the dedicated
        # allowlisted command is deterministic and does not expose coordinates.
        after_measurement = False
        measurement_navigation_seen = False
        repaired_steps: list[CommandStep] = []
        trailing_status_steps: list[CommandStep] = []
        for step in steps:
            if step.command == CommandName.MEASURE:
                after_measurement = True
            if step.command == CommandName.RETURN_MEASUREMENT_ROBOT:
                continue
            if step.command == CommandName.SHOW_STATUS:
                trailing_status_steps.append(step)
                continue
            if step.command == CommandName.EXECUTE_CANDIDATE and step.candidate_id in by_id:
                is_measurement_navigation = by_id[step.candidate_id].action_type == "measure"
                if is_measurement_navigation:
                    if measurement_navigation_seen and (
                        after_measurement or _requested_measurement(instruction)
                    ):
                        continue
                    measurement_navigation_seen = True
            repaired_steps.append(step)
        if len(repaired_steps) >= 24:
            repaired_steps = repaired_steps[:23]
        steps = [
            *repaired_steps,
            CommandStep(command=CommandName.RETURN_MEASUREMENT_ROBOT),
            *trailing_status_steps[:1],
        ]

    # Expand explicit "all surfaces" requests from the allowlisted live
    # candidates.  The model never receives or invents raster coordinates; each
    # host candidate owns its approach, irregular-surface scan, and retreat.
    if _requests_all_targets(instruction, "decontaminate"):
        feasible_by_target: dict[str, AvailableAction] = {}
        for action in sorted(context.available_actions, key=lambda item: item.action_id):
            if action.feasible and action.action_type == "decontaminate":
                feasible_by_target.setdefault(action.target or action.action_id, action)
        present = {
            step.candidate_id
            for step in steps
            if step.command == CommandName.EXECUTE_CANDIDATE
            and step.candidate_id in by_id
            and by_id[step.candidate_id].action_type == "decontaminate"
        }
        missing = [
            CommandStep(
                command=CommandName.EXECUTE_CANDIDATE,
                candidate_id=action.action_id,
            )
            for action in feasible_by_target.values()
            if action.action_id not in present
        ]
        if missing:
            decon_indices = [
                index
                for index, step in enumerate(steps)
                if step.command == CommandName.EXECUTE_CANDIDATE
                and step.candidate_id in by_id
                and by_id[step.candidate_id].action_type == "decontaminate"
            ]
            insertion = decon_indices[-1] + 1 if decon_indices else 0
            steps[insertion:insertion] = missing

    decon_condition = _requested_decontamination_condition(instruction)
    requested_attempts = _requested_max_attempts(instruction)
    if decon_condition is not None or requested_attempts is not None:
        condition = (
            None
            if decon_condition is None
            else StepCompletionCondition(
                criterion=CompletionCriterion(decon_condition[0]),
                threshold=decon_condition[1],
            )
        )
        attempts = requested_attempts or 3
        repaired: list[CommandStep] = []
        repeated_decon_ids: set[str] = set()
        for step in steps:
            is_decon = (
                step.command == CommandName.EXECUTE_CANDIDATE
                and step.candidate_id in by_id
                and by_id[step.candidate_id].action_type == "decontaminate"
            )
            if not is_decon:
                repaired.append(step)
                continue
            assert step.candidate_id is not None
            if step.candidate_id in repeated_decon_ids:
                continue
            repeated_decon_ids.add(step.candidate_id)
            repaired.append(
                step.model_copy(
                    update={
                        "max_attempts": max(step.max_attempts, attempts),
                        "until": condition if condition is not None else step.until,
                    }
                )
            )
        steps = repaired

    # An explicit series of source-line percentages means place and then move
    # the same shield through those host-planned positions.  A request for
    # multiple panels instead selects one distinct target prim per panel.
    shield_fractions = _requested_shield_fractions(instruction, context)
    shield_actions = [
        action
        for action in context.available_actions
        if action.feasible and action.action_type in {"place_shield", "move_shield"}
    ]
    if shield_actions and (shield_fractions or _requests_multiple_shields(instruction)):
        existing_shield_steps = [
            step
            for step in steps
            if step.command == CommandName.EXECUTE_CANDIDATE
            and step.candidate_id in by_id
            and by_id[step.candidate_id].action_type in {"place_shield", "move_shield"}
        ]
        preferred_target = None
        if existing_shield_steps:
            preferred_target = by_id[existing_shield_steps[0].candidate_id].target
        if preferred_target is None:
            preferred_target = sorted(shield_actions, key=lambda action: action.action_id)[0].target

        desired_actions: list[AvailableAction] = []
        if _requests_multiple_shields(instruction):
            targets: list[str] = []
            for action in sorted(shield_actions, key=lambda item: item.action_id):
                key = action.target or action.action_id
                if key not in targets:
                    targets.append(key)
            for index, target in enumerate(targets):
                fraction = (
                    shield_fractions[min(index, len(shield_fractions) - 1)]
                    if shield_fractions
                    else (0.35 if index % 2 == 0 else 0.65)
                )
                options = [
                    action
                    for action in shield_actions
                    if (action.target or action.action_id) == target
                ]
                desired_actions.append(
                    min(
                        options,
                        key=lambda action: abs(float(action.placement_fraction or 0.0) - fraction),
                    )
                )
        else:
            options = [action for action in shield_actions if action.target == preferred_target]
            for fraction in shield_fractions:
                desired_actions.append(
                    min(
                        options,
                        key=lambda action: abs(float(action.placement_fraction or 0.0) - fraction),
                    )
                )

        desired_steps = [
            CommandStep(
                command=CommandName.EXECUTE_CANDIDATE,
                candidate_id=action.action_id,
            )
            for action in desired_actions
        ]
        shield_indices = [
            index for index, step in enumerate(steps) if step in existing_shield_steps
        ]
        insertion = shield_indices[0] if shield_indices else 0
        steps = [step for step in steps if step not in existing_shield_steps]
        lowered = instruction.casefold()
        decon_position = min(
            (lowered.find(token) for token in ("decontamin",) if token in lowered),
            default=-1,
        )
        shield_position = min(
            (lowered.find(token) for token in ("shield",) if token in lowered),
            default=-1,
        )
        if decon_position >= 0 and shield_position > decon_position:
            decon_indices = [
                index
                for index, step in enumerate(steps)
                if step.command == CommandName.EXECUTE_CANDIDATE
                and step.candidate_id in by_id
                and by_id[step.candidate_id].action_type == "decontaminate"
            ]
            if decon_indices:
                insertion = decon_indices[-1] + 1
        steps[insertion:insertion] = desired_steps

    # "All/each station" is a facility tour: navigate to each distinct
    # allowlisted station and integrate there before any requested return-home
    # or status step. Routes are regenerated from the live robot pose per leg.
    if _requests_all_targets(instruction, "measure") and _requested_measurement(instruction):
        measurement_actions: list[AvailableAction] = []
        seen_targets: set[str] = set()
        for action in sorted(context.available_actions, key=lambda item: item.action_id):
            if not action.feasible or action.action_type != "measure":
                continue
            target = action.target or action.action_id
            if target in seen_targets:
                continue
            seen_targets.add(target)
            measurement_actions.append(action)
        if measurement_actions:
            old_measurement_steps = [
                step
                for step in steps
                if step.command == CommandName.MEASURE
                or (
                    step.command == CommandName.EXECUTE_CANDIDATE
                    and step.candidate_id in by_id
                    and by_id[step.candidate_id].action_type == "measure"
                )
            ]
            old_indices = [
                index for index, step in enumerate(steps) if step in old_measurement_steps
            ]
            steps = [step for step in steps if step not in old_measurement_steps]
            trailing_index = next(
                (
                    index
                    for index, step in enumerate(steps)
                    if step.command
                    in {CommandName.RETURN_MEASUREMENT_ROBOT, CommandName.SHOW_STATUS}
                ),
                len(steps),
            )
            insertion = min(old_indices[0], trailing_index) if old_indices else trailing_index
            expanded_measurements: list[CommandStep] = []
            for action in measurement_actions:
                expanded_measurements.extend(
                    (
                        CommandStep(
                            command=CommandName.EXECUTE_CANDIDATE,
                            candidate_id=action.action_id,
                        ),
                        CommandStep(
                            command=CommandName.MEASURE,
                            duration_s=requested_duration_s,
                        ),
                    )
                )
            steps[insertion:insertion] = expanded_measurements

    measurement_threshold = _requested_measurement_condition(instruction)
    if measurement_threshold is not None:
        attempts = requested_attempts or 3
        steps = [
            step.model_copy(
                update={
                    "max_attempts": max(step.max_attempts, attempts),
                    "until": StepCompletionCondition(
                        criterion=CompletionCriterion.MEASURED_RATE_CPS_AT_MOST,
                        threshold=measurement_threshold,
                    ),
                }
            )
            if step.command == CommandName.MEASURE
            else step
            for step in steps
        ]

    if _requested_status(instruction) and not any(
        step.command == CommandName.SHOW_STATUS for step in steps
    ):
        steps.append(CommandStep(command=CommandName.SHOW_STATUS))

    if len(steps) > 24:
        # The deterministic host limit is intentionally smaller than the
        # candidate context.  Refuse silent truncation by returning a status-only
        # plan that explains the bounded-workflow limit.
        steps = [CommandStep(command=CommandName.SHOW_STATUS)]
        summary = "The requested workflow exceeds the 24-step safety limit; split the targets."
    else:
        summary = plan.summary
    if (
        repaired_shield_placement
        and not _requested_decontamination(instruction)
        and len(shield_fractions) <= 1
        and not _requests_multiple_shields(instruction)
    ):
        duration = 2.0 if requested_duration_s is None else requested_duration_s
        summary = (
            "Place the lead shield between the radiation source and protected area, "
            f"then measure for {duration:g} seconds."
        )
    return CommandPlan.model_validate(
        {
            **plan.model_dump(),
            "language": language,
            "summary": summary,
            "steps": tuple(steps),
        }
    )


class OpenAICompatibleCommandInterpreter:
    """Interpret commands through llama.cpp without adding an SDK dependency."""

    def __init__(
        self,
        endpoint: str,
        *,
        model: str = "radcounter-qwen3-4b",
        timeout_s: float = 90.0,
        allow_remote: bool = False,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("endpoint must be an HTTP(S) URL")
        if not allow_remote and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("the natural-language endpoint must be loopback-only")
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    @property
    def completion_url(self) -> str:
        if self.endpoint.endswith("/v1"):
            return f"{self.endpoint}/chat/completions"
        return f"{self.endpoint}/v1/chat/completions"

    async def interpret(self, instruction: str, context: CommandContext) -> CommandPlan:
        normalized = normalize_operator_instruction(instruction)
        if not normalized:
            raise CommandInterpretationError("instruction is empty")
        return await asyncio.to_thread(self._interpret_sync, normalized, context)

    def _interpret_sync(self, instruction: str, context: CommandContext) -> CommandPlan:
        context_json = context.model_dump_json(exclude_none=True)
        schema = _inlined_command_schema()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Current public application context:\n{context_json}\n\n"
                        f"Operator instruction:\n{instruction}"
                    ),
                },
            ],
            "temperature": 0.0,
            "max_tokens": 2048,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "radcounter_command_plan",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        request = Request(
            self.completion_url,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={
                "Authorization": "Bearer radcounter-local",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:  # noqa: S310
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")[-800:]
            raise CommandInterpretationError(
                f"local model server returned HTTP {error.code}: {details}"
            ) from error
        except (OSError, URLError, json.JSONDecodeError) as error:
            raise CommandInterpretationError(f"local model request failed: {error}") from error
        try:
            content = result["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "".join(
                    str(item.get("text", "")) if isinstance(item, dict) else str(item)
                    for item in content
                )
            plan = CommandPlan.model_validate_json(str(content))
            return _normalize_plan(plan, instruction, context)
        except (KeyError, IndexError, TypeError, ValidationError, json.JSONDecodeError) as error:
            raise CommandInterpretationError(
                "local model returned a response that does not match the command schema"
            ) from error
