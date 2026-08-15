from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from pydantic import ValidationError

from radcounter.core.natural_language import (
    AvailableAction,
    CommandContext,
    CommandName,
    CommandPlan,
    CommandStep,
    CompletionCriterion,
    LlamaCppRuntime,
    LlamaCppRuntimeConfig,
    OpenAICompatibleCommandInterpreter,
    PlanValidationError,
    RuntimeAssetMissingError,
    StepCompletionCondition,
    validate_command_plan,
)
from radcounter.core.natural_language.client import (
    _inlined_command_schema,
    _normalize_plan,
    normalize_operator_instruction,
)
from radcounter.core.natural_language.runtime import NvidiaGpu

ROOT = Path(__file__).resolve().parents[2]


def _controller_class():
    import radcounter

    namespace = str(ROOT / "source/extensions/radcounter.isaac/radcounter")
    if namespace not in radcounter.__path__:
        radcounter.__path__.append(namespace)
    from radcounter.isaac.natural_language import NaturalLanguageCommandController

    return NaturalLanguageCommandController


def _context() -> CommandContext:
    return CommandContext(
        session_state="ready",
        stage_name="vertical_slice",
        available_actions=(
            AvailableAction(
                action_id="measure-protected",
                action_type="measure",
                label="Measure at Protected station",
                target="/World/DetectorStations/Protected",
            ),
            AvailableAction(
                action_id="remove-drum",
                action_type="remove_object",
                label="Move contaminated drum to disposal",
                target="/World/HiddenContaminatedDrum",
                feasible=False,
            ),
        ),
    )


def _shield_context() -> CommandContext:
    return CommandContext(
        session_state="ready",
        stage_name="vertical_slice",
        available_actions=(
            AvailableAction(
                action_id="shield-world-leadshield-35",
                action_type="place_shield",
                label="Place lead shield",
                target="/World/LeadShield",
            ),
            AvailableAction(
                action_id="shield-world-leadshield-50",
                action_type="place_shield",
                label="Place lead shield",
                target="/World/LeadShield",
                feasible=False,
            ),
        ),
    )


def _shield_clearance_context() -> CommandContext:
    return CommandContext(
        session_state="ready",
        stage_name="vertical_slice",
        available_actions=(
            AvailableAction(
                action_id="shield-world-leadshield-35",
                action_type="place_shield",
                label="Place lead shield at 35 percent",
                feasible=False,
            ),
            AvailableAction(
                action_id="shield-world-leadshield-65",
                action_type="place_shield",
                label="Place lead shield at 65 percent",
                feasible=True,
            ),
        ),
    )


def test_command_plan_rejects_free_form_arguments() -> None:
    normalized = CommandStep.model_validate(
        {"command": "pause", "candidate_id": "measure-protected", "duration_s": 0.1}
    )
    assert normalized.candidate_id is None
    assert normalized.duration_s is None
    with pytest.raises(ValidationError):
        CommandPlan.model_validate(
            {
                "language": "en",
                "summary": "Run arbitrary code",
                "steps": [{"command": "run_python", "code": "print('unsafe')"}],
            }
        )


def test_host_validation_resolves_scene_candidates_and_confirmation() -> None:
    plan = CommandPlan(
        language="en",
        summary="Measure in the protected area",
        steps=(
            CommandStep(
                command=CommandName.EXECUTE_CANDIDATE,
                candidate_id="measure-protected",
            ),
            CommandStep(command=CommandName.MEASURE, duration_s=2.0),
        ),
    )
    validated = validate_command_plan(plan, _context())
    assert validated.requires_confirmation is True
    assert validated.warnings == ("Physical action: Measure at Protected station",)


def test_host_validation_rejects_invented_or_infeasible_candidate() -> None:
    for candidate_id in ("invented-action", "remove-drum"):
        plan = CommandPlan(
            language="en",
            summary="Do it",
            steps=(
                CommandStep(
                    command=CommandName.EXECUTE_CANDIDATE,
                    candidate_id=candidate_id,
                ),
            ),
        )
        with pytest.raises(PlanValidationError):
            validate_command_plan(plan, _context())


def test_host_validation_allows_confirmed_sequential_physical_actions() -> None:
    plan = CommandPlan(
        language="en",
        summary="Move, measure, and return",
        steps=(
            CommandStep(
                command=CommandName.EXECUTE_CANDIDATE,
                candidate_id="measure-protected",
            ),
            CommandStep(command=CommandName.MEASURE, duration_s=5.0),
            CommandStep(command=CommandName.RETURN_MEASUREMENT_ROBOT),
        ),
    )
    validated = validate_command_plan(plan, _context())
    assert validated.requires_confirmation is True
    assert validated.warnings == (
        "Physical action: Measure at Protected station",
        "Physical action: return measurement robot to its starting position",
    )


def test_command_plan_supports_twenty_four_step_workflows_but_remains_bounded() -> None:
    steps = tuple(CommandStep(command=CommandName.SHOW_STATUS) for _ in range(24))
    assert len(CommandPlan(language="en", summary="Inspect repeatedly", steps=steps).steps) == 24
    with pytest.raises(ValidationError):
        CommandPlan(language="en", summary="Too long", steps=(*steps, steps[0]))


def test_bounded_workflow_rejects_more_than_forty_eight_executions() -> None:
    context = CommandContext(
        session_state="ready",
        available_actions=(
            AvailableAction(
                action_id="decon-surface",
                action_type="decontaminate",
                label="Full serpentine surface decontamination",
            ),
        ),
    )
    repeated = CommandStep(
        command=CommandName.EXECUTE_CANDIDATE,
        candidate_id="decon-surface",
        max_attempts=5,
    )
    plan = CommandPlan(language="en", summary="Too many passes", steps=(repeated,) * 10)
    with pytest.raises(PlanValidationError, match="48"):
        validate_command_plan(plan, context)


def test_llama_schema_is_inlined_and_does_not_expose_truth() -> None:
    encoded = json.dumps(_inlined_command_schema(), sort_keys=True).lower()
    assert "$ref" not in encoded
    assert "$defs" not in encoded
    assert "truth" not in encoded
    assert "execute_candidate" in encoded


def test_explicit_english_measurement_is_not_lost_after_navigation() -> None:
    incomplete = CommandPlan(
        language="en",
        summary="Move to the station",
        steps=(
            CommandStep(
                command=CommandName.EXECUTE_CANDIDATE,
                candidate_id="measure-protected",
            ),
        ),
    )
    concise = _normalize_plan(
        incomplete,
        "Move the measurement robot to the protected area and measure for 2 seconds",
        _context(),
    )
    assert concise.language == "en"
    assert concise.steps[-1] == CommandStep(command=CommandName.MEASURE, duration_s=2.0)

    english = _normalize_plan(
        incomplete,
        "Move to the protected area and measure for 3 seconds",
        _context(),
    )
    assert english.language == "en"
    assert english.steps[-1] == CommandStep(command=CommandName.MEASURE, duration_s=3.0)

    model_kept_measure_but_lost_duration = CommandPlan(
        language="en",
        summary="Move and measure",
        steps=(
            CommandStep(
                command=CommandName.EXECUTE_CANDIDATE,
                candidate_id="measure-protected",
            ),
            CommandStep(command=CommandName.MEASURE),
        ),
    )
    repaired_duration = _normalize_plan(
        model_kept_measure_but_lost_duration,
        "Move to the protected area and measure for 2 seconds",
        _context(),
    )
    assert repaired_duration.steps[-1].duration_s == 2.0


def test_explicit_move_measure_return_is_repaired_to_three_ordered_steps() -> None:
    small_model_plan = CommandPlan(
        language="en",
        summary="Move, measure, and return",
        steps=(
            CommandStep(
                command=CommandName.EXECUTE_CANDIDATE,
                candidate_id="measure-protected",
            ),
            CommandStep(command=CommandName.MEASURE),
            # A small model can incorrectly encode "return" as another station move.
            CommandStep(
                command=CommandName.EXECUTE_CANDIDATE,
                candidate_id="measure-protected",
            ),
        ),
    )
    normalized = _normalize_plan(
        small_model_plan,
        (
            "Move the measurement robot to the protected area, measure the radiation "
            "level for 5 seconds, then return it to its starting position"
        ),
        _context(),
    )
    assert normalized.steps == (
        CommandStep(
            command=CommandName.EXECUTE_CANDIDATE,
            candidate_id="measure-protected",
        ),
        CommandStep(command=CommandName.MEASURE, duration_s=5.0),
        CommandStep(command=CommandName.RETURN_MEASUREMENT_ROBOT),
    )


def test_operator_instruction_collapses_only_triple_or_more_exact_repetitions() -> None:
    instruction = "Move the measurement robot to the protected area and measure for 2 seconds."
    assert normalize_operator_instruction(instruction * 3) == instruction
    assert normalize_operator_instruction(instruction * 5) == instruction
    assert normalize_operator_instruction(instruction * 2) == instruction * 2
    assert normalize_operator_instruction("Pause the simulation") == "Pause the simulation"


def test_operator_instruction_repairs_fuzzy_repetition_and_quote_marker() -> None:
    instruction = (
        "Use the countermeasure robot's manipulator to pick up the lead shield, "
        "place it between the source and protected area, then measure for 5 seconds "
        "to  > verify the effec"
    )
    corrupted = instruction * 3 + instruction.replace("  > ", " ")
    assert normalize_operator_instruction(corrupted) == instruction.replace("  > ", " ")


def test_explicit_shield_manipulation_repairs_small_model_refusal() -> None:
    refusal = CommandPlan(
        language="en",
        summary="The requested placement lacks public coordinates.",
        steps=(CommandStep(command=CommandName.SHOW_STATUS),),
    )
    normalized = _normalize_plan(
        refusal,
        (
            "Use the countermeasure robot's manipulator to pick up the lead shield, "
            "place it between the radiation source and the protected area, then measure "
            "the radiation level for 5 seconds to verify the effect."
        ),
        _shield_context(),
    )
    assert normalized.steps == (
        CommandStep(
            command=CommandName.EXECUTE_CANDIDATE,
            candidate_id="shield-world-leadshield-35",
        ),
        CommandStep(command=CommandName.MEASURE, duration_s=5.0),
    )
    assert "Place the lead shield" in normalized.summary
    assert validate_command_plan(normalized, _shield_context()).requires_confirmation is True


def test_infeasible_shield_choice_is_replaced_by_host_verified_safe_candidate() -> None:
    plan = CommandPlan(
        language="en",
        summary="Place the shield",
        steps=(
            CommandStep(
                command=CommandName.EXECUTE_CANDIDATE,
                candidate_id="shield-world-leadshield-35",
            ),
        ),
    )
    normalized = _normalize_plan(
        plan,
        "Use the manipulator to place the lead shield between the source and protected area",
        _shield_clearance_context(),
    )
    assert normalized.steps[0].candidate_id == "shield-world-leadshield-65"
    assert validate_command_plan(normalized, _shield_clearance_context()).requires_confirmation


def test_explicit_decontamination_repairs_small_model_refusal() -> None:
    context = CommandContext(
        session_state="ready",
        stage_name="complex_work_cell",
        available_actions=(
            AvailableAction(
                action_id="decon-world-deconworksurface",
                action_type="decontaminate",
                label="Decontaminate planar Cs-137 source",
                target="/World/DeconWorkSurface",
            ),
        ),
    )
    refusal = CommandPlan(
        language="en",
        summary="The target could not be identified",
        steps=(CommandStep(command=CommandName.SHOW_STATUS),),
    )
    normalized = _normalize_plan(
        refusal,
        "Avoid the obstacle, move to the contaminated work surface, and decontaminate "
        "the entire surface source",
        context,
    )
    assert normalized.steps == (
        CommandStep(
            command=CommandName.EXECUTE_CANDIDATE,
            candidate_id="decon-world-deconworksurface",
        ),
    )
    assert validate_command_plan(normalized, context).requires_confirmation


def test_decontamination_verification_collapses_redundant_small_model_steps() -> None:
    context = CommandContext(
        session_state="ready",
        stage_name="complex_work_cell",
        available_actions=(
            AvailableAction(
                action_id="decon-world-deconworksurface",
                action_type="decontaminate",
                label="Decontaminate planar source",
            ),
            AvailableAction(
                action_id="measure-world-detectorstations-protected",
                action_type="measure",
                label="Move to protected detector",
            ),
        ),
    )
    redundant = CommandPlan(
        language="en",
        summary="Decontaminate and measure",
        steps=(
            CommandStep(command=CommandName.SHOW_STATUS),
            CommandStep(
                command=CommandName.EXECUTE_CANDIDATE,
                candidate_id="measure-world-detectorstations-protected",
            ),
            CommandStep(command=CommandName.MEASURE),
            CommandStep(
                command=CommandName.EXECUTE_CANDIDATE,
                candidate_id="measure-world-detectorstations-protected",
            ),
            CommandStep(command=CommandName.SHOW_STATUS),
        ),
    )
    normalized = _normalize_plan(
        redundant,
        "Decontaminate the surface source, move to the protected area, and measure for 5 seconds",
        context,
    )
    assert normalized.steps == (
        CommandStep(
            command=CommandName.EXECUTE_CANDIDATE,
            candidate_id="decon-world-deconworksurface",
        ),
        CommandStep(
            command=CommandName.EXECUTE_CANDIDATE,
            candidate_id="measure-world-detectorstations-protected",
        ),
        CommandStep(command=CommandName.MEASURE, duration_s=5.0),
    )


def test_complex_decon_measure_return_status_is_ordered_by_operator_semantics() -> None:
    context = CommandContext(
        session_state="ready",
        stage_name="maze",
        available_actions=(
            AvailableAction(
                action_id="decon-surface",
                action_type="decontaminate",
                label="Decontaminate surface",
            ),
            AvailableAction(
                action_id="measure-protected",
                action_type="measure",
                label="Move to protected detector",
            ),
            AvailableAction(
                action_id="measure-remote-decon-room",
                action_type="measure",
                label="Move to remote decontamination room",
            ),
            AvailableAction(
                action_id="move-obstacle",
                action_type="move_object",
                label="Move unrelated obstacle",
            ),
        ),
    )
    out_of_order = CommandPlan(
        language="en",
        summary="Combined workflow",
        steps=(
            CommandStep(
                command=CommandName.EXECUTE_CANDIDATE,
                candidate_id="move-obstacle",
            ),
            CommandStep(
                command=CommandName.EXECUTE_CANDIDATE,
                candidate_id="measure-protected",
            ),
            CommandStep(
                command=CommandName.EXECUTE_CANDIDATE,
                candidate_id="move-obstacle",
            ),
            CommandStep(
                command=CommandName.EXECUTE_CANDIDATE,
                candidate_id="decon-surface",
            ),
            CommandStep(command=CommandName.SHOW_STATUS),
            CommandStep(command=CommandName.MEASURE),
            CommandStep(command=CommandName.RETURN_MEASUREMENT_ROBOT),
        ),
    )
    normalized = _normalize_plan(
        out_of_order,
        "Decontaminate the surface source, then measure for 5 seconds in the remote "
        "decontamination room, return to the starting position, and show status",
        context,
    )
    assert normalized.steps == (
        CommandStep(command=CommandName.EXECUTE_CANDIDATE, candidate_id="decon-surface"),
        CommandStep(
            command=CommandName.EXECUTE_CANDIDATE,
            candidate_id="measure-remote-decon-room",
        ),
        CommandStep(command=CommandName.MEASURE, duration_s=5.0),
        CommandStep(command=CommandName.RETURN_MEASUREMENT_ROBOT),
        CommandStep(command=CommandName.SHOW_STATUS),
    )


def test_complex_bounded_decon_and_ordered_shield_reposition_are_repaired() -> None:
    context = CommandContext(
        session_state="ready",
        stage_name="multi_room_decommissioning_facility",
        available_actions=(
            AvailableAction(
                action_id="decon-wall-source",
                action_type="decontaminate",
                label="Full six-lane serpentine raster · irregular wall source",
                target="/World/DeconWorkSurface",
                execution_mode="full_coverage_serpentine_raster",
            ),
            AvailableAction(
                action_id="shield-primary-35",
                action_type="place_shield",
                label="Place primary shield at 35 percent",
                target="/World/LeadShield",
                placement_fraction=0.35,
            ),
            AvailableAction(
                action_id="shield-primary-65",
                action_type="place_shield",
                label="Place primary shield at 65 percent",
                target="/World/LeadShield",
                placement_fraction=0.65,
            ),
            AvailableAction(
                action_id="measure-protected",
                action_type="measure",
                label="Move to protected detector",
                target="/World/DetectorStations/Protected",
                target_label="Protected area",
            ),
        ),
    )
    refusal = CommandPlan(
        language="en",
        summary="The workflow is too complex to execute",
        steps=(CommandStep(command=CommandName.SHOW_STATUS),),
    )
    normalized = _normalize_plan(
        refusal,
        (
            "Decontaminate the irregular wall source up to three times until the removal "
            "fraction reaches at least 70%, place the lead shield at 35% of the line from "
            "the source to the protected area and then move it to 65%, move to the protected "
            "area and measure for 5 seconds, return to the start, and show status"
        ),
        context,
    )
    assert normalized.steps == (
        CommandStep(
            command=CommandName.EXECUTE_CANDIDATE,
            candidate_id="decon-wall-source",
            max_attempts=3,
            until=StepCompletionCondition(
                criterion=CompletionCriterion.DECONTAMINATION_REMOVED_FRACTION_AT_LEAST,
                threshold=0.70,
            ),
        ),
        CommandStep(
            command=CommandName.EXECUTE_CANDIDATE,
            candidate_id="shield-primary-35",
        ),
        CommandStep(
            command=CommandName.EXECUTE_CANDIDATE,
            candidate_id="shield-primary-65",
        ),
        CommandStep(
            command=CommandName.EXECUTE_CANDIDATE,
            candidate_id="measure-protected",
        ),
        CommandStep(command=CommandName.MEASURE, duration_s=5.0),
        CommandStep(command=CommandName.RETURN_MEASUREMENT_ROBOT),
        CommandStep(command=CommandName.SHOW_STATUS),
    )
    validated = validate_command_plan(normalized, context)
    assert validated.requires_confirmation
    assert any("up to 3 attempts" in warning for warning in validated.warnings)


def test_all_station_instruction_expands_to_navigation_and_measurement_pairs() -> None:
    context = CommandContext(
        session_state="ready",
        available_actions=tuple(
            AvailableAction(
                action_id=f"measure-{name}",
                action_type="measure",
                label=f"Move to {name}",
                target=f"/World/DetectorStations/{name}",
            )
            for name in ("north", "protected", "remote")
        ),
    )
    small_model = CommandPlan(
        language="en",
        summary="Measure",
        steps=(CommandStep(command=CommandName.MEASURE),),
    )
    normalized = _normalize_plan(
        small_model,
        "Visit every measurement station in order and measure for 2 seconds at each station",
        context,
    )
    assert len(normalized.steps) == 6
    assert [step.command for step in normalized.steps] == [
        CommandName.EXECUTE_CANDIDATE,
        CommandName.MEASURE,
    ] * 3
    assert all(
        step.duration_s == 2.0 for step in normalized.steps if step.command == CommandName.MEASURE
    )


def test_builtin_command_misreported_as_candidate_is_normalized() -> None:
    context = CommandContext(
        session_state="playing",
        capabilities=("pause", "show_status"),
    )
    small_model_plan = CommandPlan(
        language="en",
        summary="Pause the simulation",
        steps=(
            CommandStep(
                command=CommandName.EXECUTE_CANDIDATE,
                candidate_id="pause",
            ),
        ),
    )
    normalized = _normalize_plan(small_model_plan, "Pause the simulation", context)
    assert normalized.steps == (CommandStep(command=CommandName.PAUSE),)


def test_openai_compatible_client_accepts_english_structured_plan() -> None:
    requests: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers["Content-Length"])
            requests.append(json.loads(self.rfile.read(length)))
            content = json.dumps(
                {
                    "language": "en",
                    "summary": "Measure in the protected area",
                    "steps": [
                        {
                            "command": "execute_candidate",
                            "candidate_id": "measure-protected",
                        },
                        {"command": "measure", "duration_s": 2.0},
                    ],
                },
                ensure_ascii=False,
            )
            body = json.dumps(
                {"choices": [{"message": {"role": "assistant", "content": content}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        interpreter = OpenAICompatibleCommandInterpreter(f"http://127.0.0.1:{server.server_port}")
        import asyncio

        plan = asyncio.run(
            interpreter.interpret(
                "Move the measurement robot to the protected area and measure for 2 seconds",
                _context(),
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert plan.language == "en"
    assert [step.command for step in plan.steps] == [
        CommandName.EXECUTE_CANDIDATE,
        CommandName.MEASURE,
    ]
    response_format = requests[0]["response_format"]
    assert isinstance(response_format, dict)
    assert response_format["type"] == "json_schema"
    assert "measurement robot" in str(requests[0]["messages"])


def test_client_refuses_non_loopback_endpoint() -> None:
    with pytest.raises(ValueError, match="loopback"):
        OpenAICompatibleCommandInterpreter("https://example.com/v1")

    runtime = LlamaCppRuntime(
        LlamaCppRuntimeConfig(
            runtime_directory=ROOT / "runtime/llm",
            external_endpoint="https://example.com",
        )
    )
    with pytest.raises(ValueError, match="loopback"):
        runtime.start()


def test_runtime_selects_packaged_cuda_and_cpu_binaries(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    cpu = runtime_root / "bin/linux-x86_64-cpu/llama-server"
    cuda = runtime_root / "bin/linux-x86_64-cuda/llama-server"
    vulkan = runtime_root / "bin/linux-x86_64-vulkan/llama-server"
    model = runtime_root / "models/Qwen3-4B-Q4_K_M.gguf"
    for path in (cpu, cuda, vulkan, model):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"runtime-test")
    cpu.chmod(0o755)
    cuda.chmod(0o755)
    vulkan.chmod(0o755)

    runtime = LlamaCppRuntime(LlamaCppRuntimeConfig(runtime_directory=runtime_root))
    runtime.gpu = NvidiaGpu("Test GPU", 10_240)
    assert runtime.resolve_binary() == cuda
    command = runtime._command(cuda, model, 12345)
    assert command[-1] == "auto"

    cuda.unlink()
    assert runtime.resolve_binary() == vulkan

    cpu_runtime = LlamaCppRuntime(
        LlamaCppRuntimeConfig(runtime_directory=runtime_root, gpu_mode="cpu")
    )
    cpu_runtime.gpu = NvidiaGpu("Test GPU", 10_240)
    assert cpu_runtime.resolve_binary() == cpu

    runtime.gpu = None
    assert runtime.resolve_binary() == cpu
    assert runtime._command(cpu, model, 12345)[-1] == "0"


def test_runtime_reports_missing_product_assets(tmp_path: Path) -> None:
    runtime = LlamaCppRuntime(
        LlamaCppRuntimeConfig(
            runtime_directory=tmp_path,
            server_binary=tmp_path / "missing-llama-server",
        )
    )
    with pytest.raises(RuntimeAssetMissingError, match="build_llama_runtime"):
        runtime.resolve_binary()


def test_controller_auto_runs_safe_commands_and_confirms_physical_actions(
    tmp_path: Path,
) -> None:
    import asyncio

    class Interpreter:
        def __init__(self) -> None:
            self.plan = CommandPlan(
                language="en",
                summary="Pause",
                steps=(CommandStep(command=CommandName.PAUSE),),
            )

        async def interpret(self, instruction: str, context: CommandContext) -> CommandPlan:
            del instruction, context
            return self.plan

    class Host:
        def __init__(self) -> None:
            self.steps: list[CommandStep] = []

        def natural_language_context(self) -> CommandContext:
            return _context()

        async def execute_natural_language_step(self, step: CommandStep) -> dict[str, object]:
            self.steps.append(step)
            return {"command": step.command.value, "ok": True}

    host = Host()
    interpreter = Interpreter()
    audit = tmp_path / "commands.jsonl"
    NaturalLanguageCommandController = _controller_class()
    controller = NaturalLanguageCommandController(
        host,
        interpreter=interpreter,
        audit_path=audit,
    )
    safe = asyncio.run(controller.submit("Pause the simulation"))
    assert safe.executed is True
    assert host.steps[-1].command == CommandName.PAUSE

    interpreter.plan = CommandPlan(
        language="en",
        summary="Move to the protected area",
        steps=(
            CommandStep(
                command=CommandName.EXECUTE_CANDIDATE,
                candidate_id="measure-protected",
            ),
        ),
    )
    physical = asyncio.run(controller.submit("Move to the protected area"))
    assert physical.executed is False
    assert controller.pending is not None
    assert host.steps[-1].command == CommandName.PAUSE
    confirmed = asyncio.run(controller.confirm())
    assert confirmed.executed is True
    assert host.steps[-1].candidate_id == "measure-protected"

    rows = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
    assert [row["status"] for row in rows] == [
        "completed",
        "awaiting_confirmation",
        "completed",
    ]
    controller.shutdown()


def test_controller_stops_bounded_decontamination_when_public_condition_is_met(
    tmp_path: Path,
) -> None:
    import asyncio

    context = CommandContext(
        session_state="ready",
        available_actions=(
            AvailableAction(
                action_id="decon-irregular-wall",
                action_type="decontaminate",
                label="Full irregular-wall serpentine raster",
            ),
        ),
    )
    condition = StepCompletionCondition(
        criterion=CompletionCriterion.DECONTAMINATION_REMAINING_FRACTION_AT_MOST,
        threshold=0.30,
    )

    class Interpreter:
        async def interpret(self, instruction: str, supplied: CommandContext) -> CommandPlan:
            del instruction, supplied
            return CommandPlan(
                language="en",
                summary="Decontaminate up to five times until no more than 30% remains",
                steps=(
                    CommandStep(
                        command=CommandName.EXECUTE_CANDIDATE,
                        candidate_id="decon-irregular-wall",
                        max_attempts=5,
                        until=condition,
                    ),
                ),
            )

    class Host:
        def __init__(self) -> None:
            self.calls = 0

        def natural_language_context(self) -> CommandContext:
            return context

        async def execute_natural_language_step(self, step: CommandStep) -> dict[str, object]:
            del step
            before_after = ((100.0, 80.0), (80.0, 55.0), (55.0, 25.0))
            before, after = before_after[self.calls]
            self.calls += 1
            return {
                "command": "execute_candidate",
                "public_details": {
                    "motion_audit": {
                        "activity_before_bq": before,
                        "activity_after_bq": after,
                        "removed_fraction": (before - after) / before,
                        "coverage_fraction": 0.94,
                    }
                },
            }

    host = Host()
    NaturalLanguageCommandController = _controller_class()
    controller = NaturalLanguageCommandController(
        host,
        interpreter=Interpreter(),
        audit_path=tmp_path / "bounded.jsonl",
    )
    pending = asyncio.run(
        controller.submit("Decontaminate up to five times until no more than 30% remains")
    )
    assert pending.executed is False
    completed = asyncio.run(controller.confirm())
    assert host.calls == 3
    assert len(completed.results) == 3
    assert completed.results[-1]["completion_condition"] == {
        "criterion": "decontamination_remaining_fraction_at_most",
        "threshold": 0.30,
        "observed": 0.25,
        "met": True,
    }
    controller.shutdown()


def test_product_runtime_and_command_surface_artifacts_exist() -> None:
    dashboard = (
        ROOT / "source/extensions/radcounter.isaac/radcounter/isaac/ui/dashboard.py"
    ).read_text(encoding="utf-8")
    run_gui = (ROOT / "scripts/run_gui.py").read_text(encoding="utf-8")
    run_gui_validation = (ROOT / "scripts/run_gui_validation.py").read_text(encoding="utf-8")
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "NATURAL LANGUAGE COMMAND" in dashboard
    assert "def bind_workflow" in dashboard
    assert "execute_natural_language_step" in dashboard
    assert '"--interactive"' in run_gui
    assert '"extra_args"' not in run_gui_validation
    assert "subscribe_value_changed_fn" in dashboard
    assert "model=self._command_status" not in dashboard
    assert "update_navigation_progress" in dashboard
    assert "motion_audit" in dashboard
    assert "llama.cpp" in notices
    assert "Qwen3-4B" in notices
    assert (ROOT / "scripts/build_llama_runtime.sh").is_file()
    assert (ROOT / "scripts/fetch_llm_model.py").is_file()
