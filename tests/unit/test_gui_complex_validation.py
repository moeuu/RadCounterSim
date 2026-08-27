from __future__ import annotations

from copy import deepcopy

import pytest

from radcounter.core.natural_language import (
    AvailableAction,
    CommandContext,
    CommandName,
    CommandPlan,
    CommandStep,
)
from radcounter.core.natural_language.client import _normalize_plan
from scripts.run_gui_validation import (
    DEFAULT_COMPLEX_NATURAL_LANGUAGE_INSTRUCTION,
    DEFAULT_GUI_MAX_FPS,
    _arguments,
    _complex_process_audit,
    _GuiFrameRateLimiter,
    _type_visible_instruction,
)

SURFACE = "/World/RemoteDeconFacility/DeconWorkSurface"
SHIELD = "/World/LeadShield"
PROTECTED = "/World/DetectorStations/Protected"


def _environment(activity_bq: float, *, deployed: bool) -> dict[str, object]:
    return {
        "failed_invariants": [],
        "surface_source": {"map_activity_bq": activity_bq},
        "shields": [
            {
                "path": SHIELD,
                "deployed": deployed,
                "placement_fraction": 0.65 if deployed else None,
            },
            {
                "path": "/World/StagingLeadShield",
                "deployed": False,
                "placement_fraction": None,
            },
        ],
    }


def _workflow_row(step: int, **values: object) -> dict[str, object]:
    return {
        "workflow_step": step,
        "attempt": 1,
        "max_attempts": 1,
        **values,
    }


def _decon_row(
    attempt: int,
    before_bq: float,
    after_bq: float,
    observed: float,
    *,
    condition_met: bool,
) -> dict[str, object]:
    return {
        "command": "execute_candidate",
        "action_id": "decon-irregular-wall",
        "action_type": "decontaminate",
        "status": "completed",
        "public_details": {
            "surface_path": SURFACE,
            "accepted_contacts": 120,
            "motion_audit": {
                "success": True,
                "activity_before_bq": before_bq,
                "activity_after_bq": after_bq,
                "coverage_fraction": 0.94,
            },
            "collateral_motion_audit": [
                {"object_path": "/World/MovableObstacle", "displacement_m": 0.002}
            ],
        },
        "workflow_step": 1,
        "attempt": attempt,
        "max_attempts": 3,
        "completion_condition": {
            "criterion": "decontamination_remaining_fraction_at_most",
            "threshold": 0.60,
            "observed": observed,
            "met": condition_met,
        },
    }


def _passing_payload() -> tuple[dict[str, object], list[dict[str, object]]]:
    plan: dict[str, object] = {
        "language": "en",
        "summary": "Combined decontamination, shielding, and measurement process",
        "steps": [
            {
                "command": "execute_candidate",
                "candidate_id": "decon-irregular-wall",
                "max_attempts": 3,
                "until": {
                    "criterion": "decontamination_remaining_fraction_at_most",
                    "threshold": 0.60,
                },
            },
            {
                "command": "execute_candidate",
                "candidate_id": "shield-world-leadshield-25",
                "max_attempts": 1,
            },
            {
                "command": "execute_candidate",
                "candidate_id": "shield-world-leadshield-65",
                "max_attempts": 1,
            },
            {
                "command": "execute_candidate",
                "candidate_id": "measure-world-detectorstations-protected",
                "max_attempts": 1,
            },
            {"command": "measure", "duration_s": 2.0, "max_attempts": 1},
            {"command": "return_measurement_robot", "max_attempts": 1},
            {"command": "show_status", "max_attempts": 1},
        ],
    }
    collateral = [{"object_path": "/World/MovableObstacle", "displacement_m": 0.003}]
    results = [
        _decon_row(1, 100.0, 79.0, 0.79, condition_met=False),
        _decon_row(2, 79.0, 63.0, 0.63, condition_met=False),
        _decon_row(3, 63.0, 50.0, 0.50, condition_met=True),
        _workflow_row(
            2,
            command="execute_candidate",
            action_id="shield-world-leadshield-25",
            action_type="place_shield",
            status="completed",
            public_details={
                "object_path": SHIELD,
                "deployment_state": "deployed",
                "placement_fraction": 0.25,
                "motion_audit": {"success": True, "placement_error_m": 0.01},
                "collateral_motion_audit": collateral,
            },
        ),
        _workflow_row(
            3,
            command="execute_candidate",
            action_id="shield-world-leadshield-65",
            action_type="move_shield",
            status="completed",
            public_details={
                "object_path": SHIELD,
                "deployment_state": "deployed",
                "placement_fraction": 0.65,
                "motion_audit": {"success": True, "placement_error_m": 0.01},
                "collateral_motion_audit": collateral,
            },
        ),
        _workflow_row(
            4,
            command="execute_candidate",
            action_id="measure-world-detectorstations-protected",
            action_type="measure",
            status="completed",
            public_details={
                "detector_path": PROTECTED,
                "motion_audit": {"success": True, "displacement_m": 4.0},
            },
        ),
        _workflow_row(
            5,
            command="measure",
            duration_s=2.0,
            detector_count=1,
            measurements=[
                {
                    "detector_path": PROTECTED,
                    "duration_s": 2.0,
                    "counts": 42,
                    "measured_rate_cps": 21.0,
                }
            ],
            status="Integrated 1 detector",
        ),
        _workflow_row(
            6,
            command="return_measurement_robot",
            status="completed",
            public_details={"motion_audit": {"success": True, "displacement_m": 4.0}},
        ),
        _workflow_row(7, command="show_status", status="ready"),
    ]
    return plan, results


def _audit(plan: dict[str, object], results: list[dict[str, object]]) -> dict[str, object]:
    return _complex_process_audit(
        plan,
        results,
        _environment(100.0, deployed=False),
        _environment(50.0, deployed=True),
        {
            "measurement_robot_initial_position_m": [0.0, 0.0, 0.0],
            "measurement_robot_final_position_m": [0.02, 0.01, 0.0],
            "primary_shield_initial_position_m": [4.8, 2.8, 0.0],
            "primary_shield_final_position_m": [1.8, 0.8, 0.0],
        },
        expected_surface_path=SURFACE,
    )


def test_complex_validation_cli_requires_visible_separate_mode() -> None:
    args = _arguments(["--complex-natural-language-validation", "--no-keep-open"])
    assert args.complex_natural_language_validation is True
    assert args.keep_open is False
    assert args.headless is False
    assert args.natural_language_timeout_s == 1800.0
    assert args.max_fps == DEFAULT_GUI_MAX_FPS

    with pytest.raises(SystemExit):
        _arguments(["--complex-natural-language-validation", "--headless"])
    with pytest.raises(SystemExit):
        _arguments(["--complex-natural-language-validation", "--interactive"])


def test_gui_fps_limit_is_configurable_and_rejects_negative_values() -> None:
    assert _arguments(["--max-fps", "30"]).max_fps == 30.0
    assert _arguments(["--max-fps", "0"]).max_fps == 0.0
    with pytest.raises(SystemExit):
        _arguments(["--max-fps", "-1"])
    with pytest.raises(SystemExit):
        _arguments(["--max-fps", "nan"])


def test_prompt_video_requires_visible_confirmed_interactive_command() -> None:
    output = "artifacts/video/prompt.mp4"
    args = _arguments(
        [
            "--interactive",
            "--initial-command",
            "Decontaminate the wall.",
            "--confirm-initial-command",
            "--record-prompt-video",
            output,
            "--record-monitor-x",
            "3840",
        ]
    )
    assert str(args.record_prompt_video) == output
    assert args.record_monitor_x == 3840
    assert args.prompt_typing_delay_s == pytest.approx(0.04)

    for invalid in (
        ["--record-prompt-video", output],
        ["--interactive", "--record-prompt-video", output],
        [
            "--interactive",
            "--initial-command",
            "Decontaminate the wall.",
            "--record-prompt-video",
            output,
        ],
    ):
        with pytest.raises(SystemExit):
            _arguments(invalid)


def test_visible_prompt_typing_updates_only_the_dashboard_model() -> None:
    class FakeApp:
        def __init__(self) -> None:
            self.updates = 0

        def update(self) -> None:
            self.updates += 1

    class FakeDashboard:
        def __init__(self) -> None:
            self.values: list[str] = []

        def set_natural_language_input(self, value: str) -> None:
            self.values.append(value)

    app = FakeApp()
    dashboard = FakeDashboard()
    sleeps: list[float] = []
    _type_visible_instruction(
        app,
        dashboard,
        "robot",
        character_delay_s=0.03,
        sleeper=sleeps.append,
    )
    assert dashboard.values == ["", "r", "ro", "rob", "robo", "robot"]
    assert app.updates == len(dashboard.values)
    assert sleeps == [0.03] * 5


def test_gui_frame_limiter_accounts_for_update_time() -> None:
    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0
            self.sleeps: list[float] = []

        def __call__(self) -> float:
            return self.now

        def sleep(self, duration_s: float) -> None:
            self.sleeps.append(duration_s)
            self.now += duration_s

    clock = FakeClock()
    limiter = _GuiFrameRateLimiter(50.0, clock=clock, sleeper=clock.sleep)
    clock.now += 0.006
    limiter.wait()
    assert clock.sleeps == pytest.approx([0.014])

    clock.now += 0.025
    limiter.wait()
    assert clock.sleeps == pytest.approx([0.014])

    unlimited = _GuiFrameRateLimiter(0.0, clock=clock, sleeper=clock.sleep)
    unlimited.wait()
    assert clock.sleeps == pytest.approx([0.014])


def test_default_instruction_requests_achievable_bounded_complex_process() -> None:
    assert "irregular wall-mounted Cs-137 surface source" in (
        DEFAULT_COMPLEX_NATURAL_LANGUAGE_INSTRUCTION
    )
    assert "remaining fraction is no more than 60%" in (
        DEFAULT_COMPLEX_NATURAL_LANGUAGE_INSTRUCTION
    )
    assert "at most 3 passes" in DEFAULT_COMPLEX_NATURAL_LANGUAGE_INSTRUCTION
    assert "LeadShield" in DEFAULT_COMPLEX_NATURAL_LANGUAGE_INSTRUCTION
    assert "25%" in DEFAULT_COMPLEX_NATURAL_LANGUAGE_INSTRUCTION
    assert "65%" in DEFAULT_COMPLEX_NATURAL_LANGUAGE_INSTRUCTION
    assert "Protected" in DEFAULT_COMPLEX_NATURAL_LANGUAGE_INSTRUCTION
    assert "measure for 2 seconds" in DEFAULT_COMPLEX_NATURAL_LANGUAGE_INSTRUCTION


def test_default_instruction_repairs_small_model_refusal_to_primary_shield_process() -> None:
    context = CommandContext(
        session_state="ready",
        available_actions=(
            AvailableAction(
                action_id="decon-irregular-wall",
                action_type="decontaminate",
                label="Irregular wall full raster",
                target=SURFACE,
            ),
            *tuple(
                AvailableAction(
                    action_id=f"shield-world-leadshield-{percentage}",
                    action_type="place_shield",
                    label=f"LeadShield {percentage}%",
                    target=SHIELD,
                    target_label="LeadShield",
                    placement_fraction=fraction,
                )
                for percentage, fraction in ((25, 0.25), (65, 0.65))
            ),
            *tuple(
                AvailableAction(
                    action_id=f"shield-world-stagingleadshield-{percentage}",
                    action_type="place_shield",
                    label=f"Staged shield {percentage}%",
                    target="/World/StagingLeadShield",
                    target_label="Staged lead service shield 02",
                    placement_fraction=fraction,
                )
                for percentage, fraction in ((25, 0.25), (65, 0.65))
            ),
            AvailableAction(
                action_id="measure-world-detectorstations-protected",
                action_type="measure",
                label="Move to Protected",
                target=PROTECTED,
                target_label="Protected",
            ),
        ),
        capabilities=(
            CommandName.EXECUTE_CANDIDATE,
            CommandName.MEASURE,
            CommandName.RETURN_MEASUREMENT_ROBOT,
            CommandName.SHOW_STATUS,
        ),
    )
    refusal = CommandPlan(
        language="en",
        summary="Show status only",
        steps=(CommandStep(command=CommandName.SHOW_STATUS),),
    )
    repaired = _normalize_plan(
        refusal,
        DEFAULT_COMPLEX_NATURAL_LANGUAGE_INSTRUCTION,
        context,
    )
    assert [step.command for step in repaired.steps] == [
        CommandName.EXECUTE_CANDIDATE,
        CommandName.EXECUTE_CANDIDATE,
        CommandName.EXECUTE_CANDIDATE,
        CommandName.EXECUTE_CANDIDATE,
        CommandName.MEASURE,
        CommandName.RETURN_MEASUREMENT_ROBOT,
        CommandName.SHOW_STATUS,
    ]
    assert repaired.steps[0].max_attempts == 3
    assert repaired.steps[0].until is not None
    assert repaired.steps[0].until.threshold == pytest.approx(0.60)
    assert [step.candidate_id for step in repaired.steps[1:3]] == [
        "shield-world-leadshield-25",
        "shield-world-leadshield-65",
    ]
    assert repaired.steps[4].duration_s == pytest.approx(2.0)


def test_complex_process_audit_accepts_ordered_bounded_public_results() -> None:
    plan, results = _passing_payload()
    audit = _audit(plan, results)
    assert audit["failed_invariants"] == []
    assert audit["decontamination"]["executed_attempts"] == 3
    assert audit["collateral_motion"]["maximum_displacement_m"] == pytest.approx(0.003)


@pytest.mark.parametrize(
    ("mutation", "failed_invariant"),
    [
        (
            lambda rows: rows[4]["public_details"].update(object_path="/World/StagingLeadShield"),
            "same_primary_shield_placed_at_25_then_moved_to_65",
        ),
        (
            lambda rows: rows[4]["public_details"].update(placement_fraction=0.50),
            "same_primary_shield_placed_at_25_then_moved_to_65",
        ),
        (
            lambda rows: rows[5]["public_details"].update(
                detector_path="/World/DetectorStations/RemoteDeconRoom"
            ),
            "protected_navigation_completed",
        ),
        (
            lambda rows: rows[6].update(duration_s=3.0),
            "two_second_measurement_completed",
        ),
        (
            lambda rows: rows[3]["public_details"]["collateral_motion_audit"][0].update(
                displacement_m=0.050001
            ),
            "collateral_motion_audited_and_bounded",
        ),
    ],
)
def test_complex_process_audit_rejects_wrong_target_or_collateral_motion(
    mutation: object,
    failed_invariant: str,
) -> None:
    plan, results = _passing_payload()
    mutated = deepcopy(results)
    mutation(mutated)
    audit = _audit(plan, mutated)
    assert failed_invariant in audit["failed_invariants"]
