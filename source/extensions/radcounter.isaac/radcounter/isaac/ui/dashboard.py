"""Operational UI for measurement, dose-map inspection, and runtime synchronization."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import omni.timeline
import omni.ui as ui
import omni.usd

from radcounter.core.natural_language import (
    AvailableAction,
    CommandContext,
    CommandName,
    CommandStep,
    normalize_operator_instruction,
)
from radcounter.core.system_profiles import (
    default_catalog_path,
    default_selection_path,
    load_active_selection,
    load_system_catalog,
    resolve_system_selection,
    save_active_selection,
)

from ..runtime.simulation import IsaacRadiationSimulation, measurement_payload
from ..visualization import (
    ActionVisualizer,
    DoseMapVisualizer,
    RayDebugVisualizer,
    ResidualVisualizer,
    SourceEstimateVisualizer,
)
from ..visualization._common import author_points, set_visibility
from .robot_monitor import RobotMonitorOverlay


def _bound_label(
    model: ui.AbstractValueModel,
    subscriptions: list[Any],
    **kwargs: object,
) -> ui.Label:
    """Create a Label whose text follows a value model on current Omni UI."""

    label = ui.Label(model.get_value_as_string(), **kwargs)

    def update(changed: ui.AbstractValueModel) -> None:
        label.text = changed.get_value_as_string()

    subscriptions.append(model.subscribe_value_changed_fn(update))
    return label


class RadCounterDashboard:
    def __init__(self, ext_id: str) -> None:
        self.ext_id = ext_id
        self.root = Path(__file__).resolve().parents[6]
        self.stage_path = self.root / "assets/environments/radcounter_vertical_slice.usda"
        self.config_path = self.root / "configs/scenarios/vertical_slice.runtime.json"
        self._system_display_name = "vertical slice"
        self._compose_default_scene = True
        self.artifact_path = self.root / "artifacts/ui/latest_measurement.json"
        self.workflow_artifact_path = self.root / "artifacts/ui/latest_workflow.json"
        self._label_subscriptions: list[Any] = []
        self.simulation: IsaacRadiationSimulation | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._status = ui.SimpleStringModel("Load the vertical-slice scene to begin.")
        self._measurement = ui.SimpleStringModel("No measurement")
        self._sources = ui.SimpleStringModel("Estimator-visible sources: 0")
        self._revisions = ui.SimpleStringModel("No radiation runtime")
        self._duration = ui.SimpleFloatModel(2.0)
        self._source_path = ui.SimpleStringModel("/World/HiddenContaminatedDrum")
        self._source_activity = ui.SimpleFloatModel(5.0e7)
        self._truth_authoring = ui.SimpleBoolModel(False)
        self._show_truth = ui.SimpleBoolModel(False)
        self._show_estimate = ui.SimpleBoolModel(True)
        self._show_uncertainty = ui.SimpleBoolModel(True)
        self._show_residual = ui.SimpleBoolModel(True)
        self._show_action = ui.SimpleBoolModel(True)
        self._estimate = ui.SimpleStringModel("No estimator result supplied")
        self._residual = ui.SimpleStringModel("No verification residual supplied")
        self._plan = ui.SimpleStringModel("No countermeasure plan supplied")
        self._ray_source_path = ui.SimpleStringModel("/World/ContaminatedFloor")
        self._ray_detector_path = ui.SimpleStringModel("/World/MeasurementRobot/Detector")
        self._ray_debug = ui.SimpleStringModel("No transport ray selected")
        self._candidate_id = ui.SimpleStringModel("")
        self._candidate_confirmation = ui.SimpleBoolModel(False)
        self._experiment_baseline = ui.SimpleStringModel("risk_aware_countermeasure")
        self._experiment_run_id = ui.SimpleStringModel("ui-evaluation")
        self._selection_file = default_selection_path()
        try:
            self._active_system_selection = load_active_selection(self._selection_file)
            self._system_catalog_path = self._active_system_selection.catalog_path
        except Exception:
            self._system_catalog_path = default_catalog_path()
            self._active_system_selection = resolve_system_selection(
                catalog_path=self._system_catalog_path
            )
        self._system_catalog_path, self._system_catalog = load_system_catalog(
            self._system_catalog_path
        )
        self._profile_ids = tuple(self._system_catalog.profiles)
        self._environment_ids = tuple(self._system_catalog.environments)
        self._robot_set_ids = tuple(self._system_catalog.robot_sets)
        self._detector_set_ids = tuple(self._system_catalog.detector_sets)
        self._system_summary = ui.SimpleStringModel("")
        self._system_controls: list[Any] = []
        self._updating_system_controls = False
        self._command_input = ui.SimpleStringModel("")
        self._command_status = ui.SimpleStringModel(
            "Local command model starts when the first instruction is submitted."
        )
        self._command_preview = ui.SimpleStringModel(
            "Example: Move to the protected area and measure for 2 seconds."
        )
        self._workflow_services: Any | None = None
        self._workflow_belief: Any | None = None
        self._command_candidates: dict[str, Any] = {}
        self._physical_command_queue: list[
            tuple[CommandStep, asyncio.Future[dict[str, object]]]
        ] = []
        self._navigation_status_prefix = "Measurement robot moving"
        from ..natural_language import NaturalLanguageCommandController

        self._natural_language = NaturalLanguageCommandController(
            self,
            audit_path=self.root / "artifacts/ui/natural_language_commands.jsonl",
        )
        self._robot_monitor = RobotMonitorOverlay(ext_id)
        self._robot_monitor.set_robot_list_changed_callback(self._rebuild_robot_list)
        self._robot_monitor.configure(self._active_system_selection)
        self._window = ui.Window(
            "RadInterAct Operations",
            width=540,
            height=1000,
            dockPreference=ui.DockPreference.RIGHT,
        )
        self._window.frame.set_build_fn(self._build)
        self._window.deferred_dock_in("Stage", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)

    def _operator_style(self, **properties: object) -> dict[str, object]:
        return properties

    def _build(self) -> None:
        with ui.ScrollingFrame(
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
        ):
            self._build_content()

    def _build_content(self) -> None:
        with ui.VStack(spacing=10, height=0):
            ui.Label("RADCOUNTER / OPERATIONS", style={"font_size": 18, "color": 0xFFE3B341})
            ui.Label(
                "Measurement -> intervention -> verification",
                style={"font_size": 12, "color": 0xFF9FA6AD},
            )
            ui.Separator(height=4)
            self._build_robot_monitor_controls()
            ui.Separator(height=4)
            ui.Label(
                "NATURAL LANGUAGE COMMAND / ROBOT LLM",
                style={"font_size": 11, "color": 0xFFE3B341},
            )
            ui.Label(
                "Describe the robot task in English. Change the environment, robot, "
                "and detector with the selectors below.",
                word_wrap=True,
                style=self._operator_style(font_size=12, color=0xFFB8BDC3),
            )
            ui.StringField(
                self._command_input,
                height=46,
                style=self._operator_style(font_size=14),
            )
            with ui.HStack(height=34, spacing=8):
                self._command_run_button = ui.Button(
                    "1. Interpret",
                    width=150,
                    clicked_fn=self._submit_natural_language,
                    style=self._operator_style(font_size=12),
                )
                self._command_confirm_button = ui.Button(
                    "2. Confirm & run",
                    width=250,
                    clicked_fn=self._confirm_natural_language,
                    style=self._operator_style(
                        font_size=12,
                        background_color=0xFFE3B341,
                        color=0xFF181B1F,
                    ),
                )
                self._command_cancel_button = ui.Button(
                    "Cancel",
                    width=80,
                    clicked_fn=self._cancel_natural_language,
                )
            self._command_confirm_button.enabled = False
            self._command_cancel_button.enabled = False
            _bound_label(
                self._command_status,
                self._label_subscriptions,
                word_wrap=True,
                height=62,
                style=self._operator_style(font_size=12),
            )
            _bound_label(
                self._command_preview,
                self._label_subscriptions,
                word_wrap=True,
                height=90,
                style=self._operator_style(font_size=12, color=0xFFD3D7DC),
            )
            ui.Separator(height=4)
            self._build_system_selector()
            ui.Separator(height=4)
            with ui.HStack(height=34, spacing=8):
                self._load_scene_button = ui.Button(
                    "Load vertical slice",
                    clicked_fn=lambda: self._schedule(self._load_scene()),
                )
                ui.Button("Play / Pause", clicked_fn=self._toggle_timeline)
            with ui.HStack(height=34, spacing=8):
                ui.Button("Initialize radiation", clicked_fn=self._initialize_runtime)
                ui.Button("Sync USD -> Embree", clicked_fn=self._synchronize)
            ui.Label("STATUS", style={"font_size": 11, "color": 0xFF6CB6FF})
            _bound_label(
                self._status,
                self._label_subscriptions,
                word_wrap=True,
                height=52,
            )
            _bound_label(self._sources, self._label_subscriptions, word_wrap=True)
            _bound_label(self._revisions, self._label_subscriptions, word_wrap=True)
            ui.Separator(height=4)
            ui.Label("SOURCE AUTHORING", style={"font_size": 11, "color": 0xFF6CB6FF})
            with ui.HStack(height=28, spacing=8):
                ui.Label("USD prim", width=86)
                ui.StringField(self._source_path)
            with ui.HStack(height=28, spacing=8):
                ui.Label("Activity [Bq]", width=86)
                ui.FloatDrag(self._source_activity, min=0.0, max=1.0e12, step=1.0e5)
            with ui.HStack(height=28, spacing=8):
                ui.CheckBox(self._truth_authoring, width=20)
                ui.Label("Allow explicit Truth-source authoring", word_wrap=True)
            ui.Button("Apply source activity", height=32, clicked_fn=self._apply_source_activity)
            ui.Separator(height=4)
            ui.Label("MEASUREMENT", style={"font_size": 11, "color": 0xFF6CB6FF})
            with ui.HStack(height=28, spacing=8):
                ui.Label("Integration [s]", width=110)
                ui.FloatDrag(self._duration, min=0.05, max=120.0, step=0.25)
            with ui.HStack(height=34, spacing=8):
                ui.Button("Measure stations", clicked_fn=self._measure)
                ui.Button("Render dose proxy", clicked_fn=self._render_dose_map)
            _bound_label(
                self._measurement,
                self._label_subscriptions,
                word_wrap=True,
                height=150,
            )
            ui.Separator(height=4)
            ui.Label("ESTIMATION", style={"font_size": 11, "color": 0xFF6CB6FF})
            ui.Label(
                "Uses the estimator bound to the current workflow and public measurements.",
                word_wrap=True,
                style=self._operator_style(font_size=11, color=0xFFB8BDC3),
            )
            ui.Button(
                "Run / update source estimate",
                height=32,
                clicked_fn=lambda: self._schedule(self._run_estimation()),
            )
            ui.Separator(height=4)
            ui.Label("COUNTERMEASURE", style={"font_size": 11, "color": 0xFF6CB6FF})
            with ui.HStack(height=28, spacing=8):
                ui.Label("Candidate ID", width=100)
                ui.StringField(self._candidate_id)
            with ui.HStack(height=32, spacing=8):
                ui.Button("Refresh", clicked_fn=self._select_first_candidate)
                ui.Button(
                    "Preview",
                    clicked_fn=lambda: self._schedule(self._preview_selected_candidate()),
                )
            with ui.HStack(height=28, spacing=8):
                ui.CheckBox(self._candidate_confirmation, width=20)
                ui.Label("Confirm one physical execution", word_wrap=True)
            ui.Button(
                "Execute selected candidate",
                height=32,
                clicked_fn=lambda: self._schedule(self._execute_selected_candidate()),
            )
            ui.Separator(height=4)
            ui.Label("VISUALIZATION", style={"font_size": 11, "color": 0xFF6CB6FF})
            for model, label in (
                (self._show_estimate, "Estimated sources"),
                (self._show_uncertainty, "Source uncertainty"),
                (self._show_residual, "Predicted / observed / normalized residual"),
                (self._show_action, "Action route and target annotations"),
                (self._show_truth, "Truth sources (requires authoring consent above)"),
            ):
                with ui.HStack(height=24, spacing=8):
                    ui.CheckBox(model, width=20)
                    ui.Label(label, word_wrap=True)
            ui.Button(
                "Apply layer visibility",
                height=30,
                clicked_fn=self._apply_visualization_visibility,
            )
            with ui.HStack(height=28, spacing=8):
                ui.Label("Ray source", width=86)
                ui.StringField(self._ray_source_path)
            with ui.HStack(height=28, spacing=8):
                ui.Label("Ray detector", width=86)
                ui.StringField(self._ray_detector_path)
            ui.Button("Inspect material path", height=30, clicked_fn=self._render_selected_ray)
            _bound_label(
                self._ray_debug,
                self._label_subscriptions,
                word_wrap=True,
                height=54,
            )
            ui.Separator(height=4)
            ui.Label("WORKFLOW OUTPUT", style={"font_size": 11, "color": 0xFF6CB6FF})
            ui.Label("ESTIMATE", style={"font_size": 10, "color": 0xFF9FA6AD})
            _bound_label(
                self._estimate,
                self._label_subscriptions,
                word_wrap=True,
                height=54,
            )
            ui.Label("RESIDUAL", style={"font_size": 10, "color": 0xFF9FA6AD})
            _bound_label(
                self._residual,
                self._label_subscriptions,
                word_wrap=True,
                height=64,
            )
            ui.Label("PLAN", style={"font_size": 10, "color": 0xFF9FA6AD})
            _bound_label(
                self._plan,
                self._label_subscriptions,
                word_wrap=True,
                height=64,
            )
            ui.Button(
                "Load latest workflow artifact",
                height=32,
                clicked_fn=self._load_workflow_artifact,
            )
            ui.Separator(height=4)
            ui.Label("EXPERIMENT", style={"font_size": 11, "color": 0xFF6CB6FF})
            with ui.HStack(height=28, spacing=8):
                ui.Label("Baseline", width=86)
                ui.StringField(self._experiment_baseline)
            with ui.HStack(height=28, spacing=8):
                ui.Label("Run ID", width=86)
                ui.StringField(self._experiment_run_id)
            ui.Button(
                "Save evaluation snapshot",
                height=32,
                clicked_fn=self._save_evaluation_snapshot,
            )
            ui.Separator(height=4)
            ui.Label("COUNTERMEASURE EXECUTION", style={"font_size": 11, "color": 0xFF6CB6FF})
            ui.Label(
                "Physical actions are accepted through /radcounter/execute_robot_task. "
                "The dashboard never teleports a countermeasure object.",
                word_wrap=True,
                style={"font_size": 12, "color": 0xFFB8BDC3},
            )
            ui.Button("Export latest measurement", height=34, clicked_fn=self._export_measurement)
            ui.Spacer(height=4)
            ui.Label(
                "Synthetic validation attenuation data - not calibrated for safety decisions",
                word_wrap=True,
                style={"font_size": 11, "color": 0xFFDD7A6B},
            )

    def _build_robot_monitor_controls(self) -> None:
        ui.Label(
            "ROBOT MONITOR",
            style={"font_size": 11, "color": 0xFF6CB6FF},
        )
        ui.Label(
            "Select View to follow a robot or Onboard for its camera. Manual camera "
            "movement disables tracking.",
            word_wrap=True,
            style=self._operator_style(font_size=12, color=0xFFB8BDC3),
        )
        with ui.HStack(height=30, spacing=8):
            ui.Button("Building overview", clicked_fn=self._robot_monitor.overview)
            ui.Button(
                "Free camera",
                clicked_fn=self._robot_monitor.stop_follow,
            )
        self._robot_list_frame = ui.Frame(height=0)
        self._robot_list_frame.set_build_fn(self._build_robot_rows)

    def _build_robot_rows(self) -> None:
        robots = self._robot_monitor.robots
        if not robots:
            ui.Label(
                "The current configuration has no robots.",
                style=self._operator_style(font_size=12, color=0xFF8E989F),
            )
            return
        with ui.VStack(spacing=4, height=0):
            for robot in robots:
                active = robot.robot_id == self._robot_monitor.active_robot_id
                with ui.HStack(height=30, spacing=6):
                    ui.Rectangle(
                        width=5,
                        style={"background_color": (0xFFF4BD55 if active else 0xFF48545C)},
                    )
                    ui.Label(
                        ("● " if active else "○ ") + robot.display_name,
                        width=250,
                        style=self._operator_style(
                            font_size=12,
                            color=0xFFF4E2B9 if active else 0xFFD2D8DC,
                        ),
                    )
                    ui.Button(
                        "View",
                        width=84,
                        clicked_fn=lambda robot_id=robot.robot_id: self._robot_monitor.follow_robot(
                            robot_id
                        ),
                    )
                    ui.Button(
                        "Onboard",
                        width=84,
                        clicked_fn=lambda robot_id=robot.robot_id: (
                            self._robot_monitor.onboard_robot(robot_id)
                        ),
                    )

    def _rebuild_robot_list(self) -> None:
        frame = getattr(self, "_robot_list_frame", None)
        if frame is not None:
            frame.rebuild()

    @staticmethod
    def _choice_index(values: tuple[str, ...], selected: str) -> int:
        try:
            return values.index(selected)
        except ValueError:
            return 0

    @staticmethod
    def _combo_index(combo: Any) -> int:
        value_model = combo.model.get_item_value_model()
        getter = getattr(value_model, "get_value_as_int", None)
        return int(getter() if getter is not None else value_model.as_int)

    @staticmethod
    def _set_combo_index(combo: Any, values: tuple[str, ...], selected: str) -> None:
        value_model = combo.model.get_item_value_model()
        value_model.set_value(RadCounterDashboard._choice_index(values, selected))

    @staticmethod
    def _choice_labels(entries: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(entry.display_name for entry in entries.values())

    def _build_system_selector(self) -> None:
        selection = self._active_system_selection
        ui.Label(
            "SYSTEM CONFIGURATION",
            style={"font_size": 11, "color": 0xFF6CB6FF},
        )
        ui.Label(
            "Choose a preset or select the environment, robot, and detector independently.",
            word_wrap=True,
            style=self._operator_style(font_size=12, color=0xFFB8BDC3),
        )
        rows = (
            (
                "Preset",
                self._profile_ids,
                self._system_catalog.profiles,
                selection.profile_id,
                "_profile_combo",
            ),
            (
                "Environment",
                self._environment_ids,
                self._system_catalog.environments,
                selection.environment_id,
                "_environment_combo",
            ),
            (
                "Robot",
                self._robot_set_ids,
                self._system_catalog.robot_sets,
                selection.robot_set_id,
                "_robot_set_combo",
            ),
            (
                "Detector",
                self._detector_set_ids,
                self._system_catalog.detector_sets,
                selection.detector_set_id,
                "_detector_set_combo",
            ),
        )
        for label, values, entries, selected, attribute in rows:
            with ui.HStack(height=28, spacing=8):
                ui.Label(label, width=92)
                combo = ui.ComboBox(
                    self._choice_index(values, selected),
                    *self._choice_labels(entries),
                )
            setattr(self, attribute, combo)
            self._system_controls.append(combo)
        self._profile_combo.model.add_item_changed_fn(self._on_profile_choice_changed)
        for combo in (
            self._environment_combo,
            self._robot_set_combo,
            self._detector_set_combo,
        ):
            combo.model.add_item_changed_fn(self._on_component_choice_changed)
        with ui.HStack(height=34, spacing=8):
            self._system_apply_button = ui.Button(
                "Apply selected configuration",
                clicked_fn=self._apply_system_selection,
                style=self._operator_style(
                    font_size=12,
                    background_color=0xFF4F86C6,
                    color=0xFFF6F8FA,
                ),
            )
            ui.Button("Restore current", width=110, clicked_fn=self._restore_system_selection)
        _bound_label(
            self._system_summary,
            self._label_subscriptions,
            word_wrap=True,
            height=48,
            style=self._operator_style(font_size=11, color=0xFFD3D7DC),
        )
        self._update_system_preview()

    def _selected_system_ids(self) -> tuple[str, str, str, str]:
        return (
            self._profile_ids[self._combo_index(self._profile_combo)],
            self._environment_ids[self._combo_index(self._environment_combo)],
            self._robot_set_ids[self._combo_index(self._robot_set_combo)],
            self._detector_set_ids[self._combo_index(self._detector_set_combo)],
        )

    def _resolve_system_draft(self) -> Any:
        profile, environment, robot_set, detector_set = self._selected_system_ids()
        return resolve_system_selection(
            catalog_path=self._system_catalog_path,
            profile_id=profile,
            environment_id=environment,
            robot_set_id=robot_set,
            detector_set_id=detector_set,
        )

    def _update_system_preview(self) -> None:
        try:
            selection = self._resolve_system_draft()
        except Exception as exc:
            self._system_summary.set_value(f"Choose a different combination: {exc}")
            if hasattr(self, "_system_apply_button"):
                self._system_apply_button.enabled = False
            return
        readiness = "Ready" if selection.environment_ready else "Environment data unavailable"
        self._system_summary.set_value(
            f"{selection.environment_entry.display_name} · "
            f"{selection.robot_set.display_name} · {selection.detector_set.display_name}\n"
            f"{readiness}"
        )
        if hasattr(self, "_system_apply_button"):
            can_prepare = bool(selection.environment_preparation_scripts)
            self._system_apply_button.enabled = selection.environment_ready or can_prepare
            self._system_apply_button.text = (
                "Apply selected configuration"
                if selection.environment_ready
                else "Prepare environment and apply"
            )

    def _on_profile_choice_changed(self, *_args: Any) -> None:
        if self._updating_system_controls:
            return
        profile_id = self._profile_ids[self._combo_index(self._profile_combo)]
        profile = self._system_catalog.profiles[profile_id]
        self._updating_system_controls = True
        try:
            self._set_combo_index(
                self._environment_combo, self._environment_ids, profile.environment
            )
            self._set_combo_index(self._robot_set_combo, self._robot_set_ids, profile.robot_set)
            self._set_combo_index(
                self._detector_set_combo, self._detector_set_ids, profile.detector_set
            )
        finally:
            self._updating_system_controls = False
        self._update_system_preview()

    def _on_component_choice_changed(self, *_args: Any) -> None:
        if not self._updating_system_controls:
            self._update_system_preview()

    def _sync_system_controls(self, selection: Any) -> None:
        self._updating_system_controls = True
        try:
            self._set_combo_index(self._profile_combo, self._profile_ids, selection.profile_id)
            self._set_combo_index(
                self._environment_combo, self._environment_ids, selection.environment_id
            )
            self._set_combo_index(
                self._robot_set_combo, self._robot_set_ids, selection.robot_set_id
            )
            self._set_combo_index(
                self._detector_set_combo, self._detector_set_ids, selection.detector_set_id
            )
        finally:
            self._updating_system_controls = False
        self._active_system_selection = selection
        self._update_system_preview()

    def _restore_system_selection(self) -> None:
        self._sync_system_controls(self._active_system_selection)

    def _set_system_controls_enabled(self, enabled: bool) -> None:
        for control in self._system_controls:
            control.enabled = enabled
        self._system_apply_button.enabled = enabled

    def _apply_system_selection(self) -> None:
        self._schedule(self._apply_system_selection_async())

    async def _prepare_selected_environment(self, selection: Any) -> Any:
        scripts = selection.environment_preparation_scripts
        if not scripts:
            raise FileNotFoundError(
                selection.environment_entry.setup_hint or "Environment data is unavailable"
            )
        project_python = self.root / ".venv/bin/python"
        python = project_python if project_python.is_file() else Path(sys.executable)
        for index, script in enumerate(scripts, start=1):
            if script.suffix != ".py" or self.root not in script.parents or not script.is_file():
                raise ValueError(f"Environment preparation script is not allowed: {script}")
            self._status.set_value(f"Preparing environment ({index}/{len(scripts)}): {script.stem}")
            process = await asyncio.create_subprocess_exec(
                str(python),
                str(script),
                cwd=str(self.root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=900.0)
            if process.returncode != 0:
                detail = stdout.decode(errors="replace")[-3000:]
                raise RuntimeError(f"{script.name} failed:\n{detail}")
        prepared = self._resolve_system_draft()
        if not prepared.environment_ready:
            raise FileNotFoundError(
                "Environment data is still unavailable after preparation: "
                f"{prepared.environment_source_path}"
            )
        return prepared

    async def _apply_system_selection_async(self) -> None:
        try:
            selection = self._resolve_system_draft()
        except Exception as exc:
            self._status.set_value(f"Cannot apply configuration: {type(exc).__name__}: {exc}")
            self._update_system_preview()
            return
        self._set_system_controls_enabled(False)
        self._natural_language.cancel()
        omni.timeline.get_timeline_interface().pause()
        self._status.set_value(f"Configuring {selection.profile.display_name}...")
        try:
            if not selection.environment_ready:
                selection = await self._prepare_selected_environment(selection)
            if selection.configurable:
                from ..system_profile import (
                    compose_selected_system,
                    prepare_environment_stage,
                )

                stage_path, _ = prepare_environment_stage(selection)
                context = omni.usd.get_context()
                result, error = await context.open_stage_async(str(stage_path))
                if not result:
                    raise RuntimeError(f"stage load failed: {error}")
                composed = compose_selected_system(
                    context.get_stage(), selection, stage_path=stage_path
                )
                simulation = IsaacRadiationSimulation.from_config(
                    context.get_stage(), composed.runtime_config_path
                )
                self.configure_system_paths(
                    stage_path=composed.stage_path,
                    config_path=composed.runtime_config_path,
                    display_name=selection.profile.display_name,
                    simulation=simulation,
                    selection=selection,
                )
                self._sources.set_value(
                    f"Estimator-visible sources: {len(simulation.belief_source_paths)}"
                )
                self._revisions.set_value(
                    f"Embree geometries: {len(simulation.transport.geometry_paths)}"
                )
            else:
                source = selection.environment_source_path
                if source is None:
                    raise RuntimeError("default environment must be a local USD stage")
                self.stage_path = source
                self.config_path = selection.runtime_config_path
                self._system_display_name = selection.profile.display_name
                self._compose_default_scene = True
                self._load_scene_button.text = f"Reload {selection.profile.display_name}"
                await self._load_scene()
                self._initialize_runtime()
                if self.simulation is None:
                    raise RuntimeError("radiation runtime did not initialize")

            save_active_selection(
                self._selection_file,
                catalog_path=selection.catalog_path,
                profile_id=selection.profile_id,
                environment_id=selection.environment_id,
                robot_set_id=selection.robot_set_id,
                detector_set_id=selection.detector_set_id,
            )
            self._sync_system_controls(selection)
            if not selection.configurable:
                self._robot_monitor.configure(selection)
            self._status.set_value(f"Configuration applied: {selection.profile.display_name}")
        except Exception as exc:
            self._status.set_value(f"Failed to apply configuration: {type(exc).__name__}: {exc}")
        finally:
            self._set_system_controls_enabled(True)
            self._update_system_preview()

    def _schedule(self, coroutine) -> None:
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _set_command_busy(self, busy: bool) -> None:
        self._command_run_button.enabled = not busy
        if busy:
            self._command_confirm_button.enabled = False
            self._command_cancel_button.enabled = False

    def configure_system_paths(
        self,
        *,
        stage_path: str | Path,
        config_path: str | Path,
        display_name: str,
        simulation: IsaacRadiationSimulation | None = None,
        selection: Any | None = None,
    ) -> None:
        """Bind the dashboard reload action to a generated system-profile stage."""

        self.stage_path = Path(stage_path).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        self._system_display_name = display_name
        self._compose_default_scene = False
        self.simulation = simulation
        self._workflow_services = None
        self._workflow_belief = None
        self._command_candidates.clear()
        if hasattr(self, "_load_scene_button"):
            self._load_scene_button.text = f"Reload {display_name}"
        self._status.set_value(f"Loaded {display_name}.")
        if selection is not None:
            if hasattr(self, "_profile_combo"):
                self._sync_system_controls(selection)
            else:
                # A headless validation session does not build deferred Omni UI
                # widgets.  Keep the selected system state authoritative without
                # requiring controls that do not exist in that execution mode.
                self._active_system_selection = selection
                readiness = (
                    "Ready" if selection.environment_ready else "Environment data unavailable"
                )
                self._system_summary.set_value(
                    f"{selection.environment_entry.display_name} · "
                    f"{selection.robot_set.display_name} · "
                    f"{selection.detector_set.display_name}\n{readiness}"
                )
            self._robot_monitor.configure(selection)

    @staticmethod
    def _format_command_plan(submission) -> str:
        lines = [submission.validated.plan.summary]
        for index, step in enumerate(submission.validated.plan.steps, start=1):
            detail = step.command.value
            if step.candidate_id:
                detail += f" · {step.candidate_id}"
            if step.duration_s is not None:
                detail += f" · {step.duration_s:g} s"
            if step.max_attempts > 1:
                detail += f" · max {step.max_attempts} attempts"
            if step.until is not None:
                detail += f" · until {step.until.criterion.value}={step.until.threshold:g}"
            lines.append(f"{index}. {detail}")
        return "\n".join(lines)

    @staticmethod
    def _format_command_completion(submission: Any) -> str:
        motions: list[Mapping[str, object]] = []
        measurement_duration_s: float | None = None
        returned_home = False
        unmet_conditions: list[Mapping[str, object]] = []
        for result in submission.results:
            if not isinstance(result, Mapping):
                continue
            completion = result.get("completion_condition")
            if (
                isinstance(completion, Mapping)
                and completion.get("met") is False
                and result.get("attempt") == result.get("max_attempts")
            ):
                unmet_conditions.append(completion)
            command = result.get("command")
            if command == CommandName.MEASURE.value:
                duration = result.get("duration_s")
                if isinstance(duration, (int, float)):
                    measurement_duration_s = float(duration)
            if command == CommandName.RETURN_MEASUREMENT_ROBOT.value:
                returned_home = True
            public_details = result.get("public_details")
            if not isinstance(public_details, Mapping):
                continue
            motion = public_details.get("motion_audit")
            if isinstance(motion, Mapping):
                motions.append(motion)
        if unmet_conditions:
            last = unmet_conditions[-1]
            return (
                "The bounded workflow stopped at its attempt limit without meeting "
                f"the completion condition: {last.get('criterion')} "
                f"observed={last.get('observed')} target={last.get('threshold')}."
            )
        if motions:
            distances = [
                float(motion["displacement_m"])
                for motion in motions
                if isinstance(motion.get("displacement_m"), (int, float))
            ]
            final = motions[-1].get("final_position_m")
            measurement = (
                ""
                if measurement_duration_s is None
                else f" -> measured for {measurement_duration_s:g} seconds"
            )
            if returned_home and len(distances) >= 2:
                final_text = ""
                if isinstance(final, (tuple, list)) and len(final) >= 2:
                    final_text = f" Final position ({float(final[0]):.2f}, {float(final[1]):.2f})."
                return (
                    f"Three steps complete: moved {distances[0]:.2f} m{measurement}, "
                    f"then returned {distances[-1]:.2f} m to the start.{final_text}"
                )
            initial = motions[0].get("initial_position_m")
            if (
                isinstance(initial, (tuple, list))
                and len(initial) >= 2
                and isinstance(final, (tuple, list))
                and len(final) >= 2
                and distances
            ):
                return (
                    f"Moved {distances[0]:.2f} m: "
                    f"({float(initial[0]):.2f}, {float(initial[1]):.2f}) → "
                    f"({float(final[0]):.2f}, {float(final[1]):.2f}){measurement}."
                )
        return "Movement and measurement completed."

    def update_navigation_progress(
        self,
        step: int,
        position_m: tuple[float, float, float],
        target_xy_m: tuple[float, float],
        remaining_m: float,
    ) -> None:
        """Show observable articulation progress while the robot is moving."""

        self._command_status.set_value(
            f"{self._navigation_status_prefix}: step {step} · "
            f"position ({position_m[0]:.2f}, {position_m[1]:.2f}) -> "
            f"target ({target_xy_m[0]:.2f}, {target_xy_m[1]:.2f}) · "
            f"{remaining_m:.2f} m remaining"
        )
        self._robot_monitor.update_navigation_progress(
            position_m=position_m,
            target_xy_m=target_xy_m,
            remaining_m=remaining_m,
        )

    def update_countermeasure_progress(self, event: Mapping[str, object]) -> None:
        """Forward physical countermeasure phases to the viewport monitor."""

        self._robot_monitor.update_countermeasure_progress(event)

    def robot_monitor_audit(self) -> dict[str, object]:
        return self._robot_monitor.audit()

    def _submit_natural_language(self) -> None:
        raw_instruction = self._command_input.get_value_as_string()
        instruction = normalize_operator_instruction(raw_instruction)
        if not instruction:
            self._command_status.set_value("Enter an instruction in English.")
            return
        self._command_input.set_value("")
        self._natural_language.cancel()
        self._schedule(
            self._interpret_natural_language(
                instruction,
                duplicate_removed=instruction != raw_instruction.strip(),
            )
        )

    def set_natural_language_input(self, value: str) -> None:
        """Update the visible operator field without synthesizing keyboard input."""

        self._command_input.set_value(value)

    def show_building_overview(self) -> None:
        """Select the existing low-cost overview camera for scripted demonstrations."""

        self._robot_monitor.overview()

    async def _interpret_natural_language(
        self,
        instruction: str,
        *,
        duplicate_removed: bool = False,
    ) -> None:
        try:
            await self.interpret_natural_language_instruction(
                instruction,
                duplicate_removed=duplicate_removed,
            )
        except Exception:
            # The public method records the operator-facing error before
            # re-raising. Button callbacks must not leave an unobserved task.
            return

    async def interpret_natural_language_instruction(
        self,
        instruction: str,
        *,
        duplicate_removed: bool = False,
    ):
        """Interpret one instruction while updating the same visible UI as the button."""

        self._set_command_busy(True)
        prefix = "Duplicate input was reduced to one copy. " if duplicate_removed else ""
        self._command_status.set_value(f"{prefix}Interpreting with the local model...")
        self._command_preview.set_value(instruction)
        try:
            submission = await self._natural_language.submit(instruction)
            self._command_preview.set_value(self._format_command_plan(submission))
            if submission.executed:
                self._command_status.set_value("Command completed.")
            else:
                self._command_status.set_value(
                    "Interpretation complete. Select the yellow '2. Confirm & run' "
                    "button to move the robot."
                )
                self._command_confirm_button.enabled = True
                self._command_cancel_button.enabled = True
        except Exception as exc:
            self._command_status.set_value(f"Command rejected: {type(exc).__name__}: {exc}")
            raise
        finally:
            self._command_run_button.enabled = True
        return submission

    def _confirm_natural_language(self) -> None:
        self._schedule(self._execute_confirmed_natural_language())

    async def _execute_confirmed_natural_language(self) -> None:
        try:
            await self.execute_confirmed_natural_language_instruction()
        except Exception:
            # The public method already rendered the failure in the dashboard.
            return

    async def execute_confirmed_natural_language_instruction(self):
        """Execute the pending plan while updating the operator-facing UI."""

        self._set_command_busy(True)
        self._command_status.set_value("Executing confirmed operation...")
        try:
            submission = await self._natural_language.confirm()
            self._command_preview.set_value(self._format_command_plan(submission))
            self._command_status.set_value(self._format_command_completion(submission))
        except Exception as exc:
            self._command_status.set_value(f"Execution failed: {type(exc).__name__}: {exc}")
            raise
        finally:
            self._command_run_button.enabled = True
            self._command_confirm_button.enabled = False
            self._command_cancel_button.enabled = False
        return submission

    def _cancel_natural_language(self) -> None:
        self._natural_language.cancel()
        self._command_confirm_button.enabled = False
        self._command_cancel_button.enabled = False
        self._command_status.set_value("Pending operation cancelled.")

    async def submit_natural_language_instruction(
        self,
        instruction: str,
        *,
        confirm_physical: bool = False,
    ):
        """Submit an instruction through the same controller used by the UI."""

        submission = await self._natural_language.submit(instruction)
        if not submission.executed and confirm_physical:
            submission = await self._natural_language.confirm()
        return submission

    async def confirm_natural_language_instruction(self):
        """Confirm a previously interpreted plan through the dashboard controller."""

        return await self._natural_language.confirm()

    async def _load_scene(self) -> None:
        self._status.set_value(f"Loading {self._system_display_name}...")
        context = omni.usd.get_context()
        result, error = await context.open_stage_async(str(self.stage_path))
        if not result:
            self._status.set_value(f"Stage load failed: {error}")
            return
        if not self._compose_default_scene:
            self._workflow_services = None
            self._workflow_belief = None
            self._command_candidates.clear()
            try:
                self.simulation = IsaacRadiationSimulation.from_config(
                    context.get_stage(), self.config_path
                )
            except Exception as exc:
                self.simulation = None
                self._status.set_value(f"Profile reload failed: {type(exc).__name__}: {exc}")
                return
            self._status.set_value(f"Reloaded {self._system_display_name}.")
            return
        try:
            from omni.kit.app import get_app

            from ..robot.real_robots import (
                RealRobotAssetConfig,
                add_real_robot_references,
                author_real_robot_task_scene,
                create_decontamination_activity_map,
                enable_real_robot_extensions,
            )

            enable_real_robot_extensions()
            for _ in range(20):
                await get_app().next_update_async()
            stage = context.get_stage()
            config = RealRobotAssetConfig()
            add_real_robot_references(stage, config=config)
            for _ in range(120):
                await get_app().next_update_async()
            activity_path = create_decontamination_activity_map(
                self.root / "artifacts/ui/runtime_dashboard_activity.npz"
            )
            author_real_robot_task_scene(
                stage,
                activity_path,
                self.root / "configs/decontamination/concrete_surface.synthetic.yaml",
                config=config,
            )
            for _ in range(20):
                await get_app().next_update_async()
        except Exception as exc:
            self._status.set_value(
                f"Articulated scene composition failed: {type(exc).__name__}: {exc}"
            )
            return
        self.simulation = None
        self._workflow_services = None
        self._workflow_belief = None
        self._command_candidates.clear()
        self._status.set_value("Loaded Ridgeback+Franka and Nova Carter with IK task tooling.")

    @staticmethod
    def _toggle_timeline() -> None:
        timeline = omni.timeline.get_timeline_interface()
        timeline.pause() if timeline.is_playing() else timeline.play()

    def _initialize_runtime(self) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            self._status.set_value("No USD stage is open.")
            return
        try:
            self.simulation = IsaacRadiationSimulation.from_config(stage, self.config_path)
        except Exception as exc:
            self._status.set_value(f"Runtime initialization failed: {type(exc).__name__}: {exc}")
            return
        self._sources.set_value(
            f"Estimator-visible sources: {len(self.simulation.belief_source_paths)}"
        )
        self._revisions.set_value(
            f"Embree geometries: {len(self.simulation.transport.geometry_paths)}"
        )
        self._status.set_value(
            "Radiation runtime ready. Hidden Truth sources remain estimator-inaccessible."
        )

    def _synchronize(self) -> None:
        if self.simulation is None:
            self._initialize_runtime()
        if self.simulation is None:
            return
        changed = self.simulation.synchronize()
        self._revisions.set_value(f"Last geometry update: {len(changed)} prim(s)")
        self._status.set_value(
            "USD activity and actual geometry poses synchronized to radiation runtime."
        )

    def _apply_source_activity(self) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            self._status.set_value("No USD stage is open.")
            return
        path = self._source_path.get_value_as_string().strip()
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            self._status.set_value(f"Source prim does not exist: {path}")
            return
        role = prim.GetAttribute("rad:role")
        if not role or str(role.Get()) not in {"source", "contaminated_surface"}:
            self._status.set_value(f"Prim is not a radiation source: {path}")
            return
        hidden = prim.GetAttribute("rad:source:hiddenFromEstimator")
        if hidden and bool(hidden.Get()) and not self._truth_authoring.get_value_as_bool():
            self._status.set_value(
                "Hidden source editing requires explicit Truth-source authoring consent."
            )
            return
        activity = prim.GetAttribute("rad:source:activityBq")
        if not activity or not activity.IsValid():
            self._status.set_value(
                "This source uses an NPZ surface map; edit it through the activity-map API."
            )
            return
        value = self._source_activity.get_value_as_float()
        if not math.isfinite(value) or value < 0:
            self._status.set_value("Activity must be finite and nonnegative.")
            return
        activity.Set(float(value))
        if self.simulation is not None:
            self.simulation.refresh_scene_state()
        self._status.set_value(f"Set {path} activity to {value:.6g} Bq.")

    def _measure(self) -> None:
        if self.simulation is None:
            self._initialize_runtime()
        if self.simulation is None:
            return
        records = self.simulation.measure(duration_s=self._duration.get_value_as_float())
        self._latest_records = records
        lines = [
            f"{Path(record.detector_path).name}: {record.counts} counts / "
            f"{record.measured_rate_cps:.2f} cps (expected {record.expected_rate_cps:.2f})"
            for record in records
        ]
        self._measurement.set_value("\n".join(lines))
        self._status.set_value(f"Integrated {len(records)} detectors without exposing Truth state.")

    @staticmethod
    def _color_map(values: np.ndarray) -> list[tuple[float, float, float]]:
        logarithm = np.log10(np.maximum(values, 1.0e-9))
        span = float(np.ptp(logarithm))
        normalized = np.zeros_like(logarithm) if span == 0 else (logarithm - logarithm.min()) / span
        return [
            (
                float(min(1.0, 2.0 * value)),
                float(1.0 - abs(2.0 * value - 1.0)),
                float(min(1.0, 2.0 * (1.0 - value))),
            )
            for value in normalized
        ]

    def _render_dose_map(self) -> None:
        if self.simulation is None:
            self._initialize_runtime()
        if self.simulation is None:
            return
        grid = np.asarray(
            [(x, y, 0.65) for y in np.linspace(-3.4, 3.4, 18) for x in np.linspace(-5.4, 5.4, 28)]
        )
        values = self.simulation.dose_proxy_map(grid)
        stage = omni.usd.get_context().get_stage()
        DoseMapVisualizer(stage).author(
            grid,
            values,
            data_class="count_rate_proxy_not_dose_calibrated",
        )
        self._status.set_value(f"Rendered {len(grid)} count-rate proxy samples.")

    @staticmethod
    def _world_position(stage: Any, path: str) -> np.ndarray:
        from pxr import Gf, UsdGeom

        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise ValueError(f"USD prim does not exist: {path}")
        return np.asarray(
            UsdGeom.XformCache().GetLocalToWorldTransform(prim).Transform(Gf.Vec3d()),
            dtype=np.float64,
        )

    def _source_position(self, path: str) -> np.ndarray:
        if self.simulation is None:
            raise RuntimeError("radiation runtime is not initialized")
        source = next(
            (item for item in self.simulation.sources if item.prim_path == path),
            None,
        )
        if source is None:
            raise ValueError(f"runtime source does not exist: {path}")
        total = float(np.sum(source.activity_bq))
        if total <= 0.0:
            raise ValueError(f"runtime source has no positive activity: {path}")
        return np.average(source.positions_m, axis=0, weights=source.activity_bq)

    def _render_selected_ray(self) -> None:
        if self.simulation is None:
            self._initialize_runtime()
        if self.simulation is None:
            return
        stage = omni.usd.get_context().get_stage()
        source_path = self._ray_source_path.get_value_as_string().strip()
        detector_path = self._ray_detector_path.get_value_as_string().strip()
        try:
            origin = self._source_position(source_path)
            target = self._world_position(stage, detector_path)
            batch = self.simulation.transport.trace_path_lengths(origin[None, :], target[None, :])
            if bool(batch.error_flags[0]):
                raise RuntimeError("transport backend reported a ray-tracing error")
            material_lengths = {
                material_id: float(batch.lengths_m[0, index])
                for index, material_id in enumerate(batch.material_ids)
            }
            RayDebugVisualizer(stage).author(origin, target, material_lengths)
        except Exception as exc:
            self._ray_debug.set_value(f"Ray inspection failed: {type(exc).__name__}: {exc}")
            return
        active = [
            f"{material_id}={length:.4g} m"
            for material_id, length in material_lengths.items()
            if length > 0.0
        ]
        self._ray_debug.set_value(
            f"{source_path} -> {detector_path}\n"
            + (", ".join(active) if active else "no attenuating material intersections")
        )
        self._status.set_value("Rendered the selected finite ray and material path lengths.")

    def _author_truth_overlay(self, stage: Any) -> None:
        if self.simulation is None:
            raise RuntimeError("radiation runtime is not initialized")
        if not self.simulation.sources:
            raise RuntimeError("the radiation runtime contains no source samples")
        positions = np.concatenate([source.positions_m for source in self.simulation.sources])
        activity = np.concatenate([source.activity_bq for source in self.simulation.sources])
        maximum = max(float(np.max(activity)), np.finfo(np.float64).eps)
        widths = 0.05 + 0.16 * np.sqrt(activity / maximum)
        author_points(
            stage,
            "/World/RadInterActVisualization/TruthSources",
            positions,
            colors_rgb=np.tile((0.96, 0.12, 0.08), (len(positions), 1)),
            widths_m=widths,
            data_class="truth_source_debug_authorized",
        )

    def _apply_visualization_visibility(self) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            self._status.set_value("No USD stage is open.")
            return
        show_truth = self._show_truth.get_value_as_bool()
        if show_truth and not self._truth_authoring.get_value_as_bool():
            self._show_truth.set_value(False)
            show_truth = False
            self._status.set_value(
                "Truth overlay requires explicit Truth-source authoring consent."
            )
        if show_truth:
            try:
                self._author_truth_overlay(stage)
            except Exception as exc:
                self._show_truth.set_value(False)
                show_truth = False
                self._status.set_value(f"Truth overlay failed: {type(exc).__name__}: {exc}")
        visibility = {
            "/World/RadInterActVisualization/TruthSources": show_truth,
            "/World/RadInterActVisualization/BeliefSources": (
                self._show_estimate.get_value_as_bool()
            ),
            "/World/RadInterActVisualization/SourceUncertainty": (
                self._show_uncertainty.get_value_as_bool()
            ),
            "/World/RadInterActVisualization/PredictedPostAction": (
                self._show_residual.get_value_as_bool()
            ),
            "/World/RadInterActVisualization/ObservedPostAction": (
                self._show_residual.get_value_as_bool()
            ),
            "/World/RadInterActVisualization/NormalizedResidual": (
                self._show_residual.get_value_as_bool()
            ),
            "/World/RadInterActVisualization/Action": self._show_action.get_value_as_bool(),
        }
        for path, visible in visibility.items():
            set_visibility(stage, path, visible)

    def _render_action_candidate(self, candidate: Any) -> None:
        stage = omni.usd.get_context().get_stage()
        action = candidate.action
        parameters = action.parameters
        route_parts: list[list[float]] = []
        for name in ("base_route_m", "pickup_base_route_m", "placement_base_route_m"):
            values = parameters.get(name)
            if isinstance(values, (list, tuple)):
                route_parts.extend(values)
        route = np.asarray(route_parts, dtype=np.float64) if len(route_parts) >= 2 else None
        tool_values = parameters.get(
            "tool_path_world_m", parameters.get("decon_tool_waypoints_world_m")
        )
        tool_path = None
        if isinstance(tool_values, (list, tuple)) and len(tool_values) >= 2:
            tool_path = np.asarray(tool_values, dtype=np.float64)
        target = (
            None
            if action.target_pose_world is None
            else np.asarray(action.target_pose_world[:3, 3], dtype=np.float64)
        )
        visualizer = ActionVisualizer(stage)
        visualizer.clear()
        visualizer.author(
            base_route_world_m=route,
            tool_path_world_m=tool_path,
            target_world_m=target,
        )
        self._apply_visualization_visibility()

    def _render_workflow_layers(self, view: Mapping[str, object]) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        estimate = view.get("estimate")
        if isinstance(estimate, Mapping):
            positions = estimate.get("positions_world_m")
            activity = estimate.get("source_strength_bq", estimate.get("activity_bq"))
            uncertainty = estimate.get("activity_standard_deviation_bq")
            if positions is not None and activity is not None:
                SourceEstimateVisualizer(stage).author(
                    positions,
                    activity,
                    activity_standard_deviation_bq=uncertainty,
                )
        residual = view.get("residual")
        if isinstance(residual, Mapping):
            paths = residual.get("detector_paths")
            predicted = residual.get("predicted_rate_cps")
            observed = residual.get("observed_rate_cps")
            normalized = residual.get("normalized_residual")
            if (
                isinstance(paths, (list, tuple))
                and predicted is not None
                and observed is not None
                and normalized is not None
            ):
                positions = np.asarray([self._world_position(stage, str(path)) for path in paths])
                ResidualVisualizer(stage).author(positions, predicted, observed, normalized)
        selected = view.get("selected_action")
        if isinstance(selected, Mapping):
            action_id = selected.get("action_id")
            candidate = self._command_candidates.get(str(action_id))
            if candidate is not None:
                self._render_action_candidate(candidate)
        self._apply_visualization_visibility()

    def _export_measurement(self) -> None:
        records = getattr(self, "_latest_records", None)
        if not records:
            self._status.set_value("Run a measurement before export.")
            return
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_path.write_text(
            json.dumps(measurement_payload(records), indent=2) + "\n", encoding="utf-8"
        )
        self._status.set_value(f"Wrote {self.artifact_path.relative_to(self.root)}")

    @staticmethod
    def _compact(value: object, limit: int = 700) -> str:
        encoded = json.dumps(value, sort_keys=True, default=str)
        return encoded if len(encoded) <= limit else encoded[: limit - 3] + "..."

    def set_workflow_view(self, view: Mapping[str, object]) -> None:
        """Display outputs supplied by an estimator/workflow without owning either."""

        estimate = view.get(
            "estimate",
            {
                "visible_basis_count": view.get("belief_visible_basis_count", 0),
                "revision": view.get("revision"),
            },
        )
        residual = view.get("residual", "No residual")
        plan = {
            "selected_action": view.get("selected_action"),
            "prediction": view.get("prediction"),
        }
        self._estimate.set_value(self._compact(estimate))
        self._residual.set_value(self._compact(residual))
        self._plan.set_value(self._compact(plan))
        try:
            self._render_workflow_layers(view)
        except (RuntimeError, TypeError, ValueError) as exc:
            self._status.set_value(
                f"Workflow visualization rejected invalid data: {type(exc).__name__}: {exc}"
            )

    def _load_workflow_artifact(self) -> None:
        if not self.workflow_artifact_path.is_file():
            self._status.set_value(
                f"Workflow artifact not found: {self.workflow_artifact_path.relative_to(self.root)}"
            )
            return
        try:
            payload = json.loads(self.workflow_artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._status.set_value(f"Workflow artifact load failed: {exc}")
            return
        if not isinstance(payload, Mapping):
            self._status.set_value("Workflow artifact must contain a JSON object.")
            return
        self.set_workflow_view(payload)
        self._status.set_value(f"Loaded {self.workflow_artifact_path.relative_to(self.root)}")

    def bind_workflow(self, services: Any, belief: Any) -> None:
        """Expose a live, truth-free workflow to natural-language commands."""

        self._workflow_services = services
        self._workflow_belief = belief
        self._refresh_command_candidates()
        self._select_first_candidate()
        self._command_status.set_value("English natural-language control is ready")

    async def _run_estimation(self) -> None:
        services = self._workflow_services
        if services is None:
            self._status.set_value("Bind a workflow before running estimation.")
            return
        try:
            measurement = services.last_measurement
            if not measurement:
                measurement = await services.measure()
            belief = await asyncio.to_thread(
                services.estimate,
                measurement,
                self._workflow_belief,
            )
        except Exception as exc:
            self._status.set_value(f"Estimation failed: {type(exc).__name__}: {exc}")
            return
        self._workflow_belief = belief
        self.set_workflow_view(services.workflow_view())
        self._refresh_command_candidates()
        self._status.set_value("Updated the public source estimate and uncertainty layers.")

    def _select_first_candidate(self) -> None:
        actions = self._refresh_command_candidates()
        feasible = next((action for action in actions if action.feasible), None)
        self._candidate_id.set_value("" if feasible is None else feasible.action_id)
        if feasible is None:
            self._status.set_value("No feasible scene-derived action is available.")

    def _selected_candidate(self) -> Any:
        self._refresh_command_candidates()
        action_id = self._candidate_id.get_value_as_string().strip()
        candidate = self._command_candidates.get(action_id)
        if candidate is None:
            raise ValueError(f"candidate is not available in the current scene: {action_id}")
        from radcounter.core.planning import DeterministicFeasibilityChecker

        services = self._workflow_services
        if services is None:
            raise RuntimeError("workflow services are not bound")
        report = DeterministicFeasibilityChecker().evaluate(candidate, services.resources)
        if not report.feasible:
            raise ValueError(f"candidate is infeasible: {action_id}: {', '.join(report.reasons)}")
        return candidate

    async def _preview_selected_candidate(self) -> None:
        services = self._workflow_services
        belief = self._workflow_belief
        if services is None or belief is None:
            self._status.set_value("Bind a workflow before previewing an action.")
            return
        try:
            candidate = self._selected_candidate()
            await asyncio.sleep(0)
            prediction = services.preview(candidate.action, belief)
            self._render_action_candidate(candidate)
        except Exception as exc:
            self._status.set_value(f"Preview failed: {type(exc).__name__}: {exc}")
            return
        self.set_workflow_view(services.workflow_view())
        self._status.set_value(
            f"Previewed {candidate.action.action_id}: {self._compact(prediction)}"
        )

    async def _execute_selected_candidate(self) -> None:
        if not self._candidate_confirmation.get_value_as_bool():
            self._status.set_value("Confirm one physical execution before running the action.")
            return
        try:
            candidate = self._selected_candidate()
            result = await self.execute_natural_language_step(
                CommandStep(
                    command=CommandName.EXECUTE_CANDIDATE,
                    candidate_id=candidate.action.action_id,
                )
            )
        except Exception as exc:
            self._status.set_value(f"Action execution failed: {type(exc).__name__}: {exc}")
            return
        finally:
            self._candidate_confirmation.set_value(False)
        self._status.set_value(f"Physical action completed: {self._compact(result)}")
        self._select_first_candidate()

    def _save_evaluation_snapshot(self) -> None:
        run_id = self._experiment_run_id.get_value_as_string().strip()
        baseline = self._experiment_baseline.get_value_as_string().strip()
        if not run_id or re.fullmatch(r"[A-Za-z0-9._-]+", run_id) is None:
            self._status.set_value(
                "Run ID may contain only letters, numbers, dot, dash, underscore."
            )
            return
        if not baseline:
            self._status.set_value("Experiment baseline must not be empty.")
            return
        stage = omni.usd.get_context().get_stage()
        root_layer = None if stage is None else stage.GetRootLayer()
        stage_identifier = None if root_layer is None else str(root_layer.identifier)
        records = tuple(getattr(self, "_latest_records", ()))
        services = self._workflow_services
        config_digest = (
            hashlib.sha256(self.config_path.read_bytes()).hexdigest()
            if self.config_path.is_file()
            else None
        )
        payload = {
            "schema_version": 1,
            "artifact_type": "operator_evaluation_snapshot",
            "evidence_class": "configuration_and_public_observations",
            "run_id": run_id,
            "baseline": baseline,
            "stage_identifier": stage_identifier,
            "runtime_config": str(self.config_path),
            "runtime_config_sha256": config_digest,
            "measurements": [
                {
                    "detector_path": str(record.detector_path),
                    "duration_s": float(record.duration_s),
                    "counts": int(record.counts),
                    "measured_rate_cps": float(record.measured_rate_cps),
                }
                for record in records
            ],
            "workflow": None if services is None else services.workflow_view(),
        }
        destination = self.root / "artifacts/ui/evaluations" / f"{run_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        self._status.set_value(
            f"Saved public evaluation snapshot: {destination.relative_to(self.root)}"
        )

    @staticmethod
    def _candidate_label(candidate: Any) -> str:
        action = candidate.action
        target = "scene target"
        if action.target_prim_path:
            target = Path(str(action.target_prim_path)).name
        labels = {
            "measure": "move to station; measuring requires a separate step",
            "decontaminate": "decontaminate",
            "place_shield": "place shield",
            "move_shield": "move shield",
            "move_object": "move object",
            "remove_object": "remove object",
            "repair_action": "repair action",
        }
        action_type = str(action.action_type)
        details: list[str] = []
        if action_type == "decontaminate":
            details.append(str(action.parameters.get("decon_profile", "full serpentine raster")))
        if action_type in {"place_shield", "move_shield"}:
            fraction = action.parameters.get("placement_fraction")
            if isinstance(fraction, (int, float)):
                details.append(f"source→protected line {100.0 * float(fraction):g}%")
        suffix = "" if not details else " · " + " · ".join(details)
        return f"{labels.get(action_type, action_type)} · {target}{suffix}"

    def _refresh_command_candidates(self) -> tuple[AvailableAction, ...]:
        services = self._workflow_services
        belief = self._workflow_belief
        if services is None or belief is None:
            self._command_candidates.clear()
            return ()
        from radcounter.core.planning import DeterministicFeasibilityChecker

        checker = DeterministicFeasibilityChecker()
        candidates = services.candidate_generator.generate_all(belief, services.last_diagnosis)
        self._command_candidates = {item.action.action_id: item for item in candidates}
        stage = omni.usd.get_context().get_stage()

        def target_label(item: Any) -> str | None:
            path = item.action.target_prim_path
            if stage is None or path is None:
                return None
            prim = stage.GetPrimAtPath(str(path))
            if not prim or not prim.IsValid():
                return None
            display_name = str(prim.GetDisplayName() or "").strip()
            return display_name or prim.GetName()

        return tuple(
            AvailableAction(
                action_id=item.action.action_id,
                action_type=str(item.action.action_type),
                label=self._candidate_label(item),
                target=(
                    None
                    if item.action.target_prim_path is None
                    else str(item.action.target_prim_path)
                ),
                target_label=target_label(item),
                robot_id=str(item.action.robot_id),
                execution_mode=(
                    str(item.action.parameters["decon_profile"])
                    if "decon_profile" in item.action.parameters
                    else None
                ),
                placement_fraction=(
                    float(item.action.parameters["placement_fraction"])
                    if "placement_fraction" in item.action.parameters
                    else None
                ),
                predicted_duration_s=float(item.action.predicted_duration_s),
                feasible=checker.evaluate(item, services.resources).feasible,
            )
            for item in candidates
        )

    def natural_language_context(self) -> CommandContext:
        """Return only current public state and generated action identifiers."""

        timeline = omni.timeline.get_timeline_interface()
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            session_state = "no_stage"
            stage_name = None
        else:
            session_state = "playing" if timeline.is_playing() else "paused"
            identifier = stage.GetRootLayer().identifier
            stage_name = Path(identifier).name if identifier else "untitled"
        capabilities = {
            CommandName.LOAD_DEFAULT_SCENE,
            CommandName.PLAY,
            CommandName.PAUSE,
            CommandName.STEP,
            CommandName.RESET,
            CommandName.SHOW_STATUS,
        }
        if stage is not None:
            capabilities.add(CommandName.INITIALIZE_RADIATION)
        if self.simulation is not None:
            capabilities.update(
                {
                    CommandName.SYNC_SCENE,
                    CommandName.MEASURE,
                    CommandName.RENDER_DOSE_MAP,
                }
            )
            if getattr(self, "_latest_records", None):
                capabilities.add(CommandName.EXPORT_MEASUREMENT)
        services = self._workflow_services
        measurement_controller = (
            None if services is None else getattr(services, "measurement_controller", None)
        )
        if (
            measurement_controller is not None
            and getattr(measurement_controller, "home_position_m", None) is not None
        ):
            capabilities.add(CommandName.RETURN_MEASUREMENT_ROBOT)
        available_actions = self._refresh_command_candidates()
        if available_actions:
            capabilities.add(CommandName.EXECUTE_CANDIDATE)
            # Reloading a bound stage would invalidate articulation controllers.
            capabilities.discard(CommandName.LOAD_DEFAULT_SCENE)
            capabilities.discard(CommandName.RESET)
        return CommandContext(
            session_state=session_state,
            stage_name=stage_name,
            available_actions=available_actions,
            capabilities=tuple(sorted(capabilities, key=str)),
        )

    @staticmethod
    def _complete_immediate(coroutine: Any) -> Any:
        try:
            coroutine.send(None)
        except StopIteration as completed:
            return completed.value
        coroutine.close()
        raise RuntimeError("physical workflow unexpectedly yielded")

    def _execute_candidate_command_sync(self, candidate_id: str) -> dict[str, object]:
        from radcounter.core.models.actions import ActionStatus, ActionType
        from radcounter.core.planning import DeterministicFeasibilityChecker

        services = self._workflow_services
        belief = self._workflow_belief
        if services is None or belief is None:
            raise RuntimeError("physical workflow is not initialized")
        self._refresh_command_candidates()
        candidate = self._command_candidates.get(candidate_id)
        if candidate is None:
            raise RuntimeError(f"action is no longer available: {candidate_id}")
        report = DeterministicFeasibilityChecker().evaluate(candidate, services.resources)
        if not report.feasible:
            raise RuntimeError(
                f"action became infeasible: {candidate_id}: {', '.join(report.reasons)}"
            )
        action = candidate.action
        if action.action_type == ActionType.MEASURE:
            self._navigation_status_prefix = "Moving to measurement station"
        self._robot_monitor.begin_action(action)
        try:
            prediction = services.preview(action, belief)
            result = self._complete_immediate(services.execute(action))
            if result.status not in {ActionStatus.COMPLETED, ActionStatus.PARTIAL}:
                raise RuntimeError(f"{candidate_id} failed: {result.public_details}")
            diagnosis = None
            if action.action_type != ActionType.MEASURE:
                verification = self._complete_immediate(services.verify(action))
                diagnosis = services.diagnose(prediction, verification)
                belief = services.update(belief, diagnosis)
                self._workflow_belief = belief
            self.set_workflow_view(services.workflow_view())
        except Exception:
            self._robot_monitor.finish_action(success=False)
            raise
        self._robot_monitor.finish_action(
            success=True,
            public_details=result.public_details,
        )
        return {
            "command": CommandName.EXECUTE_CANDIDATE.value,
            "action_id": candidate_id,
            "action_type": str(action.action_type),
            "status": str(result.status),
            "public_details": dict(result.public_details),
            "diagnosed": diagnosis is not None,
        }

    def _return_measurement_robot_command_sync(self) -> dict[str, object]:
        services = self._workflow_services
        controller = None if services is None else getattr(services, "measurement_controller", None)
        if controller is None:
            raise RuntimeError("measurement robot controller is not initialized")
        self._navigation_status_prefix = "Measurement robot returning to start"
        self._robot_monitor.begin_operation(
            robot_id="measurement",
            operation="Return to start",
            phase="returning_home",
        )
        try:
            report = controller.return_home()
        except Exception:
            self._robot_monitor.finish_action(success=False)
            raise
        if not report.success:
            self._robot_monitor.finish_action(success=False)
            raise RuntimeError(report.message)
        self._robot_monitor.finish_action(success=True)
        return {
            "command": CommandName.RETURN_MEASUREMENT_ROBOT.value,
            "status": "completed",
            "public_details": {
                "message": report.message,
                "motion_audit": {
                    "success": bool(report.success),
                    "steps": int(report.steps),
                    "displacement_m": float(report.displacement_m),
                    "initial_position_m": list(report.initial_position_m),
                    "final_position_m": list(report.final_position_m),
                },
            },
        }

    def process_pending_natural_language_actions(self) -> None:
        """Execute one physical action outside Kit's asyncio callback context."""

        if not self._physical_command_queue:
            return
        step, future = self._physical_command_queue.pop(0)
        if future.cancelled():
            return
        try:
            if step.command == CommandName.EXECUTE_CANDIDATE:
                assert step.candidate_id is not None
                result = self._execute_candidate_command_sync(step.candidate_id)
            elif step.command == CommandName.RETURN_MEASUREMENT_ROBOT:
                result = self._return_measurement_robot_command_sync()
            else:
                raise RuntimeError(f"unsupported queued physical command: {step.command}")
        except Exception as exc:
            future.set_exception(exc)
        else:
            future.set_result(result)

    async def execute_natural_language_step(self, step: CommandStep) -> dict[str, object]:
        """Execute one previously validated command on the Kit main thread."""

        from omni.kit.app import get_app

        command = step.command
        if command == CommandName.LOAD_DEFAULT_SCENE:
            await self._load_scene()
        elif command == CommandName.PLAY:
            omni.timeline.get_timeline_interface().play()
        elif command == CommandName.PAUSE:
            omni.timeline.get_timeline_interface().pause()
        elif command == CommandName.STEP:
            timeline = omni.timeline.get_timeline_interface()
            timeline.play()
            await get_app().next_update_async()
            timeline.pause()
        elif command == CommandName.RESET:
            await self._load_scene()
        elif command == CommandName.INITIALIZE_RADIATION:
            self._initialize_runtime()
        elif command == CommandName.SYNC_SCENE:
            self._synchronize()
        elif command == CommandName.MEASURE:
            if step.duration_s is not None:
                self._duration.set_value(float(step.duration_s))
            duration_s = float(self._duration.get_value_as_float())
            self._measure()
            records = tuple(getattr(self, "_latest_records", ()))
            public_measurements = [
                {
                    "detector_path": str(record.detector_path),
                    "counts": int(record.counts),
                    "measured_rate_cps": float(record.measured_rate_cps),
                    "duration_s": float(record.duration_s),
                }
                for record in records
            ]
            return {
                "command": command.value,
                "duration_s": duration_s,
                "detector_count": len(records),
                "measurements": public_measurements,
                "maximum_measured_rate_cps": max(
                    (row["measured_rate_cps"] for row in public_measurements),
                    default=0.0,
                ),
                "status": self._status.get_value_as_string(),
            }
        elif command == CommandName.RENDER_DOSE_MAP:
            self._render_dose_map()
        elif command == CommandName.EXPORT_MEASUREMENT:
            self._export_measurement()
        elif command in {
            CommandName.EXECUTE_CANDIDATE,
            CommandName.RETURN_MEASUREMENT_ROBOT,
        }:
            future = asyncio.get_running_loop().create_future()
            self._physical_command_queue.append((step, future))
            return await future
        elif command != CommandName.SHOW_STATUS:
            raise RuntimeError(f"unhandled natural-language command: {command}")
        return {
            "command": command.value,
            "status": self._status.get_value_as_string(),
        }

    def shutdown(self) -> None:
        for task in tuple(self._tasks):
            task.cancel()
        self._tasks.clear()
        for _, future in self._physical_command_queue:
            future.cancel()
        self._physical_command_queue.clear()
        self._natural_language.shutdown()
        self._robot_monitor.destroy()
        self._workflow_services = None
        self._workflow_belief = None
        self._command_candidates.clear()
        self.simulation = None
        self._window = None
