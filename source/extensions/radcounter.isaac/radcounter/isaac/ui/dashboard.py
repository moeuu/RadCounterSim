"""Operational UI for measurement, dose-map inspection, and runtime synchronization."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import omni.timeline
import omni.ui as ui
import omni.usd

from ..runtime.simulation import IsaacRadiationSimulation, measurement_payload


class RadCounterDashboard:
    def __init__(self, ext_id: str) -> None:
        self.ext_id = ext_id
        self.root = Path(__file__).resolve().parents[6]
        self.stage_path = self.root / "assets/environments/radcounter_vertical_slice.usda"
        self.config_path = self.root / "configs/scenarios/vertical_slice.runtime.json"
        self.artifact_path = self.root / "artifacts/ui/latest_measurement.json"
        self.workflow_artifact_path = self.root / "artifacts/ui/latest_workflow.json"
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
        self._estimate = ui.SimpleStringModel("No estimator result supplied")
        self._residual = ui.SimpleStringModel("No verification residual supplied")
        self._plan = ui.SimpleStringModel("No countermeasure plan supplied")
        self._window = ui.Window("RadCounterSim Operations", width=460, height=880)
        self._window.frame.set_build_fn(self._build)

    def _build(self) -> None:
        with ui.VStack(spacing=10, height=0):
            ui.Label("RADCOUNTER / OPERATIONS", style={"font_size": 18, "color": 0xFFE3B341})
            ui.Label(
                "Measurement -> intervention -> verification",
                style={"font_size": 12, "color": 0xFF9FA6AD},
            )
            ui.Separator(height=4)
            with ui.HStack(height=34, spacing=8):
                ui.Button(
                    "Load vertical slice", clicked_fn=lambda: self._schedule(self._load_scene())
                )
                ui.Button("Play / Pause", clicked_fn=self._toggle_timeline)
            with ui.HStack(height=34, spacing=8):
                ui.Button("Initialize radiation", clicked_fn=self._initialize_runtime)
                ui.Button("Sync USD -> Embree", clicked_fn=self._synchronize)
            ui.Label("STATUS", style={"font_size": 11, "color": 0xFF6CB6FF})
            ui.Label("", model=self._status, word_wrap=True, height=52)
            ui.Label("", model=self._sources, word_wrap=True)
            ui.Label("", model=self._revisions, word_wrap=True)
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
            ui.Label("", model=self._measurement, word_wrap=True, height=150)
            ui.Separator(height=4)
            ui.Label("WORKFLOW OUTPUT", style={"font_size": 11, "color": 0xFF6CB6FF})
            ui.Label("ESTIMATE", style={"font_size": 10, "color": 0xFF9FA6AD})
            ui.Label("", model=self._estimate, word_wrap=True, height=54)
            ui.Label("RESIDUAL", style={"font_size": 10, "color": 0xFF9FA6AD})
            ui.Label("", model=self._residual, word_wrap=True, height=64)
            ui.Label("PLAN", style={"font_size": 10, "color": 0xFF9FA6AD})
            ui.Label("", model=self._plan, word_wrap=True, height=64)
            ui.Button(
                "Load latest workflow artifact",
                height=32,
                clicked_fn=self._load_workflow_artifact,
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

    def _schedule(self, coroutine) -> None:
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _load_scene(self) -> None:
        self._status.set_value("Loading USD stage...")
        context = omni.usd.get_context()
        result, error = await context.open_stage_async(str(self.stage_path))
        if not result:
            self._status.set_value(f"Stage load failed: {error}")
            return
        self.simulation = None
        self._status.set_value(f"Loaded {self.stage_path.name}")

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
        from pxr import Gf, Sdf, UsdGeom

        grid = np.asarray(
            [(x, y, 0.65) for y in np.linspace(-3.4, 3.4, 18) for x in np.linspace(-5.4, 5.4, 28)]
        )
        values = self.simulation.dose_proxy_map(grid)
        stage = omni.usd.get_context().get_stage()
        points = UsdGeom.Points.Define(stage, "/World/RadiationDoseProxy")
        points.CreatePointsAttr([Gf.Vec3f(*position) for position in grid])
        points.CreateDisplayColorPrimvar("vertex").Set(
            [Gf.Vec3f(*color) for color in self._color_map(values)]
        )
        points.CreateWidthsAttr([0.09 + 0.04 * math.log10(max(value, 1.0)) for value in values])
        points.GetPrim().CreateAttribute(
            "rad:visualization:dataClass", Sdf.ValueTypeNames.String, custom=True
        ).Set("count_rate_proxy_not_dose_calibrated")
        self._status.set_value(f"Rendered {len(grid)} count-rate proxy samples.")

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

    def shutdown(self) -> None:
        for task in tuple(self._tasks):
            task.cancel()
        self._tasks.clear()
        self.simulation = None
        self._window = None
