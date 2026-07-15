"""Compact host controls for a RadCounterSim episode."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


def _ui_module() -> Any:
    try:
        import omni.ui as ui  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise RuntimeError("RadCounterWindow requires omni.ui") from error
    return ui


@dataclass(frozen=True)
class WindowCallbacks:
    load_stage: Callable[[str], None]
    reset: Callable[[], None]
    start: Callable[[], None]
    pause: Callable[[], None]
    step: Callable[[], None]


class RadCounterWindow:
    """Render stage and timeline controls without importing UI outside Kit."""

    def __init__(self, callbacks: WindowCallbacks) -> None:
        ui = _ui_module()
        self._callbacks = callbacks
        self._window = ui.Window("RadCounterSim", width=430, height=230)
        self._stage_path_model = ui.SimpleStringModel("")
        self._status_model = ui.SimpleStringModel("State: empty")
        with self._window.frame, ui.VStack(spacing=8, height=0):
            ui.Label("Radiation countermeasure episode", height=24)
            ui.Label("USD stage", height=18)
            ui.StringField(self._stage_path_model, height=26)
            with ui.HStack(spacing=6, height=30):
                ui.Button(
                    "Load",
                    clicked_fn=lambda: callbacks.load_stage(self._stage_path_model.as_string),
                )
                ui.Button("Reset", clicked_fn=callbacks.reset)
            with ui.HStack(spacing=6, height=30):
                ui.Button("Start", clicked_fn=callbacks.start)
                ui.Button("Pause", clicked_fn=callbacks.pause)
                ui.Button("Step", clicked_fn=callbacks.step)
            ui.Separator(height=4)
            ui.Label(self._status_model, height=24)

    def set_status(self, text: str) -> None:
        self._status_model.set_value(text)

    def destroy(self) -> None:
        self._window.destroy()
        self._window = None
