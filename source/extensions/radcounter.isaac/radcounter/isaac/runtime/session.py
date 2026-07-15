"""Stage and timeline lifecycle for the Isaac Sim host."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any


class IsaacRuntimeUnavailable(RuntimeError):
    """Raised when a host-only operation is called outside Isaac Sim."""


class SessionState(StrEnum):
    EMPTY = "empty"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


def _host_modules() -> tuple[Any, Any, Any]:
    try:
        import omni.kit.app  # type: ignore[import-not-found]
        import omni.timeline  # type: ignore[import-not-found]
        import omni.usd  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise IsaacRuntimeUnavailable(
            "IsaacRuntimeSession requires the Isaac Sim Kit runtime"
        ) from error
    return omni.usd, omni.timeline, omni.kit.app


class IsaacRuntimeSession:
    """Provide deterministic stage and timeline controls to the extension UI."""

    def __init__(self) -> None:
        usd_module, timeline_module, app_module = _host_modules()
        self._context = usd_module.get_context()
        self._timeline = timeline_module.get_timeline_interface()
        self._app = app_module.get_app()
        self._stage_path: Path | None = None
        self._state = SessionState.EMPTY

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def status_text(self) -> str:
        if self._stage_path is None:
            return f"State: {self._state}"
        return f"State: {self._state} | {self._stage_path.name}"

    async def load_stage(self, stage_path: str) -> None:
        path = Path(stage_path).expanduser().resolve()
        if path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
            raise ValueError("stage path must be a USD, USDA, or USDC file")
        if not path.is_file():
            raise FileNotFoundError(path)
        self._timeline.stop()
        result = await self._context.open_stage_async(str(path))
        success = result[0] if isinstance(result, tuple) else result
        if success is False:
            raise RuntimeError(f"Isaac Sim failed to open stage: {path}")
        self._stage_path = path
        self._state = SessionState.READY

    async def reset(self) -> None:
        if self._stage_path is None:
            raise RuntimeError("no stage has been loaded")
        await self.load_stage(str(self._stage_path))

    def start(self) -> None:
        if self._stage_path is None:
            raise RuntimeError("no stage has been loaded")
        self._timeline.play()
        self._state = SessionState.RUNNING

    def pause(self) -> None:
        if self._stage_path is None:
            raise RuntimeError("no stage has been loaded")
        self._timeline.pause()
        self._state = SessionState.PAUSED

    async def step_once(self) -> None:
        if self._stage_path is None:
            raise RuntimeError("no stage has been loaded")
        self._timeline.play()
        await self._app.next_update_async()
        self._timeline.pause()
        self._state = SessionState.PAUSED

    def shutdown(self) -> None:
        self._timeline.stop()
        self._state = SessionState.STOPPED
