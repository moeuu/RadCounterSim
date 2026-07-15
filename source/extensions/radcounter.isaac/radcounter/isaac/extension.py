"""Isaac Sim extension entry point."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

try:
    import omni.ext  # type: ignore[import-not-found]
except ModuleNotFoundError:
    omni = None


if omni is not None:
    from radcounter.isaac.runtime.session import IsaacRuntimeSession
    from radcounter.isaac.ui.window import RadCounterWindow, WindowCallbacks

    class RadCounterExtension(omni.ext.IExt):
        """Own the RadCounterSim window and Isaac timeline session."""

        def on_startup(self, ext_id: str) -> None:
            self.ext_id = ext_id
            self._tasks: set[asyncio.Task[Any]] = set()
            self._session = IsaacRuntimeSession()
            self._window = RadCounterWindow(
                WindowCallbacks(
                    load_stage=self._load_stage,
                    reset=self._reset,
                    start=self._start,
                    pause=self._pause,
                    step=self._step,
                )
            )

        def _schedule(
            self,
            label: str,
            operation: Coroutine[Any, Any, object],
        ) -> None:
            self._window.set_status(f"{label}...")
            task = asyncio.ensure_future(operation)
            self._tasks.add(task)

            def completed(done: asyncio.Task[Any]) -> None:
                self._tasks.discard(done)
                if done.cancelled():
                    return
                error = done.exception()
                if error is None:
                    self._window.set_status(self._session.status_text)
                else:
                    self._window.set_status(f"Error: {error}")

            task.add_done_callback(completed)

        def _load_stage(self, stage_path: str) -> None:
            self._schedule("Loading stage", self._session.load_stage(stage_path))

        def _reset(self) -> None:
            self._schedule("Resetting", self._session.reset())

        def _start(self) -> None:
            self._session.start()
            self._window.set_status(self._session.status_text)

        def _pause(self) -> None:
            self._session.pause()
            self._window.set_status(self._session.status_text)

        def _step(self) -> None:
            self._schedule("Stepping", self._session.step_once())

        def on_shutdown(self) -> None:
            for task in self._tasks:
                task.cancel()
            self._tasks.clear()
            self._session.shutdown()
            self._window.destroy()
            self.ext_id = ""
