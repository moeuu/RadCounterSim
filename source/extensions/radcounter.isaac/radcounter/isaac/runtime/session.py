"""Stage and timeline lifecycle for the Isaac Sim host."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from enum import StrEnum
from pathlib import Path
from typing import Any

from radcounter.isaac.usd.registry import (
    UsdRadiationRegistry,
    UsdStageRevisionTracker,
)

from radcounter.core.environment import (
    EnvironmentDependencyError,
    EnvironmentImportConfig,
    EnvironmentImportPipeline,
    load_environment_descriptor,
    load_environment_manifest,
)
from radcounter.core.models.state import RevisionState


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
        self._request_path: Path | None = None
        self._state = SessionState.EMPTY
        self._registry: UsdRadiationRegistry | None = None
        self._revision_tracker: UsdStageRevisionTracker | None = None
        self._revision = RevisionState()

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def status_text(self) -> str:
        if self._request_path is None:
            return f"State: {self._state}"
        return f"State: {self._state} | {self._request_path.name}"

    @property
    def registry(self) -> UsdRadiationRegistry:
        if self._registry is None:
            raise RuntimeError("no radiation USD stage has been loaded")
        return self._registry

    @property
    def revision(self) -> RevisionState:
        return self._revision.copy()

    def _unbind_stage(self) -> None:
        if self._revision_tracker is not None:
            self._revision_tracker.stop()
        self._revision_tracker = None
        self._registry = None

    def _bind_stage(self, stage: Any, stage_path: Path) -> None:
        self._unbind_stage()
        self._revision = RevisionState()
        self._registry = UsdRadiationRegistry(stage, base_directory=stage_path.parent)
        self._registry.refresh()
        self._revision_tracker = UsdStageRevisionTracker(
            stage,
            self._registry,
            self._revision,
        )
        self._revision_tracker.start()

    async def load_stage(self, stage_path: str) -> None:
        path = Path(stage_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        self._timeline.stop()
        prepared_path = await self._prepare_environment(path)
        result = await self._context.open_stage_async(str(prepared_path))
        success = result[0] if isinstance(result, tuple) else result
        if success is False:
            raise RuntimeError(f"Isaac Sim failed to open stage: {prepared_path}")
        stage = self._context.get_stage()
        if stage is None:
            raise RuntimeError(f"Isaac Sim opened no USD stage for: {prepared_path}")
        self._bind_stage(stage, prepared_path)
        self._stage_path = prepared_path
        self._request_path = path
        self._state = SessionState.READY

    async def _prepare_environment(self, path: Path) -> Path:
        if path.suffix.lower() in {".usd", ".usda", ".usdc", ".usdz"}:
            return path
        if path.name == "manifest.json":
            result = load_environment_manifest(path)
        else:
            config = (
                load_environment_descriptor(path)
                if path.suffix.lower() in {".yaml", ".yml", ".json"}
                else EnvironmentImportConfig(environment_id=path.stem, uri=str(path))
            )
            try:
                result = await asyncio.to_thread(
                    EnvironmentImportPipeline().import_environment,
                    config,
                    base_directory=path.parent,
                )
            except EnvironmentDependencyError:
                result = await asyncio.to_thread(self._import_with_project_uv, path)
        from radcounter.isaac.usd.environment import EnvironmentUsdWriter

        return EnvironmentUsdWriter().write(
            result.scene,
            result.output_directory / "environment.usda",
        )

    @staticmethod
    def _import_with_project_uv(path: Path) -> Any:
        uv = shutil.which("uv")
        if uv is None:
            raise EnvironmentDependencyError(
                "the Isaac Python environment lacks an importer dependency and uv is unavailable"
            )
        repository_root = Path(__file__).resolve().parents[6]
        completed = subprocess.run(
            [
                uv,
                "run",
                "--project",
                str(repository_root),
                "--locked",
                "--group",
                "cad",
                "radcounter-import-environment",
                str(path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"environment conversion failed: {completed.stderr.strip()}")
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("environment converter returned no result")
        result = json.loads(lines[-1])
        return load_environment_manifest(result["manifest"])

    async def reset(self) -> None:
        if self._request_path is None:
            raise RuntimeError("no stage has been loaded")
        await self.load_stage(str(self._request_path))

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
        self._unbind_stage()
        self._timeline.stop()
        self._request_path = None
        self._state = SessionState.STOPPED
