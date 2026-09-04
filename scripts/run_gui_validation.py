#!/usr/bin/env python3
"""Run every vertical-slice operation in one visible Isaac Sim session."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import tomllib
import traceback
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, is_dataclass, replace
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from radcounter.core.experiments import EvidenceClass, sha256_file

ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "source/extensions/radcounter.isaac"
NATIVE = ROOT / "build/native/python"
for path in (ROOT, EXTENSION, NATIVE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
_radcounter_package = importlib.import_module("radcounter")
_extension_namespace = str(EXTENSION / "radcounter")
if _extension_namespace not in _radcounter_package.__path__:
    _radcounter_package.__path__.append(_extension_namespace)


DEFAULT_COMPLEX_NATURAL_LANGUAGE_INSTRUCTION = (
    "Decontaminate the entire irregular wall-mounted Cs-137 surface source with a "
    "serpentine pass until the remaining fraction is no more than 60%, using at most "
    "3 passes. Then place the same primary lead shield, LeadShield, at "
    "25% of the source-to-protected-area line and move that same shield to 65%. "
    "Move to the Protected measurement station, measure for 2 seconds, return the "
    "measurement robot to its starting position, and finally show the current status."
)
DEFAULT_GUI_MAX_FPS = 60.0


class ComplexNaturalLanguageValidationError(RuntimeError):
    """Preserve a failed complex-run audit in the top-level JSON artifact."""

    def __init__(self, message: str, audit: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.audit = dict(audit)


class DecontaminationSmokeValidationError(RuntimeError):
    """Preserve contact and motion evidence when the CAD smoke gate fails."""

    def __init__(self, message: str, audit: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.audit = dict(audit)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--initial-command")
    parser.add_argument(
        "--confirm-initial-command",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--record-prompt-video",
        type=Path,
        help=(
            "record a visible scripted initial-command demonstration without "
            "synthesizing mouse or keyboard input"
        ),
    )
    parser.add_argument("--record-display", default=os.environ.get("DISPLAY", ":0.0"))
    parser.add_argument("--record-monitor-x", type=int, default=0)
    parser.add_argument("--record-monitor-y", type=int, default=0)
    parser.add_argument("--record-monitor-width", type=int, default=1920)
    parser.add_argument("--record-monitor-height", type=int, default=1080)
    parser.add_argument("--record-fps", type=int, default=30)
    parser.add_argument("--prompt-typing-delay-s", type=float, default=0.04)
    parser.add_argument("--prompt-pre-hold-s", type=float, default=2.0)
    parser.add_argument("--prompt-post-hold-s", type=float, default=2.0)
    parser.add_argument("--prompt-confirm-hold-s", type=float, default=2.5)
    parser.add_argument("--record-final-hold-s", type=float, default=4.0)
    parser.add_argument(
        "--complex-natural-language-validation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "run the visible, locally interpreted multi-pass decontamination and "
            "same-shield repositioning gate"
        ),
    )
    parser.add_argument(
        "--natural-language-timeout-s",
        type=float,
        default=1800.0,
        help="wall-clock timeout for the bounded complex natural-language workflow",
    )
    parser.add_argument("--keep-open", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--phase-hold-s", type=float, default=0.35)
    parser.add_argument("--frame-delay-s", type=float, default=0.0)
    parser.add_argument(
        "--max-fps",
        type=float,
        default=DEFAULT_GUI_MAX_FPS,
        help="cap the open GUI refresh rate; use 0 to disable the limit",
    )
    parser.add_argument("--decon-duration-s", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--system-catalog", type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--environment")
    parser.add_argument("--robot-set")
    parser.add_argument("--detector-set")
    parser.add_argument("--selection-file", type=Path)
    parser.add_argument(
        "--robot-monitor-smoke-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="exercise follow, onboard, work, overview, and manual-cancel camera modes",
    )
    parser.add_argument(
        "--decontamination-smoke-test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "compose the articulated decontamination task on the selected "
            "environment and execute one contact-verified raster"
        ),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "artifacts/gui-validation/latest.json",
    )
    arguments = parser.parse_args(argv)
    if arguments.interactive and arguments.complex_natural_language_validation:
        parser.error("--interactive and --complex-natural-language-validation are separate modes")
    if arguments.complex_natural_language_validation and arguments.headless:
        parser.error(
            "--complex-natural-language-validation requires a visible GUI; remove --headless"
        )
    if arguments.natural_language_timeout_s <= 0.0:
        parser.error("--natural-language-timeout-s must be positive")
    if arguments.seed < 0:
        parser.error("--seed must be nonnegative")
    if not math.isfinite(arguments.max_fps) or arguments.max_fps < 0.0:
        parser.error("--max-fps must be a finite non-negative number")
    recording_delays = (
        arguments.prompt_typing_delay_s,
        arguments.prompt_pre_hold_s,
        arguments.prompt_post_hold_s,
        arguments.prompt_confirm_hold_s,
        arguments.record_final_hold_s,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in recording_delays):
        parser.error("prompt recording delays must be finite and non-negative")
    if arguments.record_fps <= 0:
        parser.error("--record-fps must be positive")
    if arguments.record_monitor_width <= 0 or arguments.record_monitor_height <= 0:
        parser.error("recording dimensions must be positive")
    if arguments.record_prompt_video is not None:
        if not arguments.interactive:
            parser.error("--record-prompt-video requires --interactive")
        if not arguments.initial_command:
            parser.error("--record-prompt-video requires --initial-command")
        if not arguments.confirm_initial_command:
            parser.error("--record-prompt-video requires --confirm-initial-command")
        if arguments.headless:
            parser.error("--record-prompt-video requires a visible GUI")
    return arguments


def _selected_system(args: argparse.Namespace):
    from radcounter.core.system_profiles import (
        load_active_selection,
        resolve_system_selection,
    )

    explicit = any(
        (
            args.system_catalog,
            args.profile,
            args.environment,
            args.robot_set,
            args.detector_set,
        )
    )
    if not explicit:
        return load_active_selection(args.selection_file)
    return resolve_system_selection(
        catalog_path=args.system_catalog,
        profile_id=args.profile,
        environment_id=args.environment,
        robot_set_id=args.robot_set,
        detector_set_id=args.detector_set,
    )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "as_dict"):
        return _jsonable(value.as_dict())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    return str(value)


def _required_package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError as error:
        raise RuntimeError(
            f"physical validation cannot identify required distribution {distribution!r}"
        ) from error


def _project_version() -> str:
    try:
        return version("radinteract")
    except PackageNotFoundError as error:
        payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        project = payload.get("project")
        value = None if not isinstance(project, Mapping) else project.get("version")
        if not isinstance(value, str) or not value:
            raise RuntimeError(
                "physical validation cannot identify the RadInterAct version"
            ) from error
        return value


def _physical_execution_runtime() -> dict[str, Any]:
    """Capture the runtime that produced physical simulation evidence."""

    import _radcounter_embree
    import carb.settings

    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    gpus: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise RuntimeError("nvidia-smi returned an invalid physical-runtime record")
        try:
            memory_mib = int(fields[2])
        except ValueError as error:
            raise RuntimeError("nvidia-smi returned invalid GPU memory") from error
        if not fields[0] or not fields[1] or memory_mib <= 0:
            raise RuntimeError("nvidia-smi returned incomplete GPU evidence")
        gpus.append(
            {
                "name": fields[0],
                "driver_version": fields[1],
                "memory_mib": memory_mib,
            }
        )
    if not gpus:
        raise RuntimeError("physical validation requires at least one reported NVIDIA GPU")
    renderer_mode = carb.settings.get_settings().get_as_string("/rtx/rendermode")
    if not renderer_mode:
        raise RuntimeError("physical validation could not identify the Isaac renderer mode")
    embree_version = str(_radcounter_embree.embree_version())
    if not embree_version:
        raise RuntimeError("physical validation could not identify the Embree version")
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": _required_package_version("numpy"),
        "radinteract_version": _project_version(),
        "isaac_sim_version": _required_package_version("isaacsim"),
        "embree_version": embree_version,
        "renderer_mode": renderer_mode,
        "gpus": gpus,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _complete(coroutine: Any) -> Any:
    """Complete a workflow coroutine that intentionally never suspends."""

    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    coroutine.close()
    raise RuntimeError("workflow coroutine unexpectedly yielded to the Kit event loop")


def _complete_with_updates(
    app: Any,
    coroutine: Any,
    *,
    before_update: Any | None = None,
    timeout_s: float = 180.0,
) -> Any:
    """Drive Kit's event loop until an asynchronous application command completes."""

    task = asyncio.ensure_future(coroutine)
    deadline = time.monotonic() + timeout_s
    while not task.done() and time.monotonic() < deadline:
        if before_update is not None:
            before_update()
        app.update()
        time.sleep(0.01)
    if not task.done():
        task.cancel()
        raise TimeoutError("natural-language application command timed out")
    return task.result()


class _GuiStepper:
    def __init__(self, app: Any, frame_delay_s: float, world: Any | None = None) -> None:
        self.app = app
        self.world = world
        self.frame_delay_s = max(0.0, frame_delay_s)

    def step(self, *, render: bool = False) -> None:
        if self.world is None:
            self.app.update()
        else:
            self.world.step(render=render)
        if self.frame_delay_s:
            time.sleep(self.frame_delay_s)


class _GuiFrameRateLimiter:
    """Bound the persistent GUI loop without slowing validation physics."""

    def __init__(
        self,
        max_fps: float,
        *,
        clock: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._frame_period_s = 0.0 if max_fps == 0.0 else 1.0 / max_fps
        self._clock = clock
        self._sleeper = sleeper
        self._previous_frame_s = clock()

    def wait(self) -> None:
        if self._frame_period_s == 0.0:
            return
        remaining_s = self._frame_period_s - (self._clock() - self._previous_frame_s)
        if remaining_s > 0.0:
            self._sleeper(remaining_s)
        self._previous_frame_s = self._clock()


class _ValidationPanel:
    def __init__(self) -> None:
        import omni.ui as ui

        self._ui = ui
        self.phase = ui.SimpleStringModel("Preparing scene")
        self.progress = ui.SimpleStringModel("0 / 9 operations")
        self.result = ui.SimpleStringModel("No operation has completed yet")
        self.audit = ui.SimpleStringModel("Validation is running")
        self.window = ui.Window(
            "RadInterAct Full Validation",
            width=520,
            height=690,
            dockPreference=ui.DockPreference.RIGHT,
        )
        self.window.frame.set_build_fn(self._build)
        # Keep both RadInterAct panels together from startup. Operations is
        # the normal operator-facing tab, so leave it selected after docking.
        target_active = getattr(
            ui.DockPolicy,
            "TARGET_WINDOW_IS_ACTIVE",
            ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE,
        )
        self.window.deferred_dock_in("RadInterAct Operations", target_active)

    def _build(self) -> None:
        ui = self._ui
        with ui.VStack(spacing=10, height=0):
            ui.Label("RADCOUNTER / FULL GUI VALIDATION", style={"font_size": 18})
            ui.Label(
                "Measurement -> decon -> shield -> relocation -> disposal -> verification",
                word_wrap=True,
            )
            ui.Separator(height=4)
            ui.Label("CURRENT PHASE", style={"font_size": 11, "color": 0xFF6CB6FF})
            ui.Label("", model=self.phase, word_wrap=True, height=48)
            ui.Label("", model=self.progress, height=24)
            ui.Separator(height=4)
            ui.Label("LATEST RESULT", style={"font_size": 11, "color": 0xFFE3B341})
            ui.Label("", model=self.result, word_wrap=True, height=350)
            ui.Separator(height=4)
            ui.Label("AUDIT", style={"font_size": 11, "color": 0xFF6CB6FF})
            ui.Label("", model=self.audit, word_wrap=True, height=90)
            ui.Label(
                "Synthetic validation data only. Not calibrated for safety decisions.",
                word_wrap=True,
                style={"font_size": 11, "color": 0xFFDD7A6B},
            )

    def update(self, phase: str, completed: int, total: int, result: object = "") -> None:
        self.phase.set_value(phase)
        self.progress.set_value(f"{completed} / {total} operations")
        encoded = json.dumps(_jsonable(result), indent=2, sort_keys=True)
        self.result.set_value(encoded[-4000:])
        print(json.dumps({"phase": phase, "completed": completed, "total": total}), flush=True)

    def finish(self, success: bool, artifact: Path) -> None:
        if success:
            self.audit.set_value(
                "PASS: every requested GUI operation completed and final invariants passed.\n"
                f"Evidence: {artifact}"
            )
        else:
            self.audit.set_value(f"FAIL: inspect the latest result and {artifact}")

    def hide(self) -> None:
        self.window.visible = False


def _hold(app: Any, duration_s: float) -> None:
    deadline = time.monotonic() + max(0.0, duration_s)
    while time.monotonic() < deadline and app.is_running():
        app.update()


def _type_visible_instruction(
    app: Any,
    dashboard: Any,
    instruction: str,
    *,
    character_delay_s: float,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    """Render typed text through the UI model without owning operator input devices."""

    dashboard.set_natural_language_input("")
    app.update()
    for end in range(1, len(instruction) + 1):
        dashboard.set_natural_language_input(instruction[:end])
        app.update()
        if character_delay_s:
            sleeper(character_delay_s)


def _start_display_recording(args: argparse.Namespace) -> subprocess.Popen[bytes]:
    """Capture the requested X11 monitor region without changing focus or input state."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for --record-prompt-video")
    output = args.record_prompt_video.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "warning",
        "-f",
        "x11grab",
        "-framerate",
        str(args.record_fps),
        "-video_size",
        f"{args.record_monitor_width}x{args.record_monitor_height}",
        "-i",
        (f"{args.record_display}+{args.record_monitor_x},{args.record_monitor_y}"),
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p4",
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        "18",
        "-b:v",
        "10M",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _stop_display_recording(process: subprocess.Popen[bytes]) -> None:
    """Finalize an ffmpeg capture and surface encoder failures."""

    stderr = b""
    try:
        stderr = process.communicate(input=b"q\n", timeout=20.0)[1]
    except subprocess.TimeoutExpired:
        process.terminate()
        stderr = process.communicate(timeout=10.0)[1]
    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"ffmpeg display recording failed: {message}")


def _world_position(stage: Any, prim_path: str) -> np.ndarray:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"USD prim is unavailable: {prim_path}")
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    return np.asarray(transform.Transform(Gf.Vec3d()), dtype=np.float64)


def _path_blockers(
    probe: Any,
    start_m: np.ndarray,
    target_m: np.ndarray,
    *,
    excluded_paths: tuple[str, ...] = (),
    moving_robot_path: str | None = None,
) -> list[str]:
    start = np.asarray(start_m, dtype=np.float64)
    target = np.asarray(target_m, dtype=np.float64)
    distance = float(np.linalg.norm(target[:2] - start[:2]))
    samples = np.linspace(start, target, max(2, int(np.ceil(distance / 0.12)) + 1))
    samples[:, 2] = np.maximum(samples[:, 2], 0.32)
    blockers: list[str] = []
    for path, expanded_lower, expanded_upper in probe._navigation_obstacles(
        excluded_paths=excluded_paths,
        moving_robot_path=moving_robot_path,
    ):
        inside = np.all((samples >= expanded_lower) & (samples <= expanded_upper), axis=1)
        if bool(np.any(inside)):
            blockers.append(path)
    return blockers


def _configure_camera(stage: Any) -> None:
    try:
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Gf, UsdGeom

        path = "/World/ValidationCamera"
        camera = UsdGeom.Camera.Define(stage, path)
        camera.CreateFocalLengthAttr(22.0)
        view = Gf.Matrix4d().SetLookAt(
            # Frame the original cell, connecting corridor, and separate
            # wall-decontamination room in one high oblique facility view.
            Gf.Vec3d(18.5, -19.5, 21.0),
            Gf.Vec3d(4.5, 0.0, 0.5),
            Gf.Vec3d(0.0, 0.0, 1.0),
        )
        UsdGeom.Xformable(camera).MakeMatrixXform().Set(view.GetInverse())
        viewport = get_active_viewport()
        if viewport is not None:
            viewport.set_active_camera(path)
    except Exception as error:
        print(f"camera configuration warning: {type(error).__name__}: {error}", flush=True)


def _configure_decon_room_camera(
    stage: Any,
    *,
    eye_m: tuple[float, float, float] = (10.25, -2.10, 2.75),
    target_m: tuple[float, float, float] = (14.30, 0.80, 1.12),
) -> None:
    """Leave interactive runs focused on the irregular source and real tool."""

    try:
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Gf, UsdGeom

        path = "/World/DeconRoomInspectionCamera"
        camera = UsdGeom.Camera.Define(stage, path)
        camera.CreateFocalLengthAttr(25.0)
        view = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*eye_m),
            Gf.Vec3d(*target_m),
            Gf.Vec3d(0.0, 0.0, 1.0),
        )
        UsdGeom.Xformable(camera).MakeMatrixXform().Set(view.GetInverse())
        viewport = get_active_viewport()
        if viewport is not None:
            viewport.set_active_camera(path)
    except Exception as error:
        print(f"inspection camera warning: {type(error).__name__}: {error}", flush=True)


def _configure_complex_facility_camera(stage: Any) -> None:
    """Frame all four connected rooms and the shield-staging branch."""

    try:
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Gf, UsdGeom

        path = "/World/ComplexFacilityValidationCamera"
        camera = UsdGeom.Camera.Define(stage, path)
        camera.CreateFocalLengthAttr(21.0)
        view = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(28.0, -25.0, 28.0),
            Gf.Vec3d(8.0, 3.0, 0.55),
            Gf.Vec3d(0.0, 0.0, 1.0),
        )
        UsdGeom.Xformable(camera).MakeMatrixXform().Set(view.GetInverse())
        viewport = get_active_viewport()
        if viewport is not None:
            viewport.set_active_camera(path)
    except Exception as error:
        print(f"complex facility camera warning: {type(error).__name__}: {error}", flush=True)


def _activity_total(path: Path) -> float:
    with np.load(path, allow_pickle=False) as payload:
        return float(np.sum(payload["activity_bq"]))


def _measurement_rows(items: Iterable[Any]) -> list[dict[str, Any]]:
    return [_jsonable(item) for item in items]


def _spectrum_components(value: Any) -> dict[str, object]:
    return {
        "total_cps_per_bin": np.asarray(value.total_cps_per_bin).tolist(),
        "primary_cps_per_bin": np.asarray(value.primary_cps_per_bin).tolist(),
        "corrected_source_cps_per_bin": np.asarray(value.corrected_source_cps_per_bin).tolist(),
        "background_cps_per_bin": np.asarray(value.background_cps_per_bin).tolist(),
        "live_fraction": float(value.live_fraction),
        "buildup_model_name": str(value.buildup_model_name),
    }


def _attribute_value(prim: Any, name: str, default: Any = None) -> Any:
    """Read one authored USD attribute without leaking invalid attribute handles."""

    if prim is None or not prim or not prim.IsValid():
        return default
    attribute = prim.GetAttribute(name)
    if not attribute or not attribute.IsValid() or not attribute.HasAuthoredValueOpinion():
        return default
    value = attribute.Get()
    return default if value is None else value


def _usd_string_array(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(item) for item in value]
    except TypeError:
        return [str(value)]


def _environment_audit(
    stage: Any,
    *,
    surface_path: str,
    activity_path: Path,
    facility_root_path: str = "/World/RemoteDeconFacility",
) -> dict[str, Any]:
    """Audit the visible facility and the exact activity-bearing source geometry."""

    from pxr import UsdGeom

    facility = stage.GetPrimAtPath(facility_root_path)
    surface = stage.GetPrimAtPath(surface_path)
    room_prefix = f"{facility_root_path}/Rooms/"
    corridor_prefix = f"{facility_root_path}/Corridors/"
    rooms: list[dict[str, Any]] = []
    corridors: list[dict[str, Any]] = []
    equipment: list[dict[str, Any]] = []
    shields: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        room_id = _attribute_value(prim, "rad:facility:roomId")
        corridor_id = _attribute_value(prim, "rad:facility:corridorId")
        role = str(_attribute_value(prim, "rad:role", ""))
        if path.startswith(room_prefix) and room_id:
            rooms.append({"path": path, "room_id": str(room_id)})
        if path.startswith(corridor_prefix) and corridor_id:
            corridors.append(
                {
                    "path": path,
                    "corridor_id": str(corridor_id),
                    "from_room_id": str(_attribute_value(prim, "rad:facility:fromRoomId", "")),
                    "to_room_id": str(_attribute_value(prim, "rad:facility:toRoomId", "")),
                }
            )
        if role == "facility_equipment":
            equipment.append(
                {
                    "path": path,
                    "equipment_id": str(_attribute_value(prim, "rad:facility:equipmentId", "")),
                    "equipment_type": str(_attribute_value(prim, "rad:facility:equipmentType", "")),
                    "room_id": str(room_id or ""),
                    "fixed": bool(_attribute_value(prim, "rad:facility:fixed", False)),
                }
            )
        if role == "shield":
            shields.append(
                {
                    "path": path,
                    "inventory_id": str(_attribute_value(prim, "rad:shield:inventoryId", "")),
                    "movable": bool(_attribute_value(prim, "rad:shield:movable", False)),
                    "staged": bool(_attribute_value(prim, "rad:shield:staged", False)),
                    "deployed": bool(_attribute_value(prim, "rad:shield:deployed", False)),
                    "placement_fraction": _attribute_value(prim, "rad:shield:placementFraction"),
                    "room_id": str(room_id or ""),
                }
            )

    source_map_uri = str(_attribute_value(surface, "rad:source:activityMapUri", ""))
    decon_map_uri = str(_attribute_value(surface, "rad:decon:activityMapUri", ""))
    face_count = 0
    if surface and surface.IsValid() and surface.IsA(UsdGeom.Mesh):
        face_count = len(UsdGeom.Mesh(surface).GetFaceVertexCountsAttr().Get() or [])
    map_path = Path(source_map_uri).expanduser() if source_map_uri else activity_path
    map_triangle_count = 0
    map_activity_count = 0
    map_activity_bq = 0.0
    map_load_error: str | None = None
    try:
        with np.load(map_path, allow_pickle=False) as payload:
            triangle_indices = np.asarray(payload["triangle_indices"])
            activity = np.asarray(payload["activity_bq"], dtype=np.float64)
            map_triangle_count = int(triangle_indices.size)
            map_activity_count = int(activity.size)
            map_activity_bq = float(np.sum(activity))
    except Exception as error:
        map_load_error = f"{type(error).__name__}: {error}"

    candidate_cells = int(_attribute_value(surface, "rad:source:candidateCellCount", 0))
    active_cells = int(_attribute_value(surface, "rad:source:activeCellCount", 0))
    active_faces = int(_attribute_value(surface, "rad:source:activeFaceCount", 0))
    declared = {
        "layout_id": str(_attribute_value(facility, "rad:facility:layoutId", "")),
        "deterministic": bool(_attribute_value(facility, "rad:facility:deterministic", False)),
        "room_count": int(_attribute_value(facility, "rad:facility:roomCount", 0)),
        "corridor_count": int(_attribute_value(facility, "rad:facility:corridorCount", 0)),
        "equipment_count": int(_attribute_value(facility, "rad:facility:equipmentCount", 0)),
        "obstacle_count": int(_attribute_value(facility, "rad:facility:obstacleCount", 0)),
        "reserved_route_ids": _usd_string_array(
            _attribute_value(facility, "rad:facility:reservedRouteIds", ())
        ),
        "decon_surface_geometry": str(
            _attribute_value(facility, "rad:facility:deconSurfaceGeometry", "")
        ),
    }
    source = {
        "path": surface_path,
        "role": str(_attribute_value(surface, "rad:role", "")),
        "source_type": str(_attribute_value(surface, "rad:source:type", "")),
        "source_enabled": bool(_attribute_value(surface, "rad:source:enabled", False)),
        "decon_enabled": bool(_attribute_value(surface, "rad:decon:enabled", False)),
        "irregular_mask": bool(_attribute_value(surface, "rad:source:irregularMask", False)),
        "candidate_cell_count": candidate_cells,
        "active_cell_count": active_cells,
        "active_face_count": active_faces,
        "mesh_face_count": face_count,
        "deposition_model": str(_attribute_value(surface, "rad:source:depositionModel", "")),
        "geometry": str(_attribute_value(surface, "rad:source:geometry", "")),
        "surface_orientation": str(_attribute_value(surface, "rad:decon:surfaceOrientation", "")),
        "raster_rows": int(_attribute_value(surface, "rad:decon:rasterRows", 0)),
        "collision_enabled": bool(_attribute_value(surface, "physics:collisionEnabled", False)),
        "source_activity_map_uri": source_map_uri,
        "decon_activity_map_uri": decon_map_uri,
        "requested_activity_map_uri": str(activity_path.resolve()),
        "map_triangle_count": map_triangle_count,
        "map_activity_count": map_activity_count,
        "map_activity_bq": map_activity_bq,
        "map_load_error": map_load_error,
    }
    discovered = {
        "room_count": len(rooms),
        "corridor_count": len(corridors),
        "equipment_count": len(equipment),
        "shield_count": len(shields),
    }
    invariants = {
        "facility_root_exists": bool(facility and facility.IsValid()),
        "deterministic_facility": declared["deterministic"],
        "at_least_four_rooms": (declared["room_count"] >= 4 and discovered["room_count"] >= 4),
        "room_declaration_matches_stage": (declared["room_count"] == discovered["room_count"]),
        "at_least_three_corridors": (
            declared["corridor_count"] >= 3 and discovered["corridor_count"] >= 3
        ),
        "corridor_declaration_matches_stage": (
            declared["corridor_count"] == discovered["corridor_count"]
        ),
        "at_least_ten_equipment_obstacles": (
            declared["equipment_count"] >= 10
            and declared["obstacle_count"] >= 10
            and discovered["equipment_count"] >= 10
        ),
        "equipment_declaration_matches_stage": (
            declared["equipment_count"] == discovered["equipment_count"]
        ),
        "multiple_reserved_routes": len(declared["reserved_route_ids"]) >= 2,
        "at_least_two_physical_shields": discovered["shield_count"] >= 2,
        "primary_and_staging_shields_exist": {
            "/World/LeadShield",
            "/World/StagingLeadShield",
        }.issubset({row["path"] for row in shields}),
        "surface_source_and_decon_enabled": (source["source_enabled"] and source["decon_enabled"]),
        "surface_is_irregular_dense_geometry": (
            source["irregular_mask"]
            and candidate_cells >= 48 * 28
            and active_cells > 0
            and source["deposition_model"]
            == "gaussian_lobes_correlated_roughness_holes_satellite_droplets"
            and source["geometry"] == "irregular_masked_triangle_activity_map"
        ),
        "surface_is_six_lane_vertical_raster": (
            source["surface_orientation"] == "vertical_x" and source["raster_rows"] == 6
        ),
        "visible_collision_activity_geometry_is_identical": (
            declared["decon_surface_geometry"] == "same_visible_irregular_collision_activity_mesh"
            and source["collision_enabled"]
            and active_faces == 2 * active_cells
            and active_faces == face_count
            and active_faces == map_triangle_count
            and map_triangle_count == map_activity_count
            and map_load_error is None
        ),
        "source_and_decon_use_same_activity_map": (
            bool(source_map_uri)
            and source_map_uri == decon_map_uri
            and Path(source_map_uri).expanduser().resolve() == activity_path.resolve()
        ),
    }
    return {
        "facility_root": facility_root_path,
        "declared": declared,
        "discovered": discovered,
        "rooms": rooms,
        "corridors": corridors,
        "equipment": equipment,
        "shields": shields,
        "surface_source": source,
        "invariants": invariants,
        "failed_invariants": [name for name, passed in invariants.items() if not passed],
    }


def _enum_tail(value: Any) -> str:
    return str(value).strip().lower().rsplit(".", 1)[-1]


def _completed_action_result(row: Mapping[str, Any]) -> bool:
    return _enum_tail(row.get("status", "")) in {"completed", "partial"}


def _complex_process_audit(
    plan_payload: Mapping[str, Any],
    raw_results: Iterable[Mapping[str, Any]],
    environment_before: Mapping[str, Any],
    environment_after: Mapping[str, Any],
    physical_state: Mapping[str, Any],
    *,
    expected_surface_path: str,
    expected_shield_path: str = "/World/LeadShield",
    expected_protected_path: str = "/World/DetectorStations/Protected",
) -> dict[str, Any]:
    """Verify ordered process semantics using only artifact-safe public results."""

    plan = _jsonable(plan_payload)
    results = [dict(_jsonable(row)) for row in raw_results]
    plan_steps = [dict(step) for step in plan.get("steps", []) if isinstance(step, Mapping)]

    def action_rows(action_type: str) -> list[tuple[int, dict[str, Any]]]:
        return [
            (index, row)
            for index, row in enumerate(results)
            if _enum_tail(row.get("action_type", "")) == action_type
        ]

    decon_rows = action_rows("decontaminate")
    place_rows = action_rows("place_shield")
    move_rows = action_rows("move_shield")
    protected_moves = [
        (index, row)
        for index, row in action_rows("measure")
        if str(dict(row.get("public_details", {})).get("detector_path", ""))
        == expected_protected_path
    ]
    measurement_rows = [
        (index, row)
        for index, row in enumerate(results)
        if _enum_tail(row.get("command", "")) == "measure" and "action_type" not in row
    ]
    return_rows = [
        (index, row)
        for index, row in enumerate(results)
        if _enum_tail(row.get("command", "")) == "return_measurement_robot"
    ]
    status_rows = [
        (index, row)
        for index, row in enumerate(results)
        if _enum_tail(row.get("command", "")) == "show_status"
    ]

    decon_plan_steps = [
        step
        for step in plan_steps
        if _enum_tail(step.get("command", "")) == "execute_candidate"
        and "decon" in str(step.get("candidate_id", "")).lower()
    ]
    decon_plan = decon_plan_steps[0] if decon_plan_steps else {}
    decon_max_attempts = int(decon_plan.get("max_attempts", 0) or 0)
    decon_until = decon_plan.get("until")
    if not isinstance(decon_until, Mapping):
        decon_until = {}
    completion_rows = [
        dict(row.get("completion_condition", {}))
        for _, row in decon_rows
        if isinstance(row.get("completion_condition"), Mapping)
    ]
    final_completion = completion_rows[-1] if completion_rows else {}
    final_decon_attempt = int(decon_rows[-1][1].get("attempt", 0)) if decon_rows else 0

    decon_public_details = [dict(row.get("public_details", {})) for _, row in decon_rows]
    decon_motion = [dict(details.get("motion_audit", {})) for details in decon_public_details]
    activity_before_values = [float(row.get("activity_before_bq", 0.0)) for row in decon_motion]
    activity_after_values = [float(row.get("activity_after_bq", 0.0)) for row in decon_motion]

    shield_events: list[tuple[int, dict[str, Any], str]] = []
    for index, row in (*place_rows, *move_rows):
        shield_events.append((index, row, _enum_tail(row.get("action_type", ""))))
    shield_events.sort(key=lambda item: item[0])

    mitigation_rows = [*decon_rows, *place_rows, *move_rows]
    collateral_rows: list[dict[str, Any]] = []
    missing_collateral_audit: list[str] = []
    for _, row in mitigation_rows:
        details = row.get("public_details", {})
        details = dict(details) if isinstance(details, Mapping) else {}
        rows = details.get("collateral_motion_audit")
        if not isinstance(rows, list):
            missing_collateral_audit.append(str(row.get("action_id", "unknown")))
            continue
        for collateral in rows:
            if isinstance(collateral, Mapping):
                collateral_rows.append(dict(collateral))
    maximum_collateral_displacement_m = max(
        (float(row.get("displacement_m", float("inf"))) for row in collateral_rows),
        default=0.0,
    )

    place_details = dict(place_rows[0][1].get("public_details", {})) if place_rows else {}
    move_details = dict(move_rows[0][1].get("public_details", {})) if move_rows else {}
    place_motion = dict(place_details.get("motion_audit", {}))
    move_motion = dict(move_details.get("motion_audit", {}))
    protected_motion = (
        dict(protected_moves[0][1].get("public_details", {})).get("motion_audit", {})
        if protected_moves
        else {}
    )
    protected_motion = dict(protected_motion) if isinstance(protected_motion, Mapping) else {}
    measurement = measurement_rows[0][1] if measurement_rows else {}
    return_result = return_rows[0][1] if return_rows else {}
    return_details = return_result.get("public_details", {})
    return_details = dict(return_details) if isinstance(return_details, Mapping) else {}
    return_motion = return_details.get("motion_audit", {})
    return_motion = dict(return_motion) if isinstance(return_motion, Mapping) else {}

    position_before = np.asarray(
        physical_state.get("measurement_robot_initial_position_m", ()),
        dtype=np.float64,
    )
    position_after = np.asarray(
        physical_state.get("measurement_robot_final_position_m", ()),
        dtype=np.float64,
    )
    returned_home_distance_m = (
        float(np.linalg.norm(position_after - position_before))
        if position_before.shape == (3,) and position_after.shape == (3,)
        else float("inf")
    )
    shield_before = np.asarray(
        physical_state.get("primary_shield_initial_position_m", ()), dtype=np.float64
    )
    shield_after = np.asarray(
        physical_state.get("primary_shield_final_position_m", ()), dtype=np.float64
    )
    shield_displacement_m = (
        float(np.linalg.norm(shield_after - shield_before))
        if shield_before.shape == (3,) and shield_after.shape == (3,)
        else 0.0
    )
    final_primary_shield = next(
        (
            row
            for row in environment_after.get("shields", [])
            if isinstance(row, Mapping) and row.get("path") == expected_shield_path
        ),
        {},
    )
    expected_plan_commands = [
        "execute_candidate",
        "execute_candidate",
        "execute_candidate",
        "execute_candidate",
        "measure",
        "return_measurement_robot",
        "show_status",
    ]
    result_rows_by_workflow_step = {
        step_number: [
            row for row in results if int(row.get("workflow_step", 0) or 0) == step_number
        ]
        for step_number in range(1, len(plan_steps) + 1)
    }
    physical_results_match_plan = all(
        result_rows_by_workflow_step.get(step_number)
        and all(
            row.get("action_id") == plan_steps[step_number - 1].get("candidate_id")
            for row in result_rows_by_workflow_step[step_number]
        )
        for step_number in range(1, 5)
    )

    ordered_indices = (
        decon_rows[-1][0] if decon_rows else -1,
        place_rows[0][0] if place_rows else -1,
        move_rows[0][0] if move_rows else -1,
        protected_moves[0][0] if protected_moves else -1,
        measurement_rows[0][0] if measurement_rows else -1,
        return_rows[0][0] if return_rows else -1,
        status_rows[0][0] if status_rows else -1,
    )
    invariants = {
        "environment_before_passed": not environment_before.get("failed_invariants"),
        "environment_after_passed": not environment_after.get("failed_invariants"),
        "plan_has_exact_seven_step_complex_process": (
            len(plan_steps) == 7
            and [_enum_tail(step.get("command", "")) for step in plan_steps]
            == expected_plan_commands
        ),
        "physical_results_match_confirmed_plan_candidates": physical_results_match_plan,
        "plan_has_bounded_multi_pass_decontamination": (
            bool(decon_plan_steps)
            and decon_max_attempts == 3
            and bool(decon_until)
            and _enum_tail(decon_until.get("criterion", ""))
            == "decontamination_remaining_fraction_at_most"
            and abs(float(decon_until.get("threshold", -1.0)) - 0.60) <= 1.0e-6
        ),
        "decontamination_attempt_metadata_complete": (
            bool(decon_rows)
            and len(completion_rows) == len(decon_rows)
            and [int(row.get("attempt", 0)) for _, row in decon_rows]
            == list(range(1, len(decon_rows) + 1))
            and all(int(row.get("max_attempts", 0)) == decon_max_attempts for _, row in decon_rows)
            and all(
                _enum_tail(condition.get("criterion", ""))
                == _enum_tail(decon_until.get("criterion", ""))
                and abs(
                    float(condition.get("threshold", -1.0))
                    - float(decon_until.get("threshold", -2.0))
                )
                <= 1.0e-6
                for condition in completion_rows
            )
        ),
        "bounded_decontamination_condition_met_after_repeated_passes": (
            bool(decon_rows)
            and len(decon_rows) <= decon_max_attempts
            and len(decon_rows) >= 2
            and bool(final_completion.get("met", False))
            and final_decon_attempt < decon_max_attempts + 1
            and _enum_tail(final_completion.get("criterion", ""))
            == "decontamination_remaining_fraction_at_most"
            and float(final_completion.get("observed", float("inf")))
            <= float(final_completion.get("threshold", -1.0))
        ),
        "decontamination_completed_on_visible_surface": (
            bool(decon_rows)
            and all(_completed_action_result(row) for _, row in decon_rows)
            and all(
                details.get("surface_path") == expected_surface_path
                for details in decon_public_details
            )
            and all(
                float(details.get("accepted_contacts", 0)) > 0 for details in decon_public_details
            )
            and all(bool(motion.get("success", False)) for motion in decon_motion)
            and all(float(motion.get("coverage_fraction", 0.0)) >= 0.35 for motion in decon_motion)
        ),
        "decontamination_reduced_activity": (
            bool(activity_before_values)
            and all(
                after < before
                for before, after in zip(activity_before_values, activity_after_values, strict=True)
            )
            and activity_after_values[-1] < activity_before_values[0]
            and float(environment_after.get("surface_source", {}).get("map_activity_bq", 0.0))
            < float(environment_before.get("surface_source", {}).get("map_activity_bq", 0.0))
        ),
        "same_primary_shield_placed_at_25_then_moved_to_65": (
            bool(place_rows)
            and bool(move_rows)
            and _completed_action_result(place_rows[0][1])
            and _completed_action_result(move_rows[0][1])
            and place_details.get("object_path") == expected_shield_path
            and move_details.get("object_path") == expected_shield_path
            and place_details.get("deployment_state") == "deployed"
            and move_details.get("deployment_state") == "deployed"
            and bool(place_motion.get("success", False))
            and bool(move_motion.get("success", False))
            and abs(float(place_details.get("placement_fraction", -1.0)) - 0.25) <= 1.0e-6
            and abs(float(move_details.get("placement_fraction", -1.0)) - 0.65) <= 1.0e-6
            and shield_events[:2]
            and [event[2] for event in shield_events[:2]] == ["place_shield", "move_shield"]
        ),
        "primary_shield_final_metadata_is_65_percent": (
            bool(final_primary_shield)
            and bool(final_primary_shield.get("deployed", False))
            and abs(float(final_primary_shield.get("placement_fraction", -1.0)) - 0.65) <= 1.0e-6
            and shield_displacement_m > 0.25
        ),
        "protected_navigation_completed": (
            bool(protected_moves)
            and _completed_action_result(protected_moves[0][1])
            and bool(protected_motion.get("success", False))
        ),
        "two_second_measurement_completed": (
            bool(measurement_rows)
            and abs(float(measurement.get("duration_s", -1.0)) - 2.0) <= 1.0e-6
            and int(measurement.get("detector_count", 0)) > 0
            and bool(measurement.get("measurements"))
            and any(
                isinstance(row, Mapping)
                and row.get("detector_path") == expected_protected_path
                and abs(float(row.get("duration_s", -1.0)) - 2.0) <= 1.0e-6
                for row in measurement.get("measurements", [])
            )
        ),
        "measurement_robot_returned_home": (
            bool(return_rows)
            and _completed_action_result(return_result)
            and bool(return_motion.get("success", False))
            and returned_home_distance_m <= 0.25
        ),
        "final_status_reported": bool(status_rows),
        "process_order_is_decon_place_move_measure_return_status": (
            all(index >= 0 for index in ordered_indices)
            and list(ordered_indices) == sorted(ordered_indices)
        ),
        "all_results_have_workflow_attempt_metadata": (
            bool(results)
            and all(
                int(row.get("workflow_step", 0)) >= 1
                and int(row.get("attempt", 0)) >= 1
                and int(row.get("max_attempts", 0)) >= 1
                for row in results
            )
        ),
        "collateral_motion_audited_and_bounded": (
            bool(mitigation_rows)
            and not missing_collateral_audit
            and bool(collateral_rows)
            and maximum_collateral_displacement_m <= 0.05
        ),
    }
    return {
        "plan": plan,
        "results": results,
        "expected_paths": {
            "surface": expected_surface_path,
            "primary_shield": expected_shield_path,
            "protected_station": expected_protected_path,
        },
        "decontamination": {
            "planned_max_attempts": decon_max_attempts,
            "planned_completion_condition": dict(decon_until),
            "executed_attempts": len(decon_rows),
            "completion_rows": completion_rows,
            "first_activity_before_bq": (
                activity_before_values[0] if activity_before_values else None
            ),
            "final_activity_after_bq": (
                activity_after_values[-1] if activity_after_values else None
            ),
        },
        "shield_events": [
            {
                "result_index": index,
                "action_type": action_type,
                "action_id": row.get("action_id"),
                "object_path": dict(row.get("public_details", {})).get("object_path"),
                "placement_fraction": dict(row.get("public_details", {})).get("placement_fraction"),
            }
            for index, row, action_type in shield_events
        ],
        "ordered_result_indices": ordered_indices,
        "returned_home_distance_m": returned_home_distance_m,
        "primary_shield_displacement_m": shield_displacement_m,
        "collateral_motion": {
            "audited_rows": collateral_rows,
            "missing_action_ids": missing_collateral_audit,
            "maximum_displacement_m": maximum_collateral_displacement_m,
            "limit_m": 0.05,
        },
        "invariants": invariants,
        "failed_invariants": [name for name, passed in invariants.items() if not passed],
    }


def _local_llm_audit(dashboard: Any) -> dict[str, Any]:
    controller = getattr(dashboard, "_natural_language", None)
    runtime = getattr(controller, "runtime", None)
    interpreter = getattr(controller, "interpreter", None)
    endpoint = str(getattr(interpreter, "endpoint", "") or getattr(runtime, "endpoint", ""))
    parsed = urlparse(endpoint) if endpoint else None
    runtime_status = _enum_tail(getattr(runtime, "status", ""))
    process = getattr(runtime, "process", None)
    audit = {
        "controller_class": type(controller).__name__ if controller is not None else None,
        "runtime_class": type(runtime).__name__ if runtime is not None else None,
        "runtime_status": runtime_status,
        "runtime_process_pid": getattr(process, "pid", None),
        "interpreter_class": (type(interpreter).__name__ if interpreter is not None else None),
        "model": getattr(interpreter, "model", None),
        "endpoint": endpoint,
        "endpoint_host": None if parsed is None else parsed.hostname,
    }
    invariants = {
        "openai_compatible_interpreter_was_instantiated": (
            audit["interpreter_class"] == "OpenAICompatibleCommandInterpreter"
        ),
        "llama_runtime_is_ready": runtime_status in {"ready", "external"},
        "inference_endpoint_is_loopback_only": (
            audit["endpoint_host"] in {"127.0.0.1", "localhost", "::1"}
        ),
        "qwen_model_alias_was_used": audit["model"] == "radcounter-qwen3-4b",
    }
    audit["invariants"] = invariants
    audit["failed_invariants"] = [name for name, passed in invariants.items() if not passed]
    return audit


def _capture_viewport_evidence(app: Any, path: Path) -> dict[str, Any]:
    """Request a fresh PNG and drive rendering long enough to flush it."""

    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    path.parent.mkdir(parents=True, exist_ok=True)
    started_ns = time.time_ns()
    viewport = get_active_viewport()
    if viewport is None:
        return {"path": str(path), "captured": False, "error": "no active viewport"}
    capture_viewport_to_file(viewport, file_path=str(path), is_hdr=False)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline and app.is_running():
        app.update()
        if path.is_file() and path.stat().st_size > 0 and path.stat().st_mtime_ns >= started_ns:
            break
    return {
        "path": str(path.resolve()),
        "captured": (
            path.is_file() and path.stat().st_size > 0 and path.stat().st_mtime_ns >= started_ns
        ),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "modified_ns": path.stat().st_mtime_ns if path.is_file() else None,
    }


def _run_complex_natural_language_validation(
    app: Any,
    args: argparse.Namespace,
    panel: _ValidationPanel,
    dashboard: Any,
    *,
    stage: Any,
    stage_path: Path,
    asset_manifest: Mapping[str, Any],
    robot_config: Any,
    activity_path: Path,
    generator: Any,
    countermeasure_controller: Any,
    measurement_controller: Any,
    simulation: Any,
) -> dict[str, Any]:
    """Run the production NL controller through confirmation and its physical queue."""

    instruction = args.initial_command or DEFAULT_COMPLEX_NATURAL_LANGUAGE_INSTRUCTION
    total_steps = 7
    screenshot_root = args.artifact.parent
    initial_screenshot_path = screenshot_root / (
        f"{args.artifact.stem}-complex-natural-language-initial.png"
    )
    final_screenshot_path = screenshot_root / (
        f"{args.artifact.stem}-complex-natural-language-final.png"
    )
    _configure_complex_facility_camera(stage)
    for _ in range(30):
        app.update()

    environment_before = _environment_audit(
        stage,
        surface_path=robot_config.decon_surface_path,
        activity_path=activity_path,
    )
    physical_state: dict[str, Any] = {
        "measurement_robot_initial_position_m": _world_position(
            stage, robot_config.measurement_articulation
        ).tolist(),
        "primary_shield_initial_position_m": _world_position(stage, "/World/LeadShield").tolist(),
    }
    context_before = dashboard.natural_language_context()
    context_payload = context_before.model_dump(mode="json")
    initial_screenshot = _capture_viewport_evidence(app, initial_screenshot_path)
    panel.update(
        "Complex facility and irregular source preflight",
        0,
        total_steps,
        {
            "instruction": instruction,
            "available_actions": len(context_before.available_actions),
            "environment_failed_invariants": environment_before["failed_invariants"],
            "initial_screenshot": initial_screenshot,
        },
    )
    if environment_before["failed_invariants"]:
        payload = {
            "success": False,
            "mode": "complex_natural_language_validation",
            "instruction": instruction,
            "stage": str(stage_path),
            "environment_before": environment_before,
            "public_context_before": context_payload,
            "screenshots": {"initial": initial_screenshot},
            "failed_invariants": [
                f"environment_before.{name}" for name in environment_before["failed_invariants"]
            ],
        }
        raise ComplexNaturalLanguageValidationError("complex facility preflight failed", payload)

    physical_queue_dispatches: list[dict[str, Any]] = []

    def process_physical_queue() -> None:
        queue = getattr(dashboard, "_physical_command_queue", ())
        if queue:
            step = queue[0][0]
            physical_queue_dispatches.append(
                {
                    "sequence": len(physical_queue_dispatches) + 1,
                    "command": _enum_tail(getattr(step, "command", "")),
                    "candidate_id": getattr(step, "candidate_id", None),
                }
            )
            generator.probe.invalidate_collision_cache()
        dashboard.process_pending_natural_language_actions()

    panel.update(
        "Local LLM interpretation before explicit confirmation",
        0,
        total_steps,
        {"instruction": instruction, "confirm_physical": False},
    )
    plan_payload: dict[str, Any] = {}
    plan_preflight: dict[str, Any] = {}
    try:
        interpreted = _complete_with_updates(
            app,
            dashboard.submit_natural_language_instruction(
                instruction,
                confirm_physical=False,
            ),
            timeout_s=args.natural_language_timeout_s,
        )
        plan_payload = interpreted.validated.plan.model_dump(mode="json")
        context_by_id = {action.action_id: action for action in context_before.available_actions}
        planned_steps = list(interpreted.validated.plan.steps)
        planned_actions = [context_by_id.get(step.candidate_id or "") for step in planned_steps]
        expected_commands = (
            "execute_candidate",
            "execute_candidate",
            "execute_candidate",
            "execute_candidate",
            "measure",
            "return_measurement_robot",
            "show_status",
        )
        preflight_invariants = {
            "seven_ordered_logical_steps": (
                len(planned_steps) == len(expected_commands)
                and tuple(_enum_tail(step.command) for step in planned_steps) == expected_commands
            ),
            "bounded_decontamination_is_first": (
                len(planned_actions) >= 1
                and planned_actions[0] is not None
                and planned_actions[0].action_type == "decontaminate"
                and planned_steps[0].max_attempts == 3
                and planned_steps[0].until is not None
                and _enum_tail(planned_steps[0].until.criterion)
                == "decontamination_remaining_fraction_at_most"
                and abs(planned_steps[0].until.threshold - 0.60) <= 1.0e-6
            ),
            "primary_shield_has_feasible_25_then_65_slots": (
                len(planned_actions) >= 3
                and all(action is not None for action in planned_actions[1:3])
                and all(action.feasible for action in planned_actions[1:3] if action)
                and all(
                    action.target == "/World/LeadShield"
                    for action in planned_actions[1:3]
                    if action
                )
                and [
                    round(float(action.placement_fraction or -1.0), 6)
                    for action in planned_actions[1:3]
                    if action is not None
                ]
                == [0.25, 0.65]
            ),
            "protected_navigation_is_fourth": (
                len(planned_actions) >= 4
                and planned_actions[3] is not None
                and planned_actions[3].action_type == "measure"
                and planned_actions[3].target == "/World/DetectorStations/Protected"
            ),
            "physical_plan_is_waiting_for_confirmation": (
                interpreted.validated.requires_confirmation and not interpreted.executed
            ),
        }
        plan_preflight = {
            "invariants": preflight_invariants,
            "failed_invariants": [
                name for name, passed in preflight_invariants.items() if not passed
            ],
            "shield_candidates": [
                {
                    **action.model_dump(mode="json"),
                    "feasibility_facts": _jsonable(
                        getattr(
                            getattr(dashboard, "_command_candidates", {}).get(action.action_id),
                            "feasibility",
                            None,
                        )
                    ),
                }
                for action in context_before.available_actions
                if action.action_type in {"place_shield", "move_shield"}
            ],
        }
        if plan_preflight["failed_invariants"]:
            raise RuntimeError(
                f"natural-language plan preflight failed: {plan_preflight['failed_invariants']}"
            )
        panel.update(
            "Plan verified; executing explicit confirmation through the physical queue",
            0,
            total_steps,
            {"plan": plan_payload, "plan_preflight": plan_preflight},
        )
        submission = _complete_with_updates(
            app,
            dashboard.confirm_natural_language_instruction(),
            before_update=process_physical_queue,
            timeout_s=args.natural_language_timeout_s,
        )
    except Exception as error:
        partial_results = list(getattr(error, "results", ()))
        environment_after = _environment_audit(
            stage,
            surface_path=robot_config.decon_surface_path,
            activity_path=activity_path,
        )
        physical_state.update(
            {
                "measurement_robot_final_position_m": _world_position(
                    stage, robot_config.measurement_articulation
                ).tolist(),
                "primary_shield_final_position_m": _world_position(
                    stage, "/World/LeadShield"
                ).tolist(),
            }
        )
        local_llm = _local_llm_audit(dashboard)
        partial_audit = _complex_process_audit(
            plan_payload,
            partial_results,
            environment_before,
            environment_after,
            physical_state,
            expected_surface_path=robot_config.decon_surface_path,
        )
        payload = {
            "success": False,
            "mode": "complex_natural_language_validation",
            "instruction": instruction,
            "stage": str(stage_path),
            "error": f"{type(error).__name__}: {error}",
            "local_llm": local_llm,
            "confirmation_requested": True,
            "plan_preflight": plan_preflight,
            "physical_queue_dispatches": physical_queue_dispatches,
            "partial_results": _jsonable(partial_results),
            "environment_before": environment_before,
            "environment_after": environment_after,
            "physical_state": physical_state,
            "process_audit": partial_audit,
            "public_context_before": context_payload,
            "screenshots": {"initial": initial_screenshot},
            "failed_invariants": [
                "natural_language_workflow_completed",
                *local_llm["failed_invariants"],
                *partial_audit["failed_invariants"],
            ],
        }
        raise ComplexNaturalLanguageValidationError(
            "complex natural-language workflow failed", payload
        ) from error

    plan_payload = submission.validated.plan.model_dump(mode="json")
    results = list(submission.results)
    environment_after = _environment_audit(
        stage,
        surface_path=robot_config.decon_surface_path,
        activity_path=activity_path,
    )
    physical_state.update(
        {
            "measurement_robot_final_position_m": _world_position(
                stage, robot_config.measurement_articulation
            ).tolist(),
            "measurement_robot_home_position_m": _jsonable(
                getattr(measurement_controller, "home_position_m", None)
            ),
            "primary_shield_final_position_m": _world_position(stage, "/World/LeadShield").tolist(),
        }
    )
    process_audit = _complex_process_audit(
        plan_payload,
        results,
        environment_before,
        environment_after,
        physical_state,
        expected_surface_path=robot_config.decon_surface_path,
    )
    local_llm = _local_llm_audit(dashboard)
    final_screenshot = _capture_viewport_evidence(app, final_screenshot_path)
    physical_result_count = sum(
        1
        for row in results
        if _enum_tail(row.get("command", "")) in {"execute_candidate", "return_measurement_robot"}
    )
    runner_invariants = {
        "visible_gui_mode_was_enforced": not args.headless,
        "fresh_initial_and_final_viewport_evidence": (
            bool(initial_screenshot.get("captured")) and bool(final_screenshot.get("captured"))
        ),
        "physical_plan_required_confirmation": bool(submission.validated.requires_confirmation),
        "confirmed_plan_executed": bool(submission.executed),
        "physical_actions_used_dashboard_queue": (
            bool(physical_queue_dispatches)
            and len(physical_queue_dispatches) == physical_result_count
            and any(row["command"] == "execute_candidate" for row in physical_queue_dispatches)
            and any(
                row["command"] == "return_measurement_robot" for row in physical_queue_dispatches
            )
        ),
        "actual_loopback_local_llm_used": not local_llm["failed_invariants"],
        "complex_process_invariants_passed": not process_audit["failed_invariants"],
    }
    failed = [name for name, passed in runner_invariants.items() if not passed]
    payload = {
        "success": not failed,
        "mode": "complex_natural_language_validation",
        "stage": str(stage_path),
        "assets": _jsonable(asset_manifest),
        "robot_models": {
            "countermeasure": "Clearpath Ridgeback + Franka Panda",
            "measurement": "NVIDIA Nova Carter",
        },
        "instruction": instruction,
        "natural_language": {
            "plan": plan_payload,
            "warnings": list(submission.validated.warnings),
            "requires_confirmation": submission.validated.requires_confirmation,
            "confirmation_requested": True,
            "executed": submission.executed,
            "results": _jsonable(results),
        },
        "plan_preflight": plan_preflight,
        "local_llm": local_llm,
        "physical_queue_dispatches": physical_queue_dispatches,
        "public_context_before": context_payload,
        "environment_before": environment_before,
        "environment_after": environment_after,
        "physical_state": physical_state,
        "process_audit": process_audit,
        "screenshots": {
            "initial": initial_screenshot,
            "final": final_screenshot,
        },
        "runner_invariants": runner_invariants,
        "failed_invariants": failed,
        "transport_statistics": _jsonable(simulation.transport.statistics),
        "controller_audit": {
            "countermeasure_dof_count": len(countermeasure_controller.dof_names),
            "measurement_dof_count": len(measurement_controller.robot.dof_names),
            "franka_arm_joint_excursion_rad": (countermeasure_controller.arm_joint_excursion_rad),
        },
    }
    panel.update(
        "Complex natural-language validation completed",
        total_steps if not failed else 0,
        total_steps,
        payload,
    )
    if failed:
        raise ComplexNaturalLanguageValidationError(
            f"complex validation invariants failed: {failed}", payload
        )
    return payload


def _monitor_update(app: Any) -> None:
    for _ in range(4):
        app.update()
    time.sleep(0.10)
    app.update()


def _exercise_robot_monitor(app: Any, dashboard: Any) -> dict[str, object]:
    """Exercise camera state without authoring a second rendered viewport."""

    from pxr import Gf

    monitor = dashboard._robot_monitor
    if not monitor.robots:
        raise RuntimeError("robot monitor smoke test requires at least one catalog robot")
    robot_id = monitor.robots[0].robot_id

    monitor.follow_robot(robot_id)
    _monitor_update(app)
    follow = monitor.audit()
    if follow["camera_mode"] != "follow":
        raise RuntimeError(f"follow camera did not activate: {follow}")

    monitor.onboard_robot(robot_id)
    _monitor_update(app)
    onboard = monitor.audit()
    if onboard["camera_mode"] != "onboard":
        raise RuntimeError(f"onboard camera did not activate: {onboard}")

    monitor.begin_operation(
        robot_id=robot_id,
        operation="Decontamination",
        phase="navigating",
        route_m=((0.0, 0.0, 0.0), (1.0, 0.5, 0.0)),
        target_m=(1.0, 0.5, 1.0),
        target_path="/World/RobotMonitorSmokeTarget",
        auto_work_view=True,
    )
    _monitor_update(app)
    monitor.update_countermeasure_progress(
        {
            "phase": "decontaminating",
            "progress": 0.63,
            "coverage_fraction": 0.57,
        }
    )
    _monitor_update(app)
    work = monitor.audit()
    if work["camera_mode"] != "work" or not math.isclose(work["progress"], 0.63):
        raise RuntimeError(f"work camera/progress did not activate: {work}")

    camera_op = monitor._camera_op
    if camera_op is None or camera_op.Get() is None:
        raise RuntimeError("operator camera did not author a transform")
    camera_op.Set(Gf.Matrix4d(1.0))
    _monitor_update(app)
    manual_cancel = monitor.audit()
    if manual_cancel["camera_mode"] != "free":
        raise RuntimeError(f"manual camera edit did not cancel tracking: {manual_cancel}")

    monitor.overview()
    _monitor_update(app)
    overview = monitor.audit()
    if overview["camera_mode"] != "overview":
        raise RuntimeError(f"overview camera did not activate: {overview}")
    monitor.stop_follow(manual=False)
    return {
        "follow": follow,
        "onboard": onboard,
        "work": work,
        "manual_cancel": manual_cancel,
        "overview": overview,
    }


def _run_configurable_system(
    app: Any,
    args: argparse.Namespace,
    panel: _ValidationPanel,
    dashboard: Any,
    selection: Any,
) -> dict[str, object]:
    """Load a catalog selection without assuming the vertical-slice task layout."""

    import omni.usd
    from radcounter.isaac.runtime import IsaacRadiationSimulation, RuntimeConfiguration
    from radcounter.isaac.system_profile import (
        compose_selected_system,
        prepare_environment_stage,
    )

    panel.update(
        "Preparing selected environment",
        0,
        3,
        selection.as_dict(),
    )
    stage_path, environment_manifest = prepare_environment_stage(selection)
    context = omni.usd.get_context()
    if not context.open_stage(str(stage_path)):
        raise RuntimeError(f"failed to open selected environment stage: {stage_path}")
    for _ in range(30):
        app.update()
    stage = context.get_stage()
    panel.update("Composing selected robots and detectors", 1, 3, selection.as_dict())
    composed = compose_selected_system(stage, selection, stage_path=stage_path)
    for _ in range(30):
        app.update()
    runtime_configuration = replace(
        RuntimeConfiguration.from_json(composed.runtime_config_path),
        seed=args.seed,
    )
    simulation = IsaacRadiationSimulation(stage, runtime_configuration)
    dashboard.configure_system_paths(
        stage_path=composed.stage_path,
        config_path=composed.runtime_config_path,
        display_name=selection.profile.display_name,
        simulation=simulation,
        selection=selection,
    )
    panel.update("Selected system is ready", 3, 3, composed.profile_manifest)
    robot_monitor_validation = (
        _exercise_robot_monitor(app, dashboard) if args.robot_monitor_smoke_test else None
    )
    if args.interactive:
        panel.hide()
    return {
        "success": True,
        "mode": "configurable",
        "selection": selection.as_dict(),
        "stage": str(composed.stage_path),
        "environment_manifest": str(environment_manifest),
        "runtime_config": str(composed.runtime_config_path),
        "robot_paths": composed.robot_paths,
        "detector_paths": composed.detector_paths,
        "disabled_detector_paths": list(composed.disabled_detector_paths),
        "rebased_asset_paths": composed.rebased_asset_paths,
        "source_count": len(simulation.sources),
        "detector_count": len(simulation.detectors),
        "robot_monitor": dashboard.robot_monitor_audit(),
        "robot_monitor_validation": robot_monitor_validation,
        "note": (
            "The configurable session composes selected assets without adding the "
            "vertical-slice-only decontamination task layout."
        ),
    }


def _run_validation(
    app: Any,
    args: argparse.Namespace,
    panel: _ValidationPanel,
    dashboard: Any,
) -> dict:
    selection = _selected_system(args)
    if args.decontamination_smoke_test and selection.robot_set.kind != "decommissioning":
        raise RuntimeError(
            "--decontamination-smoke-test requires the explicit central-catalog "
            "robot set 'articulated-decommissioning'; do not inject an unselected "
            "task robot into the operator roster"
        )
    if selection.configurable and not args.decontamination_smoke_test:
        return _run_configurable_system(app, args, panel, dashboard, selection)

    import omni.usd
    from isaacsim.core.api import World
    from isaacsim.core.prims import SingleArticulation
    from isaacsim.robot.wheeled_robots.robots import WheeledRobot
    from radcounter.isaac.planning import (
        IsaacActionCandidateGenerator,
        SceneCandidateConfig,
    )
    from radcounter.isaac.robot import (
        ContactDrivenDecontaminator,
        DecontaminationConfig,
        NovaCarterController,
        RealRobotAssetConfig,
        RidgebackFrankaController,
        add_real_robot_references,
        author_real_robot_task_scene,
        create_decontamination_activity_map,
        enable_real_robot_extensions,
    )
    from radcounter.isaac.runtime import IsaacRadiationSimulation, RuntimeConfiguration
    from radcounter.isaac.workflow import (
        IsaacPublicPoissonEstimator,
        IsaacWorkflowServices,
    )

    from radcounter.core.actions import ResourceState
    from radcounter.core.models import BeliefState, RevisionState
    from radcounter.core.models.actions import ActionStatus, ActionType
    from radcounter.core.planning import DeterministicFeasibilityChecker

    enable_real_robot_extensions()
    for _ in range(20):
        app.update()
    context = omni.usd.get_context()
    environment_manifest = None
    if args.decontamination_smoke_test:
        from radcounter.isaac.system_profile import prepare_environment_stage

        environment_operation_offset_m = selection.environment_config.translation_world_m
        stage_path, environment_manifest = prepare_environment_stage(selection)
    else:
        environment_operation_offset_m = None
        stage_path = ROOT / "assets/environments/radcounter_vertical_slice.usda"
    if not context.open_stage(str(stage_path)):
        raise RuntimeError(f"failed to open {stage_path}")
    for _ in range(24):
        app.update()
    stage = context.get_stage()
    if args.decontamination_smoke_test:
        decon_anchor = selection.spawn_anchor("decon-surface")
        ground_primary = selection.spawn_anchor("ground-primary")
        ground_secondary = selection.spawn_anchor("ground-secondary")
        camera_eye = selection.spawn_anchor("inspection-camera-eye")
        camera_target = selection.spawn_anchor("inspection-camera-target")
        robot_config = replace(
            RealRobotAssetConfig(),
            decon_workbench_center_m=decon_anchor.translation_m,
            shield_initial_position_m=(0.0, 3.0, ground_primary.translation_m[2]),
            include_validation_facility=False,
        )
    else:
        decon_anchor = None
        ground_primary = None
        ground_secondary = None
        camera_eye = None
        camera_target = None
        robot_config = replace(
            RealRobotAssetConfig(),
            decon_workbench_center_m=(14.39, 0.80, 1.15),
            shield_initial_position_m=(4.80, 2.80, 0.0),
        )
    asset_manifest = add_real_robot_references(stage, config=robot_config)
    panel.update("Loading articulated robot assets", 0, 8, asset_manifest)
    for _ in range(120):
        app.update()
    artifact_root = args.artifact.parent
    activity_path = create_decontamination_activity_map(
        artifact_root / "runtime_workbench_activity.npz"
    )
    floor_activity_before = _activity_total(activity_path)
    author_real_robot_task_scene(
        stage,
        activity_path,
        ROOT / "configs/decontamination/concrete_surface.synthetic.yaml",
        config=robot_config,
    )
    if args.decontamination_smoke_test:
        # This run selects only contact decontamination. Do not leave an
        # unrelated movable validation shield in the imported CAD where its
        # settling motion could be mistaken for task collateral.
        shield = stage.GetPrimAtPath(robot_config.shield_path)
        if shield.IsValid():
            stage.RemovePrim(robot_config.shield_path)
    if args.decontamination_smoke_test:
        assert camera_eye is not None and camera_target is not None
        _configure_decon_room_camera(
            stage,
            eye_m=camera_eye.translation_m,
            target_m=camera_target.translation_m,
        )
    else:
        _configure_camera(stage)
    for _ in range(60):
        app.update()

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
    )
    franka_articulation = world.scene.add(
        SingleArticulation(
            robot_config.countermeasure_root,
            name="radcounter_gui_ridgeback_franka",
        )
    )
    carter_articulation = world.scene.add(
        WheeledRobot(
            prim_path=robot_config.measurement_articulation,
            name="radcounter_gui_nova_carter",
            wheel_dof_names=list(NovaCarterController.wheel_names),
        )
    )
    world.reset()
    spawn_pose_before_settle = _world_position(stage, robot_config.panda_base_path)
    for _ in range(90):
        world.step(render=False)
    spawn_pose_after_settle = _world_position(stage, robot_config.panda_base_path)
    spawn_vertical_drop_m = max(
        0.0,
        float(spawn_pose_before_settle[2] - spawn_pose_after_settle[2]),
    )
    if args.decontamination_smoke_test and spawn_vertical_drop_m > 0.15:
        raise RuntimeError(
            "catalog robot spawn is unsupported by the selected CAD floor: "
            f"vertical drop={spawn_vertical_drop_m:.3f} m"
        )
    stepper = _GuiStepper(app, args.frame_delay_s, world)
    countermeasure_controller = RidgebackFrankaController(
        stage,
        stepper,
        config=robot_config,
        articulation=franka_articulation,
    )
    measurement_controller = NovaCarterController(
        stage,
        stepper,
        config=robot_config,
        articulation=carter_articulation,
    )
    countermeasure_controller.progress_callback = dashboard.update_countermeasure_progress
    measurement_controller.progress_callback = dashboard.update_navigation_progress
    if args.decontamination_smoke_test:
        assert ground_primary is not None and ground_secondary is not None
        measurement_controller.set_initial_pose(ground_secondary.translation_m)
        surface_x, surface_y, _ = robot_config.decon_workbench_center_m
        staging = countermeasure_controller.navigate_route(
            (
                ground_primary.translation_m,
                (surface_x - 0.98, surface_y - 0.55, ground_primary.translation_m[2]),
            ),
            final_yaw_rad=0.0,
        )
    else:
        measurement_controller.set_initial_pose((-4.1, -2.5, 0.0))
        staging = countermeasure_controller.navigate_route(
            ((0.0, -1.00, 0.0), (1.20, -1.00, 0.0)),
            final_yaw_rad=0.0,
        )
    if not staging.success:
        raise RuntimeError(f"countermeasure staging motion failed: {staging.message}")
    if args.interactive:
        # Keep later operator-triggered physics motion observable instead of
        # advancing hundreds of rendered steps as fast as the GPU allows.
        stepper.frame_delay_s = max(stepper.frame_delay_s, 1.0 / 120.0)
    runtime_config_path = ROOT / "configs/scenarios/vertical_slice.runtime.json"
    simulation = IsaacRadiationSimulation(
        stage,
        replace(RuntimeConfiguration.from_json(runtime_config_path), seed=args.seed),
    )
    dashboard.simulation = simulation
    candidate_config = SceneCandidateConfig(
        countermeasure_pose_path=robot_config.panda_base_path,
        measurement_pose_path=robot_config.measurement_articulation,
        end_effector_offset_m=(0.72, 0.0, 0.0),
        decon_end_effector_offset_m=(0.90, 0.0, 0.0),
        manipulator_workspace_m=0.95,
        mobile_clearance_m=0.55,
        ignored_collision_paths=(
            ("/World/Environment",) if args.decontamination_smoke_test else ()
        ),
    )
    generator = IsaacActionCandidateGenerator(
        stage,
        simulation,
        controller=countermeasure_controller,
        config=candidate_config,
    )
    belief = BeliefState(
        (robot_config.decon_surface_path,),
        np.asarray([2.4e7]),
        np.asarray([[4.0e12]]),
        RevisionState(),
    )
    resources = ResourceState(
        remaining_measurement_time_s=180.0,
        remaining_work_time_s=600.0,
        remaining_robot_runtime_s={
            "/World/MeasurementRobot": 300.0,
            "/World/CountermeasureRobot": 600.0,
        },
        remaining_shield_units={"lead": 3},
        remaining_decon_media=120.0,
        remaining_clean_water_l=40.0,
        remaining_wastewater_capacity_l=40.0,
        remaining_countermeasure_count=12,
    )
    decontaminator = ContactDrivenDecontaminator(
        stage,
        robot_config.decon_tool_path,
        robot_config.decon_surface_path,
        DecontaminationConfig(
            footprint_points_local_m=tuple(
                (float(local_x), float(local_y), 0.0)
                for local_x in np.linspace(-0.085, 0.085, 9)
                for local_y in np.linspace(-0.065, 0.065, 9)
            ),
            treatment_axis_local=(0.0, 0.0, 1.0),
            max_contact_distance_m=0.045,
            max_surface_speed_m_s=0.35,
            transfer_mode="transfer_to_waste",
        ),
    )

    public_estimator = IsaacPublicPoissonEstimator(simulation)

    services = IsaacWorkflowServices(
        simulation,
        generator,
        public_estimator,
        controller=countermeasure_controller,
        measurement_controller=measurement_controller,
        decontaminators={robot_config.decon_surface_path: decontaminator},
        resources=resources,
        reestimate_after_verification=True,
        simulation_time_s=lambda: float(world.current_time),
        artifact_path=ROOT / "artifacts/ui/latest_workflow.json",
    )
    checker = DeterministicFeasibilityChecker()
    _complete(services.initialize())
    initial_resource_state = _jsonable(services.workflow_view()["resources"])
    dashboard.configure_system_paths(
        stage_path=stage_path,
        config_path=ROOT / "configs/scenarios/vertical_slice.runtime.json",
        display_name=selection.profile.display_name,
        simulation=simulation,
        selection=selection,
    )
    dashboard.bind_workflow(services, belief)

    if args.decontamination_smoke_test:
        candidates = generator.generate_decon_actions(belief)
        if not candidates:
            raise RuntimeError("the selected environment generated no decontamination action")
        candidate = candidates[0]
        feasibility = checker.evaluate(candidate, resources)
        if not feasibility.feasible:
            raise RuntimeError(
                "decontamination action is infeasible in the selected environment: "
                f"{feasibility.reasons}; facts={_jsonable(candidate.feasibility)}"
            )
        action = replace(
            candidate.action,
            predicted_duration_s=args.decon_duration_s,
            parameters={
                **candidate.action.parameters,
                "duration_s": args.decon_duration_s,
                "decon_media": args.decon_duration_s,
            },
        )
        activity_before = _activity_total(activity_path)
        robot_before = _world_position(stage, robot_config.countermeasure_root)
        panel.update(
            "Executing Fukushima articulated contact decontamination",
            0,
            1,
            _jsonable(candidate.feasibility),
        )
        dashboard._robot_monitor.begin_action(action)
        try:
            result = _complete(services.execute(action))
        except Exception:
            dashboard._robot_monitor.finish_action(success=False)
            raise
        dashboard._robot_monitor.finish_action(
            success=True,
            public_details=result.public_details,
        )
        activity_after = _activity_total(activity_path)
        robot_after = _world_position(stage, robot_config.countermeasure_root)
        public_details = dict(result.public_details)
        motion = dict(public_details.get("motion_audit", {}))
        accepted_contacts = int(public_details.get("accepted_contacts", 0))
        coverage_fraction = float(motion.get("coverage_fraction", 0.0))
        invariants = {
            "articulated_action_completed": result.status == ActionStatus.COMPLETED,
            "visible_irregular_surface_reduced": activity_after < activity_before,
            "verified_contacts_recorded": accepted_contacts > 0,
            "meaningful_surface_coverage": coverage_fraction >= 0.35,
            "franka_arm_moved": countermeasure_controller.arm_joint_excursion_rad > 0.2,
            "fukushima_geometry_in_transport": len(simulation.transport.geometry_paths) >= 995,
            "catalog_spawn_stayed_supported": spawn_vertical_drop_m <= 0.15,
            "separate_validation_facility_absent": not stage.GetPrimAtPath(
                "/World/RemoteDeconFacility"
            ).IsValid(),
        }
        failed = [name for name, passed in invariants.items() if not passed]
        payload = {
            "success": not failed,
            "mode": "fukushima_decontamination_smoke_test",
            "selection": selection.as_dict(),
            "stage": str(stage_path),
            "environment_manifest": (
                None if environment_manifest is None else str(environment_manifest)
            ),
            "environment_operation_offset_m": environment_operation_offset_m,
            "spawn_anchors": {
                "countermeasure": "ground-primary",
                "measurement": "ground-secondary",
                "decontamination_surface": "decon-surface",
            },
            "spawn_vertical_drop_m": spawn_vertical_drop_m,
            "robot_model": "Clearpath Ridgeback + Franka Panda",
            "surface_path": robot_config.decon_surface_path,
            "activity_before_bq": activity_before,
            "activity_after_bq": activity_after,
            "removed_fraction": (
                (activity_before - activity_after) / activity_before
                if activity_before > 0.0
                else 0.0
            ),
            "accepted_contacts": accepted_contacts,
            "coverage_fraction": coverage_fraction,
            "robot_displacement_m": float(np.linalg.norm(robot_after - robot_before)),
            "arm_joint_excursion_rad": countermeasure_controller.arm_joint_excursion_rad,
            "transport_geometries": len(simulation.transport.geometry_paths),
            "result": _jsonable(result.public_view()),
            "robot_monitor": dashboard.robot_monitor_audit(),
            "invariants": invariants,
            "failed_invariants": failed,
        }
        dashboard.set_workflow_view(services.workflow_view())
        assert camera_eye is not None and camera_target is not None
        _configure_decon_room_camera(
            stage,
            eye_m=camera_eye.translation_m,
            target_m=camera_target.translation_m,
        )
        panel.update(
            "Completed Fukushima articulated contact decontamination",
            1,
            1,
            payload,
        )
        if failed:
            raise DecontaminationSmokeValidationError(
                f"Fukushima decontamination invariants failed: {failed}", payload
            )
        return payload

    if args.complex_natural_language_validation:
        return _run_complex_natural_language_validation(
            app,
            args,
            panel,
            dashboard,
            stage=stage,
            stage_path=stage_path,
            asset_manifest=asset_manifest,
            robot_config=robot_config,
            activity_path=activity_path,
            generator=generator,
            countermeasure_controller=countermeasure_controller,
            measurement_controller=measurement_controller,
            simulation=simulation,
        )

    if args.interactive:
        context_view = dashboard.natural_language_context()
        panel.update(
            "Ready for natural-language operation",
            0,
            0,
            {
                "languages": ["en"],
                "available_actions": len(context_view.available_actions),
                "local_inference": "llama.cpp + Qwen3-4B GGUF",
            },
        )
        panel.hide()
        command_result = None
        if args.initial_command:
            if args.record_prompt_video is not None:
                dashboard.show_building_overview()
                for _ in range(30):
                    app.update()
                recording_process = _start_display_recording(args)
                try:
                    _hold(app, args.prompt_pre_hold_s)
                    _type_visible_instruction(
                        app,
                        dashboard,
                        args.initial_command,
                        character_delay_s=args.prompt_typing_delay_s,
                    )
                    _hold(app, args.prompt_post_hold_s)
                    submission = _complete_with_updates(
                        app,
                        dashboard.interpret_natural_language_instruction(
                            args.initial_command,
                        ),
                        before_update=dashboard.process_pending_natural_language_actions,
                        timeout_s=args.natural_language_timeout_s,
                    )
                    _hold(app, args.prompt_confirm_hold_s)
                    submission = _complete_with_updates(
                        app,
                        dashboard.execute_confirmed_natural_language_instruction(),
                        before_update=dashboard.process_pending_natural_language_actions,
                        timeout_s=args.natural_language_timeout_s,
                    )
                    _hold(app, args.record_final_hold_s)
                finally:
                    _stop_display_recording(recording_process)
            else:
                submission = _complete_with_updates(
                    app,
                    dashboard.submit_natural_language_instruction(
                        args.initial_command,
                        confirm_physical=args.confirm_initial_command,
                    ),
                    before_update=dashboard.process_pending_natural_language_actions,
                    timeout_s=args.natural_language_timeout_s,
                )
            command_result = {
                "instruction": args.initial_command,
                "plan": submission.validated.plan.model_dump(mode="json"),
                "executed": submission.executed,
                "results": list(submission.results),
                "video": (
                    None
                    if args.record_prompt_video is None
                    else str(args.record_prompt_video.expanduser().resolve())
                ),
            }
            if submission.executed and args.record_prompt_video is None:
                _configure_decon_room_camera(stage)
        return {
            "success": True,
            "mode": "interactive",
            "stage": str(stage_path),
            "robot_models": {
                "countermeasure": "Clearpath Ridgeback + Franka Panda",
                "measurement": "NVIDIA Nova Carter",
            },
            "natural_language": {
                "languages": ["en"],
                "backend": "bundled llama.cpp",
                "model": "Qwen3-4B-Q4_K_M.gguf",
                "available_actions": len(context_view.available_actions),
                "confirmation_required_for_physical_actions": True,
                "initial_command": command_result,
            },
        }

    total_operations = 8
    completed = 0
    records: list[dict[str, Any]] = []
    initial_robot_position = _world_position(stage, robot_config.measurement_articulation)
    initial_shield_position = _world_position(stage, "/World/LeadShield")
    initial_drum_position = _world_position(stage, "/World/HiddenContaminatedDrum")
    initial_obstacle_position = _world_position(stage, "/World/MovableObstacle")
    protected = "/World/DetectorStations/Protected"
    initial_protected_rate = simulation.expected_rates(detector_paths=[protected])[protected]
    initial_spectrum_components = _spectrum_components(
        simulation.expected_spectrum_components(detector_paths=[protected])[protected]
    )

    def execute_candidate(
        label: str,
        candidate: Any,
        *,
        action_override: Any | None = None,
        update_residual: bool = True,
    ) -> None:
        nonlocal belief, completed
        report = checker.evaluate(candidate, resources)
        if not report.feasible:
            action = candidate.action if action_override is None else action_override
            parameters = action.parameters
            diagnostics: dict[str, Any] = {
                "reasons": report.reasons,
                "facts": _jsonable(candidate.feasibility),
                "parameters": _jsonable(parameters),
            }
            if "pickup_base_position_m" in parameters:
                robot_position = generator.probe.world_position(
                    generator.config.countermeasure_pose_path
                    or generator.config.countermeasure_robot_path
                )
                pickup_position = np.asarray(parameters["pickup_base_position_m"])
                placement_position = np.asarray(parameters["placement_base_position_m"])
                excluded = (
                    () if action.target_prim_path is None else (str(action.target_prim_path),)
                )
                diagnostics.update(
                    {
                        "robot_position_m": robot_position.tolist(),
                        "to_pickup_blockers": _path_blockers(
                            generator.probe,
                            robot_position,
                            pickup_position,
                            excluded_paths=excluded,
                        ),
                        "to_placement_blockers": _path_blockers(
                            generator.probe,
                            pickup_position,
                            placement_position,
                            excluded_paths=excluded,
                        ),
                    }
                )
            raise RuntimeError(f"{label} is infeasible: {diagnostics}")
        action = candidate.action if action_override is None else action_override
        prediction = services.preview(action, belief)
        panel.update(f"Executing: {label}", completed, total_operations, prediction)
        result = _complete(services.execute(action))
        if result.status != ActionStatus.COMPLETED:
            raise RuntimeError(f"{label} failed: {result.public_details}")
        verification = _complete(services.verify(action))
        diagnosis = None
        if update_residual:
            diagnosis = services.diagnose(prediction, verification)
            belief = services.update(belief, diagnosis)
        completed += 1
        row = {
            "label": label,
            "action": _jsonable(action),
            "feasibility": _jsonable(report),
            "result": _jsonable(result.public_view()),
            "verification": _measurement_rows(verification),
            "diagnosis": _jsonable(diagnosis),
        }
        records.append(row)
        dashboard.set_workflow_view(services.workflow_view())
        panel.update(f"Completed: {label}", completed, total_operations, row)
        _hold(app, args.phase_hold_s)

    measurement_candidates = generator.generate_measurement_actions(belief)
    if len(measurement_candidates) < 4:
        raise RuntimeError("the scene did not generate all four measurement stations")
    measurement_station_paths = tuple(
        candidate.action.target_prim_path for candidate in measurement_candidates
    )
    executed_measurement_action_ids: list[str] = []
    station_executions: list[dict[str, object]] = []
    panel.update("Moving measurement robot through all stations", completed, total_operations)
    for station_path in measurement_station_paths:
        # Each route must begin at Nova Carter's live pose.  Reusing the routes
        # generated before the first station can send a later leg through an
        # obstacle because those routes all share the original start position.
        # Rebuild collision bounds too: movable props may have settled since
        # the dashboard's initial candidate preview populated the cache.
        generator.probe.invalidate_collision_cache()
        live_candidates = generator.generate_measurement_actions(belief)
        candidate = next(
            item for item in live_candidates if item.action.target_prim_path == station_path
        )
        report = checker.evaluate(candidate, resources)
        if not report.feasible:
            raise RuntimeError(
                f"measurement move {candidate.action.action_id} is infeasible: {report.reasons}"
            )
        result = _complete(services.execute(candidate.action))
        if result.status != ActionStatus.COMPLETED:
            position = _world_position(stage, robot_config.measurement_articulation)
            target = candidate.action.target_pose_world[:3, 3]
            direct_blockers = _path_blockers(
                generator.probe,
                position,
                target,
                moving_robot_path=robot_config.measurement_root,
            )
            raise RuntimeError(
                "measurement move failed: "
                f"details={result.public_details}, position_m={position.tolist()}, "
                f"target_m={target.tolist()}, "
                f"direct_blockers={direct_blockers}"
            )
        executed_measurement_action_ids.append(candidate.action.action_id)
        station_executions.append(
            {
                "action": _jsonable(candidate.action),
                "result": _jsonable(result.public_view()),
            }
        )
    initial_measurement = _complete(services.measure())
    belief = services.estimate(initial_measurement, None)
    initial_estimate = _jsonable(services.workflow_view()["estimate"])
    initial_estimator_audit = _jsonable(public_estimator.last_audit)
    completed += 1
    records.append(
        {
            "label": "all-station measurement",
            "station_actions": executed_measurement_action_ids,
            "station_executions": station_executions,
            "measurement": _measurement_rows(initial_measurement),
        }
    )
    panel.update("Completed: all-station measurement", completed, total_operations, records[-1])
    _hold(app, args.phase_hold_s)

    decon_candidate = generator.generate_decon_actions(belief)[0]
    decon_action = replace(
        decon_candidate.action,
        predicted_duration_s=args.decon_duration_s,
        parameters={
            **decon_candidate.action.parameters,
            "duration_s": args.decon_duration_s,
            "decon_media": args.decon_duration_s,
        },
    )
    execute_candidate(
        "IK raster contact decontamination",
        decon_candidate,
        action_override=decon_action,
    )

    generated_shields = generator.generate_shield_actions(belief)
    shield_candidates = [
        item for item in generated_shields if checker.evaluate(item, resources).feasible
    ]
    if not shield_candidates:
        diagnostics = [
            {
                "action_id": item.action.action_id,
                "target_m": item.action.target_pose_world[:3, 3].tolist(),
                "facts": _jsonable(item.feasibility),
                "reasons": checker.evaluate(item, resources).reasons,
                "to_pickup_blockers": _path_blockers(
                    generator.probe,
                    generator.probe.world_position(
                        generator.config.countermeasure_pose_path
                        or generator.config.countermeasure_robot_path
                    ),
                    np.asarray(item.action.parameters["pickup_base_position_m"]),
                    excluded_paths=(str(item.action.target_prim_path),),
                ),
                "to_placement_blockers": _path_blockers(
                    generator.probe,
                    np.asarray(item.action.parameters["pickup_base_position_m"]),
                    np.asarray(item.action.parameters["placement_base_position_m"]),
                    excluded_paths=(str(item.action.target_prim_path),),
                ),
            }
            for item in generated_shields
        ]
        raise RuntimeError(f"the scene generated no feasible shield placement: {diagnostics}")
    shield_candidate = shield_candidates[0]
    execute_candidate("shield placement", shield_candidate)

    current_shield = _world_position(stage, "/World/LeadShield")
    generated_corrections = generator.generate_shield_actions(belief)
    correction_candidates = [
        item
        for item in generated_corrections
        if item.action.action_type == ActionType.MOVE_SHIELD
        and item.action.target_prim_path == shield_candidate.action.target_prim_path
        and checker.evaluate(item, resources).feasible
    ]
    if not correction_candidates:
        diagnostics = [
            {
                "action_id": item.action.action_id,
                "target_m": item.action.target_pose_world[:3, 3].tolist(),
                "facts": _jsonable(item.feasibility),
                "reasons": checker.evaluate(item, resources).reasons,
            }
            for item in generated_corrections
        ]
        raise RuntimeError(f"the scene generated no feasible shield correction: {diagnostics}")
    correction = max(
        correction_candidates,
        key=lambda item: float(
            np.linalg.norm(item.action.target_pose_world[:3, 3] - current_shield)
        ),
    )
    execute_candidate("shield pose correction", correction)

    object_candidates = generator.generate_move_remove_actions(belief)
    drum_move = next(
        item
        for item in object_candidates
        if item.action.action_type == ActionType.MOVE_OBJECT
        and item.action.target_prim_path == "/World/HiddenContaminatedDrum"
    )
    execute_candidate("contaminated object relocation", drum_move)

    object_candidates = generator.generate_move_remove_actions(belief)
    drum_remove = next(
        item
        for item in object_candidates
        if item.action.action_type == ActionType.REMOVE_OBJECT
        and item.action.target_prim_path == "/World/HiddenContaminatedDrum"
    )
    execute_candidate("contaminated object disposal", drum_remove)

    object_candidates = generator.generate_move_remove_actions(belief)
    obstacle_move = next(
        item
        for item in object_candidates
        if item.action.action_type == ActionType.MOVE_OBJECT
        and item.action.target_prim_path == "/World/MovableObstacle"
    )
    execute_candidate("non-contaminated obstacle relocation", obstacle_move)

    final_measurement = _complete(services.verify(obstacle_move.action))
    final_prediction = {
        "detector_path": protected,
        "predicted_rate_proxy_cps": initial_protected_rate,
    }
    final_diagnosis = services.diagnose(final_prediction, final_measurement)
    belief = services.update(belief, final_diagnosis)
    completed += 1
    records.append(
        {
            "label": "final verification and belief update",
            "measurement": _measurement_rows(final_measurement),
            "diagnosis": _jsonable(final_diagnosis),
            "belief_revision": _jsonable(belief.revision),
        }
    )
    dashboard.set_workflow_view(services.workflow_view())
    panel.update("Completed: final verification", completed, total_operations, records[-1])

    floor_activity_after = _activity_total(activity_path)
    final_robot_position = _world_position(stage, robot_config.measurement_articulation)
    final_shield_position = _world_position(stage, "/World/LeadShield")
    final_obstacle_position = _world_position(stage, "/World/MovableObstacle")
    drum = stage.GetPrimAtPath("/World/HiddenContaminatedDrum")
    drum_secured = False
    if drum and drum.IsValid() and drum.IsActive():
        source_enabled = drum.GetAttribute("rad:source:enabled")
        disposed = drum.GetAttribute("rad:disposal:disposed")
        disposition = drum.GetAttribute("rad:disposal:disposition")
        contained = drum.GetAttribute("rad:source:contained")
        containment_path = drum.GetAttribute("rad:source:containmentPrimPath")
        drum_secured = (
            source_enabled
            and source_enabled.HasAuthoredValueOpinion()
            and bool(source_enabled.Get())
            and disposed
            and disposed.HasAuthoredValueOpinion()
            and bool(disposed.Get())
            and disposition
            and disposition.Get() == "shielded_storage"
            and contained
            and bool(contained.Get())
            and containment_path
            and containment_path.Get() == "/World/DisposalStorage"
        )
    final_protected_rate = simulation.expected_rates(detector_paths=[protected])[protected]
    final_spectrum_components = _spectrum_components(
        simulation.expected_spectrum_components(detector_paths=[protected])[protected]
    )
    final_workflow_view = services.workflow_view()
    physics_data_class = str(
        json.loads(runtime_config_path.read_text(encoding="utf-8"))["physics_data_class"]
    )

    invariants = {
        "all_operations_completed": completed == total_operations,
        "measurement_robot_moved": float(
            np.linalg.norm(final_robot_position - initial_robot_position)
        )
        > 0.25,
        "floor_activity_reduced": floor_activity_after < floor_activity_before,
        "shield_moved": float(np.linalg.norm(final_shield_position - initial_shield_position))
        > 0.25,
        "contaminated_drum_relocated_before_disposal": any(
            row["label"] == "contaminated object relocation" for row in records
        ),
        "contaminated_drum_secured_in_shielded_storage": drum_secured,
        "obstacle_moved": float(np.linalg.norm(final_obstacle_position - initial_obstacle_position))
        > 0.25,
        "post_action_measurement_available": bool(final_measurement),
        "residual_available": final_diagnosis is not None,
        "native_transport_used": simulation.transport.statistics["native_trace_calls"] > 0,
        "franka_arm_moved": countermeasure_controller.arm_joint_excursion_rad > 0.2,
        "articulated_countermeasure_robot": len(countermeasure_controller.dof_names) >= 12,
        "articulated_measurement_robot": len(measurement_controller.robot.dof_names) >= 7,
    }
    failed = [name for name, passed in invariants.items() if not passed]
    if failed:
        raise RuntimeError(f"final GUI invariants failed: {failed}")

    return {
        "success": True,
        "evidence_class": EvidenceClass.PHYSICAL_ROBOT_EXECUTION.value,
        "seed": args.seed,
        "physics_data_class": physics_data_class,
        "runtime_config": str(runtime_config_path),
        "runtime_config_sha256": sha256_file(runtime_config_path),
        "stage": str(stage_path),
        "stage_sha256": sha256_file(stage_path),
        "assets": asset_manifest,
        "robot_models": {
            "countermeasure": "Clearpath Ridgeback + Franka Panda",
            "measurement": "NVIDIA Nova Carter",
        },
        "dof_audit": {
            "countermeasure_dofs": list(countermeasure_controller.dof_names),
            "measurement_dofs": list(measurement_controller.robot.dof_names),
            "franka_arm_joint_excursion_rad": (countermeasure_controller.arm_joint_excursion_rad),
        },
        "motion_policy": {
            "teleport_during_operations": False,
            "base_control": "articulation joints / differential wheel joints",
            "arm_control": "Lula IK to seven Franka joint targets",
            "grasp_control": "finger closure plus hand-attached PhysX joint",
            "decon_control": "IK raster with live PhysX contact acceptance",
        },
        "operations": records,
        "invariants": invariants,
        "initial": {
            "protected_rate_cps": initial_protected_rate,
            "floor_activity_bq": floor_activity_before,
            "measurement_robot_position_m": initial_robot_position,
            "shield_position_m": initial_shield_position,
            "drum_position_m": initial_drum_position,
            "obstacle_position_m": initial_obstacle_position,
        },
        "final": {
            "protected_rate_cps": final_protected_rate,
            "floor_activity_bq": floor_activity_after,
            "measurement_robot_position_m": final_robot_position,
            "shield_position_m": final_shield_position,
            "obstacle_position_m": final_obstacle_position,
            "drum_secured_in_shielded_storage": drum_secured,
        },
        "transport_statistics": simulation.transport.statistics,
        "radiation_audit": {
            "detector_path": protected,
            "initial": initial_spectrum_components,
            "final": final_spectrum_components,
        },
        "resource_audit": {
            "initial": initial_resource_state,
            "final": _jsonable(final_workflow_view["resources"]),
        },
        "estimation_audit": {
            "initial": initial_estimate,
            "final": _jsonable(final_workflow_view["estimate"]),
            "residual": _jsonable(final_workflow_view["residual"]),
            "initial_solver": initial_estimator_audit,
            "final_solver": _jsonable(public_estimator.last_audit),
        },
        "controller_trace": countermeasure_controller.trace,
        "measurement_controller_trace": measurement_controller.trace,
        "robot_monitor": dashboard.robot_monitor_audit(),
    }


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    from isaacsim import SimulationApp

    app = SimulationApp(
        {
            "headless": args.headless,
            "width": 1600,
            "height": 1000,
            "window_width": 1600,
            "window_height": 1000,
        }
    )
    panel = None
    dashboard = None
    payload: dict[str, Any]
    exit_code = 0
    started = time.perf_counter()
    try:
        from radcounter.isaac.ui.dashboard import RadCounterDashboard

        dashboard = RadCounterDashboard("radcounter.gui.validation")
        panel = _ValidationPanel()
        payload = _run_validation(app, args, panel, dashboard)
        if payload.get("evidence_class") == EvidenceClass.PHYSICAL_ROBOT_EXECUTION.value:
            payload["execution_runtime"] = _physical_execution_runtime()
    except Exception as error:
        exit_code = 1
        preserved = (
            dict(error.audit)
            if isinstance(
                error,
                (ComplexNaturalLanguageValidationError, DecontaminationSmokeValidationError),
            )
            else {}
        )
        payload = {
            **preserved,
            "success": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        print(payload["traceback"], flush=True)
    payload["wall_time_s"] = time.perf_counter() - started
    _atomic_json(args.artifact, payload)
    print(
        json.dumps({"validation_artifact": str(args.artifact), **payload}, default=str), flush=True
    )
    if panel is not None and not args.interactive:
        panel.finish(exit_code == 0, args.artifact)
        total = 7 if args.complex_natural_language_validation else 8
        disposition = "remains open for inspection" if args.keep_open else "will close"
        panel.update(
            f"PASS - GUI {disposition}" if exit_code == 0 else f"FAIL - GUI {disposition}",
            total if exit_code == 0 else 0,
            total,
            payload,
        )
    elif panel is not None and exit_code != 0:
        panel.finish(False, args.artifact)
    try:
        if args.keep_open:
            frame_limiter = _GuiFrameRateLimiter(args.max_fps)
            while app.is_running():
                if dashboard is not None:
                    dashboard.process_pending_natural_language_actions()
                app.update()
                frame_limiter.wait()
    finally:
        if dashboard is not None:
            dashboard.shutdown()
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
