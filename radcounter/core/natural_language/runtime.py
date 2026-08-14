"""Lifecycle management for a bundled llama.cpp server and GGUF model."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TextIO
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen


class RuntimeAssetMissingError(RuntimeError):
    """Raised when the packaged server binary or model is unavailable."""


class RuntimeStatus(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    EXTERNAL = "external"


@dataclass(frozen=True, slots=True)
class NvidiaGpu:
    name: str
    memory_mib: int


@dataclass(frozen=True, slots=True)
class LlamaCppRuntimeConfig:
    """Product-owned inference configuration with environment overrides."""

    runtime_directory: Path
    server_binary: Path | None = None
    model_path: Path | None = None
    pid_path: Path | None = None
    external_endpoint: str | None = None
    model_alias: str = "radcounter-qwen3-4b"
    context_size: int = 8192
    startup_timeout_s: float = 180.0
    request_parallelism: int = 1
    gpu_mode: str = "auto"

    @classmethod
    def default(cls, repository_root: str | Path | None = None) -> LlamaCppRuntimeConfig:
        root = (
            Path(repository_root).resolve()
            if repository_root is not None
            else Path(__file__).resolve().parents[3]
        )
        runtime = Path(
            os.environ.get("RADCOUNTER_LLM_RUNTIME_DIR", root / "runtime/llm")
        ).expanduser()
        binary = os.environ.get("RADCOUNTER_LLAMA_SERVER")
        model = os.environ.get("RADCOUNTER_LLM_MODEL")
        pid_path = os.environ.get("RADCOUNTER_LLM_PID_FILE")
        return cls(
            runtime_directory=runtime,
            server_binary=None if not binary else Path(binary).expanduser(),
            model_path=None if not model else Path(model).expanduser(),
            pid_path=None if not pid_path else Path(pid_path).expanduser(),
            external_endpoint=os.environ.get("RADCOUNTER_LLM_ENDPOINT"),
            gpu_mode=os.environ.get("RADCOUNTER_LLM_GPU_MODE", "auto"),
        )


def _detect_nvidia_gpu() -> NvidiaGpu | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return None
        name, memory = completed.stdout.splitlines()[0].rsplit(",", 1)
        return NvidiaGpu(name.strip(), int(memory.strip()))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class LlamaCppRuntime:
    """Start, monitor, and stop the product-bundled llama.cpp sidecar."""

    def __init__(self, config: LlamaCppRuntimeConfig | None = None) -> None:
        self.config = config or LlamaCppRuntimeConfig.default()
        self.status = RuntimeStatus.STOPPED
        self.endpoint: str | None = None
        self.process: subprocess.Popen[str] | None = None
        self.gpu = _detect_nvidia_gpu()
        self._log_stream: TextIO | None = None
        self._lock = threading.RLock()

    def _binary_candidates(self) -> tuple[Path, ...]:
        executable = "llama-server.exe" if os.name == "nt" else "llama-server"
        platform_name = "windows-x86_64" if os.name == "nt" else "linux-x86_64"
        configured = () if self.config.server_binary is None else (self.config.server_binary,)
        system = shutil.which("llama-server")
        packaged = (
            (
                self.config.runtime_directory / "bin" / f"{platform_name}-cpu" / executable,
                self.config.runtime_directory
                / "bin"
                / f"{platform_name}-vulkan"
                / executable,
            )
            if self.config.gpu_mode == "cpu"
            else
            (
                self.config.runtime_directory / "bin" / f"{platform_name}-cuda" / executable,
                self.config.runtime_directory
                / "bin"
                / f"{platform_name}-vulkan"
                / executable,
                self.config.runtime_directory / "bin" / f"{platform_name}-cpu" / executable,
            )
            if self.gpu is not None
            else (
                self.config.runtime_directory / "bin" / f"{platform_name}-cpu" / executable,
                self.config.runtime_directory
                / "bin"
                / f"{platform_name}-vulkan"
                / executable,
            )
        )
        return (
            *configured,
            *packaged,
            self.config.runtime_directory / "bin" / platform_name / executable,
            self.config.runtime_directory / "bin" / executable,
            *((Path(system),) if system else ()),
        )

    def resolve_binary(self) -> Path:
        for candidate in self._binary_candidates():
            path = candidate.expanduser().resolve()
            if path.is_file() and os.access(path, os.X_OK):
                return path
        raise RuntimeAssetMissingError(
            "llama-server is not installed in runtime/llm; run scripts/build_llama_runtime.sh"
        )

    def resolve_model(self) -> Path:
        candidates = (
            *((self.config.model_path,) if self.config.model_path is not None else ()),
            self.config.runtime_directory / "models" / "Qwen3-4B-Q4_K_M.gguf",
        )
        for candidate in candidates:
            path = candidate.expanduser().resolve()
            if path.is_file():
                return path
        raise RuntimeAssetMissingError(
            "Qwen3-4B GGUF is not installed in runtime/llm; run scripts/fetch_llm_model.py"
        )

    def _health(self, endpoint: str) -> bool:
        try:
            with urlopen(f"{endpoint}/health", timeout=2.0) as response:  # noqa: S310
                if response.status != 200:
                    return False
                payload = json.loads(response.read().decode("utf-8"))
                return payload.get("status") in {"ok", "no slot available"}
        except (OSError, URLError, json.JSONDecodeError):
            return False

    def _command(self, binary: Path, model: Path, port: int) -> list[str]:
        threads = max(2, min(16, (os.cpu_count() or 4) // 2))
        command = [
            str(binary),
            "--model",
            str(model),
            "--alias",
            self.config.model_alias,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx-size",
            str(self.config.context_size),
            "--parallel",
            str(self.config.request_parallelism),
            "--threads",
            str(threads),
            "--jinja",
            "--reasoning",
            "off",
            "--n-gpu-layers",
        ]
        if self.config.gpu_mode == "cpu" or self.gpu is None:
            command.append("0")
        elif self.config.gpu_mode in {"auto", "hybrid"}:
            command.append("auto")
        elif self.config.gpu_mode == "gpu":
            command.append("all")
        else:
            raise ValueError("gpu_mode must be auto, hybrid, gpu, or cpu")
        return command

    def _write_pid_file(self, pid: int) -> None:
        path = self.config.pid_path
        if path is None:
            return
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(f"{pid}\n", encoding="ascii")
        temporary.replace(path)

    def _remove_pid_file(self, pid: int | None) -> None:
        path = self.config.pid_path
        if path is None:
            return
        path = path.expanduser().resolve()
        try:
            recorded = int(path.read_text(encoding="ascii").strip())
        except (FileNotFoundError, OSError, ValueError):
            return
        if pid is None or recorded == pid:
            path.unlink(missing_ok=True)

    def start(self) -> str:
        with self._lock:
            if self.status in {RuntimeStatus.READY, RuntimeStatus.EXTERNAL}:
                assert self.endpoint is not None
                return self.endpoint
            if self.config.external_endpoint:
                endpoint = self.config.external_endpoint.rstrip("/")
                parsed = urlparse(endpoint)
                if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
                    raise ValueError("the configured model endpoint must be loopback-only")
                if not self._health(endpoint):
                    raise RuntimeError(
                        f"configured local model endpoint is unavailable: {endpoint}"
                    )
                self.endpoint = endpoint
                self.status = RuntimeStatus.EXTERNAL
                return endpoint
            binary = self.resolve_binary()
            model = self.resolve_model()
            port = _free_loopback_port()
            endpoint = f"http://127.0.0.1:{port}"
            log_directory = self.config.runtime_directory / "logs"
            log_directory.mkdir(parents=True, exist_ok=True)
            self._log_stream = (log_directory / "llama-server.log").open(
                "a", encoding="utf-8"
            )
            self.status = RuntimeStatus.STARTING
            try:
                self.process = subprocess.Popen(
                    self._command(binary, model, port),
                    stdin=subprocess.DEVNULL,
                    stdout=self._log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                self._write_pid_file(self.process.pid)
            except OSError:
                self.status = RuntimeStatus.FAILED
                failed_process = self.process
                self.process = None
                if failed_process is not None and failed_process.poll() is None:
                    failed_process.terminate()
                    failed_process.wait(timeout=8.0)
                self._remove_pid_file(
                    None if failed_process is None else failed_process.pid
                )
                self._log_stream.close()
                self._log_stream = None
                raise
            deadline = time.monotonic() + self.config.startup_timeout_s
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    self.status = RuntimeStatus.FAILED
                    stopped_pid = self.process.pid
                    self.process = None
                    self._remove_pid_file(stopped_pid)
                    if self._log_stream is not None:
                        self._log_stream.close()
                        self._log_stream = None
                    raise RuntimeError(
                        "llama-server stopped during startup; inspect "
                        "runtime/llm/logs/llama-server.log"
                    )
                if self._health(endpoint):
                    self.endpoint = endpoint
                    self.status = RuntimeStatus.READY
                    return endpoint
                time.sleep(0.2)
            self.stop()
            self.status = RuntimeStatus.FAILED
            raise TimeoutError("llama-server did not become healthy before the startup timeout")

    def stop(self) -> None:
        with self._lock:
            process = self.process
            self.process = None
            self.endpoint = None
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3.0)
            self._remove_pid_file(None if process is None else process.pid)
            if self._log_stream is not None:
                self._log_stream.close()
                self._log_stream = None
            self.status = RuntimeStatus.STOPPED

    def describe(self) -> str:
        if self.gpu is None:
            device = "CPU fallback"
        else:
            device = f"{self.gpu.name} ({self.gpu.memory_mib} MiB), {self.config.gpu_mode}"
        return f"llama.cpp {self.status.value} · {device}"
