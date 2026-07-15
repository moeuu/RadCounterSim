#!/usr/bin/env python3
"""Write the installed host and project environment manifest."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def _output(command: list[str]) -> str | None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return completed.stdout.strip() if completed.returncode == 0 else None


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    home = Path.home()
    isaac_root = home / ".local/isaacsim/6.0.1-uv"
    isaac_python = isaac_root / ".venv/bin/python"
    isaac_version = _output(
        [
            str(isaac_python),
            "-c",
            "import importlib.metadata as m; print(m.version('isaacsim'))",
        ]
    )
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "host": {
            "os": platform.platform(),
            "architecture": platform.machine(),
            "cpu": platform.processor(),
            "gpu": _output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
        },
        "runtime": {
            "python_environment_manager": _output(["uv", "--version"]),
            "project_python": platform.python_version(),
            "isaac_sim": {
                "version": isaac_version,
                "root": str(isaac_root),
                "lock": str(isaac_root / "uv.lock"),
            },
            "embree": {
                "version": "4.3.0",
                "root": str(home / ".local/embree/4.3.0/usr"),
            },
            "ros_2": {
                "distribution": "jazzy",
                "release": "20260618",
                "root": str(home / ".local/ros2/jazzy"),
                "python_lock": str(home / ".local/ros2/python-runtime/uv.lock"),
            },
        },
    }
    output = repository_root / "environment_manifest.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
