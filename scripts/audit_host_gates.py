#!/usr/bin/env python3
"""Report concrete host runtime gates without importing or launching Isaac Sim."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Gate:
    available: bool
    evidence: str


def _metadata_version(python: Path, distribution: str) -> str | None:
    if not python.is_file():
        return None
    completed = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.metadata as m; print(m.version(" + repr(distribution) + "))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _isaac_gate() -> Gate:
    root = Path(
        os.environ.get(
            "RADCOUNTER_ISAAC_ROOT",
            Path.home() / ".local/isaacsim/6.0.1-uv",
        )
    )
    python = root / ".venv/bin/python"
    executable = root / ".venv/bin/isaacsim"
    version = _metadata_version(python, "isaacsim")
    available = executable.is_file() and version == "6.0.1.0"
    return Gate(available, f"version={version or 'missing'}, executable={executable}")


def _embree_gate() -> Gate:
    root = Path(
        os.environ.get(
            "RADCOUNTER_EMBREE_ROOT",
            Path.home() / ".local/embree/4.3.0/usr",
        )
    )
    library = root / "lib/x86_64-linux-gnu/libembree4.so.4"
    header = root / "include/embree4/rtcore.h"
    cmake_files = tuple((root / "lib/x86_64-linux-gnu/cmake").glob("embree-*/embree-config.cmake"))
    available = library.is_file() and header.is_file() and bool(cmake_files)
    return Gate(available, f"version=4.3.0, root={root}")


def _ros_gate() -> Gate:
    root = Path(
        os.environ.get(
            "RADCOUNTER_ROS2_ROOT",
            Path.home() / ".local/ros2/jazzy",
        )
    )
    ros2 = Path.home() / ".local/ros2/bin/ros2"
    colcon = shutil.which("colcon") or str(Path.home() / ".local/bin/colcon")
    available = (root / "setup.bash").is_file() and ros2.is_file() and Path(colcon).is_file()
    return Gate(available, f"distro=jazzy-20260618, root={root}, colcon={colcon}")


def _gpu_gate() -> Gate:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return Gate(False, "nvidia-smi not found")
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return Gate(completed.returncode == 0, completed.stdout.strip() or executable)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="return nonzero unless every installed-runtime gate is available",
    )
    args = parser.parse_args()
    gates = {
        "isaac_sim_6_0_1": _isaac_gate(),
        "embree_4": _embree_gate(),
        "ros_2_jazzy": _ros_gate(),
        "nvidia_gpu": _gpu_gate(),
    }
    payload = {
        "all_available": all(gate.available for gate in gates.values()),
        "gates": {name: asdict(gate) for name, gate in gates.items()},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_all and not payload["all_available"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
