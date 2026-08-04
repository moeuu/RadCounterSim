"""Execute the real robot motion gate in headless Isaac Sim."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    sys.argv = [
        str(ROOT / "scripts/run_real_robot_validation.py"),
        "--headless",
        "--no-capture",
        "--no-keep-open",
        "--artifact",
        str(ROOT / "artifacts/real-robot/gate.json"),
    ]
    runpy.run_path(sys.argv[0], run_name="__main__")
