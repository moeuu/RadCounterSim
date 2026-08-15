#!/usr/bin/env python3
"""Launch the complete articulated RadCounterSim GUI workflow."""

from __future__ import annotations

import sys
from pathlib import Path

from run_gui_validation import main

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    arguments = [
        "--interactive",
        "--artifact",
        str(root / "artifacts/app/latest.json"),
        *sys.argv[1:],
    ]
    raise SystemExit(main(arguments))
