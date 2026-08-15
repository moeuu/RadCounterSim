#!/usr/bin/env python3
"""Install a per-user desktop entry for the RadCounterSim launcher."""

from __future__ import annotations

import os
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    launcher = root / "scripts/run_app.sh"
    applications = Path.home() / ".local/share/applications"
    applications.mkdir(parents=True, exist_ok=True)
    desktop = applications / "radcountersim.desktop"
    desktop.write_text(
        "\n".join(
            (
                "[Desktop Entry]",
                "Type=Application",
                "Version=1.0",
                "Name=RadCounterSim",
                "Comment=Radiation measurement and countermeasure simulation",
                f"Exec={launcher}",
                f"Path={root}",
                "Terminal=false",
                "Icon=applications-science",
                "Categories=Science;Education;",
                "StartupNotify=true",
                "",
            )
        ),
        encoding="utf-8",
    )
    os.chmod(desktop, 0o755)
    print(f"Installed desktop entry: {desktop}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
