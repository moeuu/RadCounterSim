"""Single-entry launcher for a user-owned Isaac Sim installation."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    repository_root = Path(
        os.environ.get("RADCOUNTER_APP_ROOT", Path(__file__).resolve().parents[1])
    ).resolve()
    launcher = repository_root / "scripts/run_app.sh"
    if not launcher.is_file():
        raise FileNotFoundError(
            f"RadInterAct application launcher is missing: {launcher}; "
            "set RADCOUNTER_APP_ROOT to the release directory"
        )
    os.execvpe(
        "bash",
        ["bash", str(launcher), *sys.argv[1:]],
        {**os.environ, "RADCOUNTER_APP_ROOT": str(repository_root)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
