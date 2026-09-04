from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY_BRAND = "".join(("Rad", "Counter", "Sim"))
LEGACY_TOKENS = (LEGACY_BRAND, LEGACY_BRAND.lower())


def test_distribution_uses_current_project_name() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]

    assert project["name"] == "radinteract"


def test_tracked_files_do_not_restore_legacy_project_name() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")

    for raw_path in tracked:
        if not raw_path:
            continue
        relative_path = Path(raw_path.decode())
        relative_text = relative_path.as_posix()
        assert all(token not in relative_text for token in LEGACY_TOKENS)

        path = ROOT / relative_path
        if not path.is_file():
            continue
        try:
            contents = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        assert all(token not in contents for token in LEGACY_TOKENS), relative_text
