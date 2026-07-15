"""Run manifest collection without hard dependency on external runtimes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any


def _git_value(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, check=True, capture_output=True, text=True, timeout=2
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def build_manifest(
    *, repository_root: str | Path, config_bytes: bytes, run_seed: int
) -> dict[str, Any]:
    """Collect reproducibility metadata for one run."""

    root = Path(repository_root).resolve()
    packages: dict[str, str] = {}
    for name in ("numpy", "scipy", "pydantic", "pyyaml", "pandas", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    status = _git_value(root, "status", "--porcelain")
    return {
        "git_commit_sha": _git_value(root, "rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "run_seed": run_seed,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "isaac_sim_version": None,
        "embree_version": None,
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Write a deterministic, human-readable manifest."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
