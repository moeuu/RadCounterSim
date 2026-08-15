#!/usr/bin/env python3
"""Fetch the pinned CC BY 4.0 Fukushima Daiichi SolidWorks source."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE = "https://github.com/Qualot/fukushima_daiichi_solidworks.git"
DEFAULT_REVISION = "f6541deb6159c5d908a4f028d021e3d2c9f7f8e8"
DEFAULT_DESTINATION = ROOT / ".cache/external/fukushima_daiichi_solidworks/source"


def _run(arguments: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        arguments,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the pinned Fukushima Daiichi SolidWorks environment source"
    )
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args(argv)
    if shutil.which("git") is None:
        raise RuntimeError("git is required to fetch the Fukushima CAD source")

    destination = args.destination.expanduser().resolve()
    if destination.exists():
        if not (destination / ".git").is_dir():
            raise FileExistsError(
                f"destination exists but is not a git checkout: {destination}"
            )
        dirty = _run(["git", "status", "--porcelain"], cwd=destination)
        if dirty:
            raise RuntimeError(
                f"refusing to change a modified Fukushima CAD checkout: {destination}"
            )
        _run(["git", "fetch", "--depth", "1", "origin", args.revision], cwd=destination)
        _run(["git", "switch", "--detach", args.revision], cwd=destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                args.remote,
                str(destination),
            ]
        )
        _run(["git", "fetch", "--depth", "1", "origin", args.revision], cwd=destination)
        _run(["git", "checkout", "--detach", args.revision], cwd=destination)

    actual_revision = _run(["git", "rev-parse", "HEAD"], cwd=destination)
    if actual_revision != args.revision:
        raise RuntimeError(
            f"unexpected Fukushima CAD revision: expected {args.revision}, got {actual_revision}"
        )
    assembly = destination / "Building.SLDASM"
    license_path = destination / "LICENSE"
    if not assembly.is_file() or not license_path.is_file():
        raise RuntimeError("checkout lacks Building.SLDASM or LICENSE")

    provenance = {
        "schema_version": 1,
        "source_repository": args.remote,
        "source_revision": actual_revision,
        "source_license": "CC BY 4.0",
        "source_author_attribution": "Qualot/fukushima_daiichi_solidworks contributors",
        "top_level_assembly": str(assembly),
        "top_level_assembly_sha256": _sha256(assembly),
        "license_sha256": _sha256(license_path),
        "redistribution_note": (
            "Preserve upstream attribution and the CC BY 4.0 license when sharing exports."
        ),
    }
    manifest = destination.parent / "source-provenance.json"
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest)
    print(json.dumps({"source": str(destination), "provenance": str(manifest), **provenance}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
