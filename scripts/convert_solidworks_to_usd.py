#!/usr/bin/env python3
"""Convert a native SolidWorks assembly to USD with Isaac Sim's Linux HOOPS backend."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT / ".cache/external/fukushima_daiichi_solidworks/source/Building.SLDASM"
)
DEFAULT_OUTPUT = (
    ROOT / ".cache/external/fukushima_daiichi_solidworks/export/Building.usdc"
)
DEFAULT_CONFIG = ROOT / "configs/environment-converters/solidworks_to_usd.hoops.json"


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert .SLDASM/.SLDPRT to a monolithic USD file using the Linux "
            "HOOPS Exchange converter bundled with Isaac Sim"
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--isaac-root", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version_key(path: Path) -> tuple[int, ...]:
    numbers = re.findall(r"\d+", path.name.split("-standalone", maxsplit=1)[0])
    return tuple(int(value) for value in numbers)


def discover_isaac_root(explicit: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    environment = os.environ.get("RADCOUNTER_ISAAC_ROOT")
    if environment:
        candidates.append(Path(environment))
    candidates.extend(Path.home().glob(".local/isaacsim/*-standalone"))
    valid = [path.expanduser().resolve() for path in candidates if (path / "kit/kit").is_file()]
    if not valid:
        raise FileNotFoundError(
            "Isaac Sim standalone was not found; set RADCOUNTER_ISAAC_ROOT or pass --isaac-root"
        )
    return max(valid, key=_version_key)


def _newest_extension(root: Path, prefix: str) -> Path:
    matches = sorted((root / "extscache").glob(f"{prefix}-*"))
    if not matches:
        raise FileNotFoundError(f"Isaac Sim extension is missing: {prefix}")
    return matches[-1]


def _conversion_command(
    isaac_root: Path,
    *,
    source: Path,
    output: Path,
    config: Path,
) -> tuple[list[str], str, str]:
    hoops = _newest_extension(isaac_root, "omni.kit.converter.hoops")
    hoops_core = _newest_extension(isaac_root, "omni.kit.converter.hoops_core")
    launch_script = hoops / "omni/kit/converter/hoops/process/launch_hoops_app.py"
    if not launch_script.is_file():
        raise FileNotFoundError(f"HOOPS launch script is missing: {launch_script}")
    script_arguments = " ".join(
        (
            shlex.quote(str(launch_script)),
            "--config-path",
            shlex.quote(str(config)),
            "--input-path",
            shlex.quote(str(source)),
            "--output-path",
            shlex.quote(str(output)),
        )
    )
    command = [
        str(isaac_root / "kit/kit"),
        str(isaac_root / "kit/apps/omni.app.empty.kit"),
        "--ext-folder",
        str(isaac_root / "extscache"),
        "--ext-folder",
        str(isaac_root / "kit/extscore"),
        "--ext-folder",
        str(isaac_root / "kit/exts"),
        "--enable",
        "omni.kit.converter.hoops_core",
        "--exec",
        script_arguments,
        "--no-window",
        "--/app/fastShutdown=1",
        "--/persistent/app/usd/muteUsdDiagnostics=false",
    ]
    return command, hoops.name, hoops_core.name


def convert(args: argparse.Namespace) -> dict[str, object]:
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    config = args.config.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"SolidWorks source is missing: {source}. Run scripts/fetch_fukushima_daiichi_cad.py"
        )
    if source.suffix.lower() not in {".sldasm", ".sldprt"}:
        raise ValueError(f"expected .SLDASM or .SLDPRT input: {source}")
    if not config.is_file():
        raise FileNotFoundError(f"HOOPS converter config is missing: {config}")
    if output.exists() and not args.force:
        raise FileExistsError(f"output already exists (pass --force to replace it): {output}")

    isaac_root = discover_isaac_root(args.isaac_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".radcounter-cad-", dir=output.parent) as raw_temp:
        temporary = Path(raw_temp) / output.name
        command, hoops_extension, hoops_core_extension = _conversion_command(
            isaac_root,
            source=source,
            output=temporary,
            config=config,
        )
        process = subprocess.run(
            command,
            cwd=source.parent,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=900,
        )
        log = process.stdout
        success_marker = "[omni.converter.hoops_progress]*end*0*"
        if process.returncode != 0 or success_marker not in log or not temporary.is_file():
            tail = "\n".join(log.splitlines()[-80:])
            raise RuntimeError(
                f"HOOPS conversion failed with process code {process.returncode}:\n{tail}"
            )
        if temporary.stat().st_size < 1024:
            raise RuntimeError("HOOPS conversion produced an implausibly small USD file")
        temporary.replace(output)

    mesh_match = re.search(r"Total Meshes in USD = (\d+)", log)
    triangle_match = re.search(r"Total Triangles in USD = (\d+)", log)
    converter_match = re.search(r"HOOPS Converter version: ([^\r\n]+)", log)
    revision = None
    with contextlib.suppress(OSError, subprocess.CalledProcessError):
        revision = subprocess.run(
            ["git", "-C", str(source.parent), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    provenance = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source": str(source),
        "source_sha256": _sha256(source),
        "source_repository_revision": revision,
        "output": str(output),
        "output_sha256": _sha256(output),
        "output_bytes": output.stat().st_size,
        "mesh_count": int(mesh_match.group(1)) if mesh_match else None,
        "triangle_count": int(triangle_match.group(1)) if triangle_match else None,
        "converter": "NVIDIA Omniverse HOOPS Exchange",
        "converter_version": converter_match.group(1).strip() if converter_match else None,
        "hoops_extension": hoops_extension,
        "hoops_core_extension": hoops_core_extension,
        "isaac_root": str(isaac_root),
        "config": str(config),
        "config_sha256": _sha256(config),
    }
    provenance_path = output.with_suffix(output.suffix + ".provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**provenance, "provenance": str(provenance_path)}


def main(argv: list[str] | None = None) -> int:
    try:
        payload = convert(_arguments(argv))
    except Exception as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
