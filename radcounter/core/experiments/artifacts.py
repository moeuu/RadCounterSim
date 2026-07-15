"""Specification-compliant run directories and artifact serialization."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from radcounter.core.logging import JsonlEventLogger, build_manifest, write_manifest

REQUIRED_TABLES = ("measurements", "estimates", "actions", "resources")


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def asset_sha256(paths: Sequence[str | Path]) -> dict[str, str]:
    """Hash every declared asset using resolved absolute paths."""

    result: dict[str, str] = {}
    for value in paths:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"asset does not exist: {path}")
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


@dataclass(frozen=True)
class RunDirectory:
    root: Path
    manifest: Path
    resolved_config: Path
    events: Path
    metrics: Path
    report: Path
    maps: Path
    snapshots: Path


class RunArtifactWriter:
    """Write all required reproducibility artifacts for one run."""

    def __init__(
        self,
        *,
        run_root: str | Path,
        scenario_id: str,
        run_id: str,
        repository_root: str | Path,
        resolved_config: Mapping[str, Any],
        run_seed: int,
        asset_paths: Sequence[str | Path] = (),
        timestamp: datetime | None = None,
    ) -> None:
        instant = timestamp if timestamp is not None else datetime.now(UTC)
        directory_name = f"{instant.strftime('%Y%m%dT%H%M%SZ')}_{run_id}"
        root = Path(run_root) / scenario_id / directory_name
        root.mkdir(parents=True, exist_ok=False)
        maps = root / "maps"
        snapshots = root / "snapshots"
        maps.mkdir()
        snapshots.mkdir()
        self.paths = RunDirectory(
            root,
            root / "manifest.json",
            root / "resolved_config.yaml",
            root / "events.jsonl",
            root / "metrics.json",
            root / "report.html",
            maps,
            snapshots,
        )
        self.events = JsonlEventLogger(self.paths.events)
        self.paths.events.touch(exist_ok=True)
        config_serializable = _json_value(resolved_config)
        config_bytes = yaml.safe_dump(config_serializable, sort_keys=True).encode("utf-8")
        self.paths.resolved_config.write_bytes(config_bytes)
        manifest = build_manifest(
            repository_root=repository_root,
            config_bytes=config_bytes,
            run_seed=run_seed,
        )
        manifest.update(
            {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "timestamp_utc": instant.isoformat(),
                "cpu": platform.processor() or platform.machine(),
                "gpu": None,
                "asset_sha256": asset_sha256(asset_paths),
            }
        )
        write_manifest(self.paths.manifest, manifest)
        self._tables_written: set[str] = set()

    def write_table(self, name: str, records: Sequence[Mapping[str, Any]]) -> Path:
        """Write one Parquet table with JSON-normalized nested values."""

        if name not in REQUIRED_TABLES:
            raise ValueError(f"unsupported required table name: {name}")
        normalized = [
            {
                key: (
                    json.dumps(_json_value(value), sort_keys=True)
                    if isinstance(value, (dict, list, tuple, np.ndarray))
                    else _json_value(value)
                )
                for key, value in record.items()
            }
            for record in records
        ]
        frame = pd.DataFrame(normalized)
        if frame.empty:
            frame = pd.DataFrame({"empty": pd.Series(dtype="bool")})
        path = self.paths.root / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        self._tables_written.add(name)
        return path

    def write_map(self, name: str, **arrays: np.ndarray) -> Path:
        """Write a compressed NPZ map/snapshot array bundle."""

        if not name or not arrays:
            raise ValueError("map name and arrays are required")
        path = self.paths.maps / f"{name}.npz"
        np.savez_compressed(path, **arrays)
        return path

    def finalize(self, metrics: Mapping[str, Any]) -> RunDirectory:
        """Ensure required tables exist, write metrics, and render HTML."""

        for name in REQUIRED_TABLES:
            if name not in self._tables_written:
                self.write_table(name, ())
        serializable_metrics = _json_value(metrics)
        self.paths.metrics.write_text(
            json.dumps(serializable_metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        from radcounter.core.experiments.report import render_run_report

        render_run_report(self.paths.root, self.paths.report)
        return self.paths


@dataclass(frozen=True)
class CaseResult:
    """In-memory result returned by an experiment case."""

    metrics: Mapping[str, Any]
    tables: Mapping[str, Sequence[Mapping[str, Any]]]
    maps: Mapping[str, Mapping[str, np.ndarray]]
    events: Sequence[Mapping[str, Any]]

    def as_json(self) -> dict[str, Any]:
        return _json_value(asdict(self))
