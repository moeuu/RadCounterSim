"""Atomic, self-describing experiment output bundles."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


class EvidenceClass(StrEnum):
    """What kind of evidence an experiment record is allowed to support."""

    KINEMATIC_SCENE_EDIT = "kinematic_scene_edit"
    PHYSICAL_ROBOT_EXECUTION = "physical_robot_execution"
    ANALYTIC_VALIDATION = "analytic_validation"
    INDEPENDENT_TRANSPORT_REFERENCE = "independent_transport_reference"
    CONTROLLED_MEASUREMENT = "controlled_measurement"


@dataclass(frozen=True, slots=True)
class ExperimentCaseRecord:
    case_id: str
    seed: int
    evidence_class: EvidenceClass
    metrics: Mapping[str, float | int | str | bool]
    parameters: Mapping[str, float | int | str | bool]

    def __post_init__(self) -> None:
        if not self.case_id or self.seed < 0:
            raise ValueError("experiment case requires a nonempty ID and nonnegative seed")
        object.__setattr__(self, "evidence_class", EvidenceClass(self.evidence_class))


class ExperimentArtifactBundle:
    """Write immutable case records plus a machine-readable provenance manifest."""

    def __init__(
        self,
        output_root: str | Path,
        run_id: str,
        *,
        stage_path: str | Path,
        config_path: str | Path,
        evidence_class: EvidenceClass,
        metadata: Mapping[str, object] | None = None,
        execution_runtime: Mapping[str, object] | None = None,
    ) -> None:
        self.run_id = run_id
        self.path = Path(output_root) / run_id
        self.cases_path = self.path / "cases"
        self.cases_path.mkdir(parents=True, exist_ok=False)
        self._records: list[ExperimentCaseRecord] = []
        self.evidence_class = EvidenceClass(evidence_class)
        stage = Path(stage_path)
        config = Path(config_path)
        self.manifest = {
            "schema_version": 2,
            "run_id": run_id,
            "evidence_class": self.evidence_class.value,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "stage": {"path": str(stage), "sha256": sha256_file(stage)},
            "configuration": {"path": str(config), "sha256": sha256_file(config)},
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "numpy": _package_version("numpy"),
                "isaacsim": _package_version("isaacsim"),
            },
            "metadata": dict(metadata or {}),
        }
        if execution_runtime is not None:
            self.manifest["execution_runtime"] = dict(execution_runtime)
        self._atomic_json(self.path / "manifest.json", self.manifest)

    @staticmethod
    def _atomic_json(path: Path, payload: object) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, path)
        finally:
            Path(temporary_name).unlink(missing_ok=True)

    def record(self, record: ExperimentCaseRecord) -> Path:
        if record.evidence_class is not self.evidence_class:
            raise ValueError(
                f"case evidence_class={record.evidence_class.value!r} does not match "
                f"bundle evidence_class={self.evidence_class.value!r}"
            )
        if any(existing.case_id == record.case_id for existing in self._records):
            raise ValueError(f"duplicate experiment case_id: {record.case_id}")
        path = self.cases_path / f"{record.case_id}.json"
        self._atomic_json(path, asdict(record))
        self._records.append(record)
        return path

    def finalize(self) -> Path:
        metric_names = sorted({name for record in self._records for name in record.metrics})
        parameter_names = sorted({name for record in self._records for name in record.parameters})
        summary = self.path / "summary.csv"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".summary.", suffix=".csv", dir=self.path
        )
        try:
            with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
                fieldnames = [
                    "case_id",
                    "seed",
                    *[f"parameter:{name}" for name in parameter_names],
                    *metric_names,
                ]
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                for record in self._records:
                    row = {"case_id": record.case_id, "seed": record.seed, **record.metrics}
                    row.update(
                        {
                            f"parameter:{name}": record.parameters.get(name, "")
                            for name in parameter_names
                        }
                    )
                    writer.writerow(row)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, summary)
        finally:
            Path(temporary_name).unlink(missing_ok=True)
        self.manifest["case_count"] = len(self._records)
        self.manifest["summary_sha256"] = sha256_file(summary)
        self._atomic_json(self.path / "manifest.json", self.manifest)
        return summary
