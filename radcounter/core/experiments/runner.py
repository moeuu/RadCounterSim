"""Seed/planner batch execution with required artifact output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from radcounter.core.experiments.artifacts import CaseResult, RunArtifactWriter, RunDirectory


class ExperimentCase(Protocol):
    case_id: str

    def run(self, *, seed: int, planner_id: str) -> CaseResult: ...


@dataclass(frozen=True)
class BatchRun:
    case_id: str
    planner_id: str
    seed: int
    artifacts: RunDirectory


class BatchRunner:
    """Run the same case across seeds/planners with isolated directories."""

    def __init__(self, repository_root: str | Path, run_root: str | Path) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.run_root = Path(run_root)

    def run(
        self,
        cases: Sequence[ExperimentCase],
        *,
        seeds: Sequence[int],
        planner_ids: Sequence[str],
        common_config: Mapping[str, object] | None = None,
    ) -> tuple[BatchRun, ...]:
        """Execute a deterministic Cartesian batch."""

        outputs: list[BatchRun] = []
        common = {} if common_config is None else dict(common_config)
        for case in cases:
            for planner_id in planner_ids:
                for seed in seeds:
                    run_id = f"{case.case_id}-{planner_id}-seed-{seed}"
                    writer = RunArtifactWriter(
                        run_root=self.run_root,
                        scenario_id=case.case_id,
                        run_id=run_id,
                        repository_root=self.repository_root,
                        resolved_config={
                            **common,
                            "case_id": case.case_id,
                            "planner_id": planner_id,
                            "seed": seed,
                        },
                        run_seed=seed,
                        timestamp=datetime.now(UTC),
                    )
                    result = case.run(seed=seed, planner_id=planner_id)
                    writer.events.log("scene_loaded", sim_time_s=0.0, details={})
                    for event in result.events:
                        writer.events.log(
                            str(event.get("event_type", "event")),
                            sim_time_s=float(event.get("sim_time_s", 0.0)),
                            details={
                                key: value
                                for key, value in event.items()
                                if key not in {"event_type", "sim_time_s"}
                            },
                        )
                    for name, records in result.tables.items():
                        writer.write_table(name, records)
                    for name, arrays in result.maps.items():
                        writer.write_map(name, **arrays)
                    writer.events.log("episode_completed", sim_time_s=0.0, details={})
                    paths = writer.finalize(result.metrics)
                    outputs.append(BatchRun(case.case_id, planner_id, seed, paths))
        return tuple(outputs)
