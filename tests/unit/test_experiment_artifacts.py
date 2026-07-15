import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from radcounter.core.experiments import (
    AnalyticRadiationValidationCase,
    BatchRunner,
    RunArtifactWriter,
)


def test_run_writer_creates_required_artifacts(tmp_path) -> None:
    writer = RunArtifactWriter(
        run_root=tmp_path,
        scenario_id="demo",
        run_id="run-1",
        repository_root=Path(__file__).resolve().parents[2],
        resolved_config={"seed": 7},
        run_seed=7,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )
    writer.write_table("measurements", [{"measurement_id": "m1", "counts": [1, 2]}])
    writer.write_map("dose", points=np.zeros((1, 3)), dose_rate=np.ones(1))
    paths = writer.finalize({"score": 1.5})
    required = {
        "manifest.json",
        "resolved_config.yaml",
        "events.jsonl",
        "measurements.parquet",
        "estimates.parquet",
        "actions.parquet",
        "resources.parquet",
        "metrics.json",
        "report.html",
    }
    assert all((paths.root / name).is_file() for name in required)
    assert paths.maps.is_dir() and paths.snapshots.is_dir()
    frame = pd.read_parquet(paths.root / "measurements.parquet")
    assert frame.loc[0, "measurement_id"] == "m1"
    assert json.loads(paths.metrics.read_text())["score"] == 1.5


def test_analytic_batch_case_meets_validation_thresholds(tmp_path) -> None:
    root = Path(__file__).resolve().parents[2]
    runs = BatchRunner(root, tmp_path).run(
        [AnalyticRadiationValidationCase()],
        seeds=[2, 3],
        planner_ids=["closed_loop_residual"],
    )
    assert len(runs) == 2
    for run in runs:
        metrics = json.loads(run.artifacts.metrics.read_text())
        assert metrics["inverse_square_relative_error"] < 1e-12
        assert metrics["slab_relative_error"] < 1e-12
        assert (run.artifacts.root / "report.html").stat().st_size > 100
