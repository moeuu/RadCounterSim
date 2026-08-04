from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from radcounter.core.experiments import ExperimentArtifactBundle, ExperimentCaseRecord

ROOT = Path(__file__).resolve().parents[2]


def test_vertical_slice_activity_map_and_config_remain_self_consistent() -> None:
    activity_path = ROOT / "assets/contaminated_objects/floor_activity.npz"
    stage = (ROOT / "assets/environments/radcounter_vertical_slice.usda").read_text(
        encoding="utf-8"
    )
    config = json.loads(
        (ROOT / "configs/scenarios/vertical_slice.runtime.json").read_text(encoding="utf-8")
    )
    digest = hashlib.sha256(activity_path.read_bytes()).hexdigest()
    assert digest in stage
    assert config["physics_data_class"] == "synthetic_validation_only"
    with np.load(activity_path, allow_pickle=False) as payload:
        assert np.array_equal(payload["triangle_indices"], np.array([0, 1]))
        assert np.all(payload["activity_bq"] > 0)
        assert np.all(payload["cumulative_treatment_exposure"] == 0)
    required_roles = {
        'rad:role = "contaminated_surface"',
        'rad:role = "source"',
        'rad:role = "shield"',
        'rad:role = "measurement_robot"',
        'rad:role = "countermeasure_robot"',
        'rad:role = "disposal_zone"',
    }
    assert all(role in stage for role in required_roles)


def test_experiment_bundle_is_atomic_and_checksums_inputs(tmp_path: Path) -> None:
    stage = tmp_path / "scene.usda"
    config = tmp_path / "scenario.json"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    bundle = ExperimentArtifactBundle(
        tmp_path / "out", "run-1", stage_path=stage, config_path=config
    )
    bundle.record(
        ExperimentCaseRecord(
            case_id="seed-00000",
            seed=0,
            metrics={"residual_cps": 1.5},
            parameters={"pose_error_m": 0.02},
        )
    )
    summary = bundle.finalize()
    manifest = json.loads((summary.parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["case_count"] == 1
    assert manifest["stage"]["sha256"] == hashlib.sha256(stage.read_bytes()).hexdigest()
    assert manifest["summary_sha256"] == hashlib.sha256(summary.read_bytes()).hexdigest()
    assert not tuple(summary.parent.glob(".*.tmp"))
