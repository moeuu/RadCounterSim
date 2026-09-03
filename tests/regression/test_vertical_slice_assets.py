from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from radcounter.core.experiments import (
    EvidenceClass,
    ExperimentArtifactBundle,
    ExperimentCaseRecord,
)

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
    treatment_path = ROOT / "configs/decontamination/concrete_surface.synthetic.yaml"
    treatment_digest = hashlib.sha256(treatment_path.read_bytes()).hexdigest()
    assert digest in stage
    assert treatment_digest in stage
    assert config["schema_version"] == 2
    assert config["physics_data_class"] == "synthetic_validation_only"
    assert config["photon_buildup"] == {"mode": "primary_only"}
    with np.load(activity_path, allow_pickle=False) as payload:
        triangle_indices = payload["triangle_indices"]
        assert len(triangle_indices) == 994
        assert np.array_equal(triangle_indices, np.arange(994))
        assert np.all(payload["activity_bq"] > 0)
        assert np.sum(payload["activity_bq"]) == 8.0e5
        assert np.all(payload["cumulative_treatment_exposure"] == 0)
        assert np.all(payload["verified_contact_dwell_s"] == 0)
    assert "rad:source:irregularMask = true" in stage
    assert "rad:source:candidateCellCount = 1344" in stage
    assert "rad:source:activeCellCount = 497" in stage
    assert "rad:source:activeFaceCount = 994" in stage
    assert 'rad:source:geometry = "irregular_masked_triangle_activity_map"' in stage
    assert 'rad:decon:substrateMaterialId = "concrete"' in stage
    assert "rad:decon:treatmentModelUri" in stage
    assert "rad:decon:efficiencyMean" not in stage
    assert "rad:decon:rateConstantSInv" not in stage
    assert "int[] faceVertexCounts = [3, 3]" not in stage
    assert 'rad:disposal:disposition = "shielded_storage"' in stage
    assert 'rad:disposal:storagePrimPath = "/World/DisposalStorage"' in stage
    assert 'rad:disposal:accessSide = "east"' in stage
    assert stage.count('rad:manipulation:placementReference = "root"') == 2
    assert 'rad:manipulation:parkingOffsetM = (1.6, -1.8, 0)' in stage
    assert 'rad:manipulation:parkingOffsetM = (5.2, -1.8, 0)' in stage
    assert 'rad:manipulation:baseStandOffM = 0.9' in stage
    assert 'rad:manipulation:baseStandOffM = 0.72' in stage
    assert 'def Xform "DisposalStorage"' in stage
    assert 'def Cube "EastWall"' not in stage
    assert 'double3 xformOp:scale = (0.9, 1.3, 0.02)' in stage
    assert 'double3 xformOp:translate = (0, -1.25, 0.7)' in stage
    assert 'double3 xformOp:translate = (0, 1.25, 0.7)' in stage
    assert 'double3 xformOp:scale = (0.8, 1.2, 0.02)' in stage
    assert stage.count('rad:role = "shielded_storage"') == 1
    assert stage.count('rad:material:id = "lead"') >= 8
    required_roles = {
        'rad:role = "contaminated_surface"',
        'rad:role = "source"',
        'rad:role = "shield"',
        'rad:role = "measurement_robot"',
        'rad:role = "countermeasure_robot"',
        'rad:role = "disposal_zone"',
        'rad:role = "shielded_storage"',
    }
    assert all(role in stage for role in required_roles)


def test_experiment_bundle_is_atomic_and_checksums_inputs(tmp_path: Path) -> None:
    stage = tmp_path / "scene.usda"
    config = tmp_path / "scenario.json"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    bundle = ExperimentArtifactBundle(
        tmp_path / "out",
        "run-1",
        stage_path=stage,
        config_path=config,
        evidence_class=EvidenceClass.ANALYTIC_VALIDATION,
    )
    bundle.record(
        ExperimentCaseRecord(
            case_id="seed-00000",
            seed=0,
            evidence_class=EvidenceClass.ANALYTIC_VALIDATION,
            metrics={"residual_cps": 1.5},
            parameters={"pose_error_m": 0.02},
        )
    )
    summary = bundle.finalize()
    manifest = json.loads((summary.parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["case_count"] == 1
    assert manifest["evidence_class"] == "analytic_validation"
    assert manifest["stage"]["sha256"] == hashlib.sha256(stage.read_bytes()).hexdigest()
    assert manifest["summary_sha256"] == hashlib.sha256(summary.read_bytes()).hexdigest()
    assert not tuple(summary.parent.glob(".*.tmp"))


def test_experiment_bundle_rejects_mixed_evidence_classes(tmp_path: Path) -> None:
    stage = tmp_path / "scene.usda"
    config = tmp_path / "scenario.json"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    bundle = ExperimentArtifactBundle(
        tmp_path / "out",
        "run-mixed",
        stage_path=stage,
        config_path=config,
        evidence_class=EvidenceClass.PHYSICAL_ROBOT_EXECUTION,
    )
    with pytest.raises(ValueError, match="does not match"):
        bundle.record(
            ExperimentCaseRecord(
                case_id="scene-edit",
                seed=0,
                evidence_class=EvidenceClass.KINEMATIC_SCENE_EDIT,
                metrics={},
                parameters={},
            )
        )
