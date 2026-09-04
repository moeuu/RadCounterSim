import hashlib
import json
from pathlib import Path

import pytest
import yaml

from radcounter.core.experiments import EvidenceClass, sha256_file
from radcounter.core.experiments.paper import (
    EvaluationDataMode,
    PaperEvaluationError,
    collect_physical_paper_evaluation,
    summarize_physical_robot_artifact,
    validate_paper_physics_evidence,
)


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _numeric_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _research_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    material_numeric = {
        "energies_keV": [100.0, 1000.0],
        "linear_attenuation_m_inv": [2.0, 1.0],
    }
    material = {
        "schema_version": 1,
        "material_id": "test-material",
        "data_status": "authoritative_reference",
        "provenance": {
            "source_name": "unit-test material reference",
            "retrieved_on": "2026-08-29",
            "payload_sha256": _numeric_digest(material_numeric),
        },
        **material_numeric,
    }
    isotope_numeric = {"emission_lines": [{"energy_keV": 500.0, "photons_per_decay": 0.8}]}
    isotope = {
        "schema_version": 1,
        "isotope_id": "Test-500",
        "data_status": "authoritative_reference",
        "provenance": {
            "source_name": "unit-test isotope reference",
            "retrieved_on": "2026-08-29",
            "payload_sha256": _numeric_digest(isotope_numeric),
        },
        **isotope_numeric,
    }
    detector_numeric = {
        "energy_bin_edges_keV": [0.0, 800.0],
        "response_energy_keV": [100.0, 1000.0],
        "effective_area_m2_per_bin": [[0.01], [0.02]],
        "background_cps_per_bin": [0.1],
        "dead_time_s": 1.0e-6,
    }
    detector = {
        "schema_version": 1,
        "detector_id": "test-detector",
        "data_status": "calibrated_measurement",
        "provenance": {
            "source_name": "unit-test detector measurement",
            "retrieved_on": "2026-08-29",
            "payload_sha256": _numeric_digest(detector_numeric),
        },
        **detector_numeric,
    }
    buildup_numeric = {
        "composition_rule": "product_by_material",
        "materials": [
            {
                "material_id": "test-material",
                "energies_keV": [100.0, 1000.0],
                "optical_depths": [0.0, 20.0],
                "factors_by_energy_depth": [[1.0, 2.0], [1.0, 1.5]],
            }
        ],
    }
    buildup = {
        "schema_version": 1,
        "data_status": "reference_calibrated_correction",
        "provenance": {
            "source_name": "unit-test transport reference",
            "retrieved_on": "2026-08-29",
            "payload_sha256": _numeric_digest(buildup_numeric),
        },
        **buildup_numeric,
    }
    material_path = tmp_path / "material.yaml"
    isotope_path = tmp_path / "isotope.yaml"
    detector_path = tmp_path / "detector.yaml"
    buildup_path = tmp_path / "buildup.yaml"
    for path, payload in (
        (material_path, material),
        (isotope_path, isotope),
        (detector_path, detector),
        (buildup_path, buildup),
    ):
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    config = _write_json(
        tmp_path / "research-runtime.json",
        {
            "physics_data_class": "research_evaluation",
            "photon_buildup": {
                "mode": "reference_calibrated_optical_depth_buildup",
                "data_path": buildup_path.name,
                "file_sha256": sha256_file(buildup_path),
            },
            "materials": {
                "test-material": {
                    "energy_keV": material_numeric["energies_keV"],
                    "mu_m_inv": material_numeric["linear_attenuation_m_inv"],
                }
            },
            "isotopes": {"Test-500": {"lines": [{"energy_keV": 500.0, "yield_per_decay": 0.8}]}},
            "detectors": {"test-detector": detector_numeric},
        },
    )
    return config, material_path, isotope_path, detector_path, buildup_path


def _result(public_details: dict[str, object], started: float) -> dict[str, object]:
    return {
        "status": "completed",
        "started_sim_s": started,
        "completed_sim_s": started + 1.0,
        "public_details": public_details,
    }


def _execution_runtime() -> dict[str, object]:
    return {
        "python_version": "3.12.12",
        "platform": "Linux-test",
        "numpy_version": "2.5.1",
        "radinteract_version": "0.1.0",
        "isaac_sim_version": "6.0.1.0",
        "embree_version": "4.3.0",
        "renderer_mode": "RaytracedLighting",
        "gpus": [
            {
                "name": "Test GPU",
                "driver_version": "580.0",
                "memory_mib": 8192,
            }
        ],
    }


def _artifact(
    path: Path,
    *,
    config: Path,
    stage: Path,
    seed: int,
) -> Path:
    decon_details: dict[str, object] = {
        "removed_activity_bq": 1200.0,
        "treated_area_coverage_fraction": 0.75,
        "activity_balance_error_bq": 1.0e-9,
        "treatment_data_status": "synthetic_validation_only",
        "treatment_numeric_sha256": "a" * 64,
    }
    operations = [
        {
            "label": "measurement survey",
            "station_executions": [
                {
                    "action": {"action_type": "measure"},
                    "result": _result({"motion_audit": {"success": True}}, 0.0),
                }
            ],
        },
        {
            "action": {"action_type": "decontaminate"},
            "result": _result(decon_details, 1.0),
        },
        {
            "action": {"action_type": "place_shield"},
            "result": _result({"motion_audit": {"placement_error_m": 0.02}}, 2.0),
        },
        {
            "action": {"action_type": "move_shield"},
            "result": _result({"motion_audit": {"placement_error_m": 0.03}}, 3.0),
        },
        {
            "action": {"action_type": "move_object"},
            "result": _result({"motion_audit": {"placement_error_m": 0.04}}, 4.0),
        },
        {
            "action": {"action_type": "remove_object"},
            "result": _result({"motion_audit": {"placement_error_m": 0.05}}, 5.0),
        },
    ]
    payload = {
        "success": True,
        "evidence_class": EvidenceClass.PHYSICAL_ROBOT_EXECUTION.value,
        "physics_data_class": "synthetic_validation_only",
        "runtime_config_sha256": sha256_file(config),
        "stage_sha256": sha256_file(stage),
        "seed": seed,
        "execution_runtime": _execution_runtime(),
        "operations": operations,
        "radiation_audit": {
            "initial": {
                "total_cps_per_bin": [10.0, 5.0],
                "primary_cps_per_bin": [8.0, 4.0],
                "corrected_source_cps_per_bin": [8.0, 4.0],
                "buildup_model_name": "primary_only",
            },
            "final": {
                "total_cps_per_bin": [5.0, 2.5],
                "primary_cps_per_bin": [4.0, 2.0],
                "corrected_source_cps_per_bin": [4.0, 2.0],
                "buildup_model_name": "primary_only",
            },
        },
        "resource_audit": {
            "initial": {
                "measurement_time_s": 20.0,
                "work_time_s": 30.0,
                "robot_runtime_s": {"survey": 20.0, "work": 30.0},
                "shield_units": {"lead": 2},
                "decon_media": 10.0,
                "clean_water_l": 5.0,
                "wastewater_capacity_l": 5.0,
                "countermeasure_count": 10,
            },
            "final": {
                "measurement_time_s": 16.0,
                "work_time_s": 25.0,
                "robot_runtime_s": {"survey": 18.0, "work": 25.0},
                "shield_units": {"lead": 1},
                "decon_media": 8.0,
                "clean_water_l": 5.0,
                "wastewater_capacity_l": 5.0,
                "countermeasure_count": 5,
            },
        },
        "estimation_audit": {
            "initial": {
                "source_strength_bq": [1000.0],
                "covariance_diagonal": [4.0],
            },
            "final": {
                "source_strength_bq": [500.0],
                "covariance_diagonal": [1.0],
            },
            "residual": {"normalized_residual": [0.5, -0.5]},
            "initial_solver": {
                "solver": "nonparalyzable_dead_time_poisson_L-BFGS-B",
                "template_kinds": ["uniform_area_on_visible_surface"],
                "observed_counts": [100, 50],
                "predicted_counts": [99.0, 51.0],
            },
            "final_solver": {
                "solver": "nonparalyzable_dead_time_poisson_L-BFGS-B",
                "template_kinds": ["uniform_area_on_visible_surface"],
                "observed_counts": [50, 25],
                "predicted_counts": [50.0, 25.0],
            },
        },
        "transport_statistics": {
            "native_trace_calls": 2,
            "traced_rays": 100,
            "cache_hits": 20,
            "selectively_patched_rays": 5,
        },
        "wall_time_s": 12.0,
    }
    return _write_json(path, payload)


def _synthetic_inputs(tmp_path: Path) -> tuple[Path, Path]:
    stage = tmp_path / "stage.usda"
    stage.write_text("#usda 1.0\n", encoding="utf-8")
    config = _write_json(
        tmp_path / "runtime.json",
        {
            "physics_data_class": "synthetic_validation_only",
            "photon_buildup": {"mode": "primary_only"},
        },
    )
    return stage, config


def test_physical_paper_collection_writes_separate_multiseed_metrics(
    tmp_path: Path,
) -> None:
    stage, config = _synthetic_inputs(tmp_path)
    physics = validate_paper_physics_evidence(
        config,
        mode=EvaluationDataMode.SYNTHETIC_VALIDATION,
    )
    artifacts = [
        _artifact(tmp_path / f"seed-{seed}.json", config=config, stage=stage, seed=seed)
        for seed in (3, 7)
    ]
    summary = collect_physical_paper_evaluation(
        output_root=tmp_path / "paper",
        run_id="physical-test",
        stage_path=stage,
        config_path=config,
        artifact_paths=artifacts,
        physics=physics,
    )
    assert summary.is_file()
    aggregate = json.loads((summary.parent / "aggregate_metrics.json").read_text(encoding="utf-8"))
    assert aggregate["seed_count"] == 2
    assert aggregate["metrics"]["action_success_fraction"]["mean"] == 1.0
    assert aggregate["metrics"]["measurement_time_used_s"]["mean"] == 4.0
    manifest = json.loads((summary.parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["evidence_class"] == "physical_robot_execution"
    assert manifest["aggregate_metrics_sha256"] == sha256_file(
        summary.parent / "aggregate_metrics.json"
    )
    assert manifest["execution_runtime"] == _execution_runtime()


def test_physical_paper_artifact_requires_execution_runtime(tmp_path: Path) -> None:
    stage, config = _synthetic_inputs(tmp_path)
    physics = validate_paper_physics_evidence(
        config,
        mode=EvaluationDataMode.SYNTHETIC_VALIDATION,
    )
    artifact = _artifact(tmp_path / "run.json", config=config, stage=stage, seed=2)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    del payload["execution_runtime"]
    _write_json(artifact, payload)
    with pytest.raises(PaperEvaluationError, match="execution_runtime"):
        summarize_physical_robot_artifact(artifact, physics=physics)


def test_physical_paper_collection_rejects_mixed_execution_runtimes(
    tmp_path: Path,
) -> None:
    stage, config = _synthetic_inputs(tmp_path)
    physics = validate_paper_physics_evidence(
        config,
        mode=EvaluationDataMode.SYNTHETIC_VALIDATION,
    )
    first = _artifact(tmp_path / "first.json", config=config, stage=stage, seed=2)
    second = _artifact(tmp_path / "second.json", config=config, stage=stage, seed=3)
    payload = json.loads(second.read_text(encoding="utf-8"))
    payload["execution_runtime"]["gpus"][0]["driver_version"] = "581.0"
    _write_json(second, payload)
    with pytest.raises(PaperEvaluationError, match="different execution runtimes"):
        collect_physical_paper_evaluation(
            output_root=tmp_path / "paper",
            run_id="mixed-runtime",
            stage_path=stage,
            config_path=config,
            artifact_paths=[first, second],
            physics=physics,
        )


def test_physical_paper_artifact_rejects_missing_activity_balance(tmp_path: Path) -> None:
    stage, config = _synthetic_inputs(tmp_path)
    physics = validate_paper_physics_evidence(
        config,
        mode=EvaluationDataMode.SYNTHETIC_VALIDATION,
    )
    artifact = _artifact(tmp_path / "run.json", config=config, stage=stage, seed=2)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    del payload["operations"][1]["result"]["public_details"]["activity_balance_error_bq"]
    _write_json(artifact, payload)
    with pytest.raises(PaperEvaluationError, match="activity_balance_error_bq"):
        summarize_physical_robot_artifact(artifact, physics=physics)


def test_physical_paper_artifact_rejects_zero_action_time(tmp_path: Path) -> None:
    stage, config = _synthetic_inputs(tmp_path)
    physics = validate_paper_physics_evidence(
        config,
        mode=EvaluationDataMode.SYNTHETIC_VALIDATION,
    )
    artifact = _artifact(tmp_path / "run.json", config=config, stage=stage, seed=2)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    for operation in payload["operations"]:
        results = operation.get("station_executions", [operation])
        for row in results:
            row["result"]["started_sim_s"] = 0.0
            row["result"]["completed_sim_s"] = 0.0
    _write_json(artifact, payload)
    with pytest.raises(PaperEvaluationError, match="zero simulated time"):
        summarize_physical_robot_artifact(artifact, physics=physics)


def test_research_collection_rejects_synthetic_runtime_config(tmp_path: Path) -> None:
    _, config = _synthetic_inputs(tmp_path)
    with pytest.raises(PaperEvaluationError, match="physics_data_class"):
        validate_paper_physics_evidence(
            config,
            mode=EvaluationDataMode.RESEARCH_EVALUATION,
        )


def test_research_physics_evidence_matches_exact_runtime_values(tmp_path: Path) -> None:
    config, material, isotope, detector, buildup = _research_inputs(tmp_path)
    evidence = validate_paper_physics_evidence(
        config,
        mode=EvaluationDataMode.RESEARCH_EVALUATION,
        material_paths=[material],
        isotope_paths=[isotope],
        detector_paths=[detector],
        buildup_path=buildup,
    )
    assert evidence.physics_data_class == "research_evaluation"
    assert evidence.buildup_model_name == ("reference_calibrated_optical_depth_buildup")
    assert len(evidence.physics_bundle_sha256) == 64


def test_research_physics_evidence_rejects_runtime_numeric_mismatch(
    tmp_path: Path,
) -> None:
    config, material, isotope, detector, buildup = _research_inputs(tmp_path)
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["materials"]["test-material"]["mu_m_inv"][0] *= 1.1
    _write_json(config, payload)
    with pytest.raises(PaperEvaluationError, match="differs from its validated dataset"):
        validate_paper_physics_evidence(
            config,
            mode=EvaluationDataMode.RESEARCH_EVALUATION,
            material_paths=[material],
            isotope_paths=[isotope],
            detector_paths=[detector],
            buildup_path=buildup,
        )
