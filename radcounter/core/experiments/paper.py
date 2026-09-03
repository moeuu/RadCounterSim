"""Fail-closed collection of paper-evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from radcounter.core.experiments.atomic_bundle import (
    EvidenceClass,
    ExperimentArtifactBundle,
    ExperimentCaseRecord,
    sha256_file,
)
from radcounter.core.models.radiation import DetectorSpec, IsotopeSpec, MaterialSpec
from radcounter.core.radiation.data import validate_research_evaluation_data


class PaperEvaluationError(ValueError):
    """Raised when an input cannot support the requested evaluation claim."""


class EvaluationDataMode(StrEnum):
    SYNTHETIC_VALIDATION = "synthetic_validation"
    RESEARCH_EVALUATION = "research_evaluation"


@dataclass(frozen=True, slots=True)
class PaperPhysicsEvidence:
    mode: EvaluationDataMode
    physics_data_class: str
    runtime_config_sha256: str
    physics_bundle_sha256: str
    buildup_model_name: str


def _load_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PaperEvaluationError(f"invalid JSON artifact {source}: {error}") from error
    if not isinstance(payload, dict):
        raise PaperEvaluationError(f"JSON artifact root must be an object: {source}")
    return source, payload


def validate_paper_physics_evidence(
    config_path: str | Path,
    *,
    mode: EvaluationDataMode,
    material_paths: Sequence[str | Path] = (),
    isotope_paths: Sequence[str | Path] = (),
    detector_paths: Sequence[str | Path] = (),
    buildup_path: str | Path | None = None,
) -> PaperPhysicsEvidence:
    """Validate the exact physics-evidence boundary before importing results."""

    config, payload = _load_json(config_path)
    data_class = str(payload.get("physics_data_class", ""))
    buildup = payload.get("photon_buildup")
    if not isinstance(buildup, Mapping) or not buildup.get("mode"):
        raise PaperEvaluationError("runtime config requires an explicit photon_buildup mode")
    buildup_mode = str(buildup["mode"])
    supported_buildup_modes = {
        "primary_only": "primary_only",
        "reference_calibrated_optical_depth_buildup": (
            "reference_calibrated_optical_depth_buildup"
        ),
    }
    try:
        buildup_name = supported_buildup_modes[buildup_mode]
    except KeyError as error:
        raise PaperEvaluationError(f"unsupported photon_buildup mode: {buildup_mode!r}") from error
    selected_mode = EvaluationDataMode(mode)
    if selected_mode is EvaluationDataMode.SYNTHETIC_VALIDATION:
        if data_class != "synthetic_validation_only":
            raise PaperEvaluationError(
                "synthetic-validation collection requires physics_data_class="
                "'synthetic_validation_only'"
            )
        return PaperPhysicsEvidence(
            selected_mode,
            data_class,
            sha256_file(config),
            sha256_file(config),
            buildup_name,
        )
    if data_class != "research_evaluation":
        raise PaperEvaluationError(
            "research collection requires physics_data_class='research_evaluation'"
        )
    if not material_paths or not isotope_paths or not detector_paths or buildup_path is None:
        raise PaperEvaluationError(
            "research collection requires material, isotope, detector, and buildup datasets"
        )
    try:
        bundle = validate_research_evaluation_data(
            material_paths=material_paths,
            isotope_paths=isotope_paths,
            detector_paths=detector_paths,
            buildup_path=buildup_path,
        )
    except ValueError as error:
        raise PaperEvaluationError(str(error)) from error
    if buildup_mode != "reference_calibrated_optical_depth_buildup":
        raise PaperEvaluationError(
            "research runtime config must select reference_calibrated_optical_depth_buildup"
        )
    _validate_runtime_config_against_bundle(
        config,
        payload,
        bundle=bundle,
        buildup_path=Path(buildup_path),
    )
    return PaperPhysicsEvidence(
        selected_mode,
        data_class,
        sha256_file(config),
        bundle.bundle_sha256,
        buildup_name,
    )


def _same_array(left: object, right: object) -> bool:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    return first.shape == second.shape and bool(
        np.allclose(first, second, rtol=1.0e-12, atol=1.0e-15)
    )


def _validate_runtime_config_against_bundle(
    config_path: Path,
    payload: Mapping[str, Any],
    *,
    bundle: Any,
    buildup_path: Path,
) -> None:
    """Prove that the validated datasets are the values used by the runtime."""

    runtime_materials = _mapping(payload.get("materials"), name="runtime materials")
    material_ids = {entry.data_id for entry in bundle.materials}
    if set(runtime_materials) != material_ids:
        raise PaperEvaluationError("runtime material IDs do not match validated datasets")
    for entry in bundle.materials:
        value = entry.value
        if not isinstance(value, MaterialSpec):
            raise PaperEvaluationError("validated material bundle contains an invalid value")
        runtime = _mapping(
            runtime_materials[entry.data_id],
            name=f"runtime material {entry.data_id}",
        )
        if not _same_array(runtime.get("energy_keV"), value.energies_keV) or not _same_array(
            runtime.get("mu_m_inv"), value.linear_attenuation_m_inv
        ):
            raise PaperEvaluationError(
                f"runtime material {entry.data_id!r} differs from its validated dataset"
            )

    runtime_isotopes = _mapping(payload.get("isotopes"), name="runtime isotopes")
    isotope_ids = {entry.data_id for entry in bundle.isotopes}
    if set(runtime_isotopes) != isotope_ids:
        raise PaperEvaluationError("runtime isotope IDs do not match validated datasets")
    for entry in bundle.isotopes:
        value = entry.value
        if not isinstance(value, IsotopeSpec):
            raise PaperEvaluationError("validated isotope bundle contains an invalid value")
        runtime = _mapping(
            runtime_isotopes[entry.data_id],
            name=f"runtime isotope {entry.data_id}",
        )
        lines = runtime.get("lines")
        expected = [(line.energy_keV, line.photons_per_decay) for line in value.emission_lines]
        actual = []
        if isinstance(lines, list):
            for line in lines:
                if not isinstance(line, Mapping):
                    break
                actual.append((line.get("energy_keV"), line.get("yield_per_decay")))
        if not _same_array(actual, expected):
            raise PaperEvaluationError(
                f"runtime isotope {entry.data_id!r} differs from its validated dataset"
            )

    runtime_detectors = _mapping(payload.get("detectors"), name="runtime detectors")
    detector_ids = {entry.data_id for entry in bundle.detectors}
    if set(runtime_detectors) != detector_ids:
        raise PaperEvaluationError("runtime detector IDs do not match validated datasets")
    for entry in bundle.detectors:
        value = entry.value
        if not isinstance(value, DetectorSpec):
            raise PaperEvaluationError("validated detector bundle contains an invalid value")
        runtime = _mapping(
            runtime_detectors[entry.data_id],
            name=f"runtime detector {entry.data_id}",
        )
        arrays_match = all(
            (
                _same_array(runtime.get("energy_bin_edges_keV"), value.energy_bin_edges_keV),
                _same_array(runtime.get("response_energy_keV"), value.response_energy_keV),
                _same_array(
                    runtime.get("effective_area_m2_per_bin"),
                    value.effective_area_m2_per_bin,
                ),
                _same_array(
                    runtime.get("background_cps_per_bin"),
                    value.background_cps_per_bin,
                ),
            )
        )
        if not arrays_match or not math.isclose(
            float(runtime.get("dead_time_s", -1.0)),
            value.dead_time_s,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise PaperEvaluationError(
                f"runtime detector {entry.data_id!r} differs from its validated dataset"
            )

    buildup = _mapping(payload.get("photon_buildup"), name="photon_buildup")
    configured_path = Path(str(buildup.get("data_path", "")))
    if not configured_path.is_absolute():
        configured_path = (config_path.parent / configured_path).resolve()
    supplied_path = buildup_path.expanduser().resolve()
    if configured_path != supplied_path:
        raise PaperEvaluationError("runtime buildup path does not match validated dataset")
    if str(buildup.get("file_sha256", "")) != sha256_file(supplied_path):
        raise PaperEvaluationError("runtime buildup SHA256 does not match validated dataset")


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PaperEvaluationError(f"{name} must be an object")
    return value


def _object_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_execution_runtime(payload: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _mapping(payload.get("execution_runtime"), name="execution_runtime")
    string_fields = (
        "python_version",
        "platform",
        "numpy_version",
        "radcountersim_version",
        "isaac_sim_version",
        "embree_version",
        "renderer_mode",
    )
    normalized: dict[str, Any] = {}
    for field in string_fields:
        value = runtime.get(field)
        if not isinstance(value, str) or not value.strip():
            raise PaperEvaluationError(f"execution_runtime.{field} must be a nonempty string")
        normalized[field] = value
    gpu_payload = runtime.get("gpus")
    if not isinstance(gpu_payload, list) or not gpu_payload:
        raise PaperEvaluationError("execution_runtime.gpus must be a nonempty array")
    gpus: list[dict[str, Any]] = []
    for index, value in enumerate(gpu_payload):
        gpu = _mapping(value, name=f"execution_runtime.gpus[{index}]")
        name = gpu.get("name")
        driver = gpu.get("driver_version")
        memory = gpu.get("memory_mib")
        if not isinstance(name, str) or not name.strip():
            raise PaperEvaluationError(f"execution_runtime.gpus[{index}].name is required")
        if not isinstance(driver, str) or not driver.strip():
            raise PaperEvaluationError(
                f"execution_runtime.gpus[{index}].driver_version is required"
            )
        if (
            isinstance(memory, bool)
            or not isinstance(memory, int)
            or memory <= 0
        ):
            raise PaperEvaluationError(
                f"execution_runtime.gpus[{index}].memory_mib must be a positive integer"
            )
        gpus.append(
            {"name": name, "driver_version": driver, "memory_mib": memory}
        )
    normalized["gpus"] = gpus
    return normalized


def _finite_number(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise PaperEvaluationError(f"{name} must be a finite number")
    return float(value)


def _sum_array(value: object, *, name: str) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise PaperEvaluationError(f"{name} must be a nonempty finite array")
    return float(np.sum(array))


def _recursive_numbers(value: object, key: str) -> list[float]:
    result: list[float] = []
    if isinstance(value, Mapping):
        for name, item in value.items():
            if name == key and isinstance(item, (int, float)) and math.isfinite(float(item)):
                result.append(float(item))
            else:
                result.extend(_recursive_numbers(item, key))
    elif isinstance(value, (list, tuple)):
        for item in value:
            result.extend(_recursive_numbers(item, key))
    return result


def _action_type(operation: Mapping[str, Any]) -> str:
    action = operation.get("action")
    return "" if not isinstance(action, Mapping) else str(action.get("action_type", ""))


def _physical_action_rows(operations: Sequence[object]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for operation in operations:
        if not isinstance(operation, Mapping):
            continue
        if isinstance(operation.get("result"), Mapping):
            rows.append(operation)
        station_executions = operation.get("station_executions", ())
        if isinstance(station_executions, list):
            rows.extend(
                execution
                for execution in station_executions
                if isinstance(execution, Mapping) and isinstance(execution.get("result"), Mapping)
            )
    return rows


def _used_scalar(
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
    key: str,
) -> float:
    before = _finite_number(initial.get(key), name=f"initial resource {key}")
    after = _finite_number(final.get(key), name=f"final resource {key}")
    if after > before + 1.0e-9:
        raise PaperEvaluationError(f"remaining resource {key} increased during the run")
    return max(0.0, before - after)


def _used_mapping(
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
    key: str,
) -> float:
    before = _mapping(initial.get(key), name=f"initial resource {key}")
    after = _mapping(final.get(key), name=f"final resource {key}")
    if set(before) != set(after):
        raise PaperEvaluationError(f"resource keys changed during the run: {key}")
    return sum(_used_scalar(before, after, name) for name in before)


def summarize_physical_robot_artifact(
    path: str | Path,
    *,
    physics: PaperPhysicsEvidence,
) -> ExperimentCaseRecord:
    """Convert one full physical run into one scalar, auditable case record."""

    source, payload = _load_json(path)
    if payload.get("success") is not True:
        raise PaperEvaluationError(f"physical run did not succeed: {source}")
    if payload.get("evidence_class") != EvidenceClass.PHYSICAL_ROBOT_EXECUTION.value:
        raise PaperEvaluationError(f"artifact is not physical-robot evidence: {source}")
    if payload.get("physics_data_class") != physics.physics_data_class:
        raise PaperEvaluationError(
            f"artifact physics_data_class does not match collection mode: {source}"
        )
    if payload.get("runtime_config_sha256") != physics.runtime_config_sha256:
        raise PaperEvaluationError(
            f"artifact runtime configuration does not match collection input: {source}"
        )
    execution_runtime = _validated_execution_runtime(payload)
    stage_sha256 = str(payload.get("stage_sha256", ""))
    if len(stage_sha256) != 64:
        raise PaperEvaluationError(f"artifact requires a stage SHA256: {source}")
    seed = payload.get("seed")
    if not isinstance(seed, int) or seed < 0:
        raise PaperEvaluationError(
            f"physical artifact requires a nonnegative integer seed: {source}"
        )
    operations = payload.get("operations")
    if not isinstance(operations, list) or not operations:
        raise PaperEvaluationError(f"physical artifact contains no operations: {source}")
    action_rows = _physical_action_rows(operations)
    if not action_rows:
        raise PaperEvaluationError(f"physical artifact contains no action results: {source}")
    statuses = [
        str(_mapping(row["result"], name="action result").get("status", "")) for row in action_rows
    ]
    action_success_fraction = sum(status == "completed" for status in statuses) / len(statuses)
    if action_success_fraction != 1.0:
        raise PaperEvaluationError(f"physical artifact contains a non-completed action: {source}")

    decon_rows = [row for row in action_rows if _action_type(row) == "decontaminate"]
    shield_rows = [
        row for row in action_rows if _action_type(row) in {"place_shield", "move_shield"}
    ]
    move_rows = [row for row in action_rows if _action_type(row) == "move_object"]
    removal_rows = [row for row in action_rows if _action_type(row) == "remove_object"]
    if not decon_rows or not shield_rows or not move_rows or not removal_rows:
        raise PaperEvaluationError(
            "paper physical scenario requires decontamination, shielding, relocation, and removal"
        )
    decon_details = _mapping(
        _mapping(decon_rows[-1]["result"], name="decontamination result").get("public_details"),
        name="decontamination public details",
    )
    required_decon = {
        "removed_activity_bq",
        "treated_area_coverage_fraction",
        "activity_balance_error_bq",
        "treatment_data_status",
        "treatment_numeric_sha256",
    }
    missing_decon = sorted(required_decon.difference(decon_details))
    if missing_decon:
        raise PaperEvaluationError(
            "decontamination result is missing paper metrics: " + ", ".join(missing_decon)
        )
    pose_errors = [
        *_recursive_numbers(shield_rows, "placement_error_m"),
        *_recursive_numbers(move_rows, "placement_error_m"),
        *_recursive_numbers(removal_rows, "placement_error_m"),
    ]
    if not pose_errors:
        raise PaperEvaluationError("physical actions contain no final placement error")

    resource_audit = _mapping(payload.get("resource_audit"), name="resource_audit")
    initial_resources = _mapping(resource_audit.get("initial"), name="resource_audit.initial")
    final_resources = _mapping(resource_audit.get("final"), name="resource_audit.final")
    estimation_audit = _mapping(payload.get("estimation_audit"), name="estimation_audit")
    initial_estimate = _mapping(estimation_audit.get("initial"), name="estimation_audit.initial")
    final_estimate = _mapping(estimation_audit.get("final"), name="estimation_audit.final")
    initial_solver = _mapping(
        estimation_audit.get("initial_solver"), name="estimation_audit.initial_solver"
    )
    final_solver = _mapping(
        estimation_audit.get("final_solver"), name="estimation_audit.final_solver"
    )
    expected_solver = "nonparalyzable_dead_time_poisson_L-BFGS-B"
    if any(
        str(solver.get("solver", "")) != expected_solver
        for solver in (initial_solver, final_solver)
    ):
        raise PaperEvaluationError("physical paper run requires the public Poisson estimator")
    if any(not solver.get("template_kinds") for solver in (initial_solver, final_solver)):
        raise PaperEvaluationError("estimator audit requires explicit public template kinds")
    residual = _mapping(estimation_audit.get("residual"), name="estimation_audit.residual")
    initial_covariance = _sum_array(
        initial_estimate.get("covariance_diagonal"),
        name="initial estimate covariance diagonal",
    )
    final_covariance = _sum_array(
        final_estimate.get("covariance_diagonal"),
        name="final estimate covariance diagonal",
    )
    initial_activity = _sum_array(
        initial_estimate.get("source_strength_bq"),
        name="initial estimated source activity",
    )
    final_activity = _sum_array(
        final_estimate.get("source_strength_bq"),
        name="final estimated source activity",
    )

    def solver_rmse(solver: Mapping[str, Any], *, name: str) -> float:
        observed = np.asarray(solver.get("observed_counts"), dtype=np.float64)
        predicted = np.asarray(solver.get("predicted_counts"), dtype=np.float64)
        if (
            observed.ndim != 1
            or observed.shape != predicted.shape
            or len(observed) == 0
            or not np.all(np.isfinite(observed))
            or not np.all(np.isfinite(predicted))
        ):
            raise PaperEvaluationError(f"{name} estimator count audit is invalid")
        return float(np.sqrt(np.mean((predicted - observed) ** 2)))

    normalized_residual = np.asarray(residual.get("normalized_residual"), dtype=np.float64)
    if normalized_residual.ndim != 1 or not np.all(np.isfinite(normalized_residual)):
        raise PaperEvaluationError("normalized residual must be a finite array")

    radiation = _mapping(payload.get("radiation_audit"), name="radiation_audit")
    initial = _mapping(radiation.get("initial"), name="radiation_audit.initial")
    final = _mapping(radiation.get("final"), name="radiation_audit.final")
    initial_total = _sum_array(initial.get("total_cps_per_bin"), name="initial total spectrum")
    final_total = _sum_array(final.get("total_cps_per_bin"), name="final total spectrum")
    initial_primary = _sum_array(
        initial.get("primary_cps_per_bin"), name="initial primary spectrum"
    )
    final_primary = _sum_array(final.get("primary_cps_per_bin"), name="final primary spectrum")
    initial_corrected = _sum_array(
        initial.get("corrected_source_cps_per_bin"),
        name="initial corrected source spectrum",
    )
    final_corrected = _sum_array(
        final.get("corrected_source_cps_per_bin"),
        name="final corrected source spectrum",
    )
    buildup_model = str(initial.get("buildup_model_name", ""))
    if not buildup_model or buildup_model != str(final.get("buildup_model_name", "")):
        raise PaperEvaluationError("radiation buildup model must be present and stable")
    if buildup_model != physics.buildup_model_name:
        raise PaperEvaluationError(
            "artifact buildup model does not match the validated runtime configuration"
        )

    transport = _mapping(payload.get("transport_statistics"), name="transport_statistics")
    wall_time_s = _finite_number(payload.get("wall_time_s"), name="wall_time_s")
    sim_durations = []
    for row in action_rows:
        result = _mapping(row["result"], name="action result")
        started = _finite_number(result.get("started_sim_s"), name="started_sim_s")
        completed = _finite_number(result.get("completed_sim_s"), name="completed_sim_s")
        if completed < started:
            raise PaperEvaluationError("action completion time precedes its start")
        sim_durations.append(completed - started)
    if not any(duration > 0.0 for duration in sim_durations):
        raise PaperEvaluationError(
            "physical paper artifact reports zero simulated time for every action"
        )

    metrics: dict[str, float | int | str | bool] = {
        "action_success_fraction": action_success_fraction,
        "physical_action_count": len(action_rows),
        **{
            f"action_count_{action_type}": sum(
                _action_type(row) == action_type for row in action_rows
            )
            for action_type in (
                "measure",
                "decontaminate",
                "place_shield",
                "move_shield",
                "move_object",
                "remove_object",
            )
        },
        "maximum_final_pose_error_m": max(pose_errors),
        "median_final_pose_error_m": float(np.median(pose_errors)),
        "treated_area_coverage_fraction": _finite_number(
            decon_details["treated_area_coverage_fraction"],
            name="treated_area_coverage_fraction",
        ),
        "removed_activity_bq": _finite_number(
            decon_details["removed_activity_bq"], name="removed_activity_bq"
        ),
        "activity_balance_error_bq": _finite_number(
            decon_details["activity_balance_error_bq"],
            name="activity_balance_error_bq",
        ),
        "initial_total_rate_cps": initial_total,
        "final_total_rate_cps": final_total,
        "initial_primary_rate_cps": initial_primary,
        "final_primary_rate_cps": final_primary,
        "initial_corrected_source_rate_cps": initial_corrected,
        "final_corrected_source_rate_cps": final_corrected,
        "detector_response_reduction_fraction": (
            0.0 if initial_total <= 0.0 else 1.0 - final_total / initial_total
        ),
        "measurement_time_used_s": _used_scalar(
            initial_resources, final_resources, "measurement_time_s"
        ),
        "work_time_used_s": _used_scalar(initial_resources, final_resources, "work_time_s"),
        "robot_runtime_used_s": _used_mapping(
            initial_resources, final_resources, "robot_runtime_s"
        ),
        "shield_units_used": _used_mapping(initial_resources, final_resources, "shield_units"),
        "decon_media_used": _used_scalar(initial_resources, final_resources, "decon_media"),
        "clean_water_used_l": _used_scalar(initial_resources, final_resources, "clean_water_l"),
        "wastewater_capacity_used_l": _used_scalar(
            initial_resources, final_resources, "wastewater_capacity_l"
        ),
        "countermeasure_count_used": _used_scalar(
            initial_resources, final_resources, "countermeasure_count"
        ),
        "initial_activity_covariance_trace_bq2": initial_covariance,
        "final_activity_covariance_trace_bq2": final_covariance,
        "initial_estimated_activity_bq": initial_activity,
        "final_estimated_activity_bq": final_activity,
        "initial_estimator_count_rmse": solver_rmse(initial_solver, name="initial"),
        "final_estimator_count_rmse": solver_rmse(final_solver, name="final"),
        "final_normalized_residual_l2": float(np.linalg.norm(normalized_residual)),
        "wall_time_s": wall_time_s,
        "action_sim_time_s": float(np.sum(sim_durations)),
        "native_trace_calls": int(transport.get("native_trace_calls", 0)),
        "traced_rays": int(transport.get("traced_rays", 0)),
        "cache_hits": int(transport.get("cache_hits", 0)),
        "selectively_patched_rays": int(transport.get("selectively_patched_rays", 0)),
    }
    if metrics["native_trace_calls"] <= 0 or metrics["traced_rays"] <= 0:
        raise PaperEvaluationError("physical paper artifact did not use native transport")
    parameters: dict[str, float | int | str | bool] = {
        "source_artifact": str(source),
        "source_artifact_sha256": sha256_file(source),
        "stage_sha256": stage_sha256,
        "physics_data_class": physics.physics_data_class,
        "runtime_config_sha256": physics.runtime_config_sha256,
        "physics_bundle_sha256": physics.physics_bundle_sha256,
        "buildup_model_name": buildup_model,
        "treatment_data_status": str(decon_details["treatment_data_status"]),
        "treatment_numeric_sha256": str(decon_details["treatment_numeric_sha256"]),
        "execution_runtime_sha256": _object_sha256(execution_runtime),
        "isaac_sim_version": str(execution_runtime["isaac_sim_version"]),
        "embree_version": str(execution_runtime["embree_version"]),
        "renderer_mode": str(execution_runtime["renderer_mode"]),
        "gpu_names": ", ".join(
            str(gpu["name"]) for gpu in execution_runtime["gpus"]
        ),
        "gpu_driver_versions": ", ".join(
            str(gpu["driver_version"]) for gpu in execution_runtime["gpus"]
        ),
    }
    return ExperimentCaseRecord(
        case_id=f"physical-seed-{seed:05d}",
        seed=seed,
        evidence_class=EvidenceClass.PHYSICAL_ROBOT_EXECUTION,
        metrics=metrics,
        parameters=parameters,
    )


def write_aggregate_metrics(
    path: str | Path,
    records: Iterable[ExperimentCaseRecord],
) -> Path:
    """Write mean/std/min/max for numeric metrics shared across all seeds."""

    selected = tuple(records)
    if not selected:
        raise PaperEvaluationError("at least one paper record is required")
    common_names = set.intersection(*(set(record.metrics) for record in selected))
    aggregate: dict[str, dict[str, float | int]] = {}
    for name in sorted(common_names):
        values = [record.metrics[name] for record in selected]
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values
        ):
            continue
        array = np.asarray(values, dtype=np.float64)
        aggregate[name] = {
            "n": len(array),
            "mean": float(np.mean(array)),
            "sample_std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
            "minimum": float(np.min(array)),
            "maximum": float(np.max(array)),
        }
    payload = {
        "schema_version": 1,
        "seed_count": len(selected),
        "metrics": aggregate,
    }
    destination = Path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return destination


def collect_physical_paper_evaluation(
    *,
    output_root: str | Path,
    run_id: str,
    stage_path: str | Path,
    config_path: str | Path,
    artifact_paths: Sequence[str | Path],
    physics: PaperPhysicsEvidence,
) -> Path:
    """Collect multiple physical seeds without mixing other evidence classes."""

    if not artifact_paths:
        raise PaperEvaluationError("at least one physical artifact is required")
    if not run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in run_id
    ):
        raise PaperEvaluationError(
            "run_id may contain only letters, numbers, dot, dash, underscore"
        )
    execution_runtimes = tuple(
        _validated_execution_runtime(_load_json(path)[1]) for path in artifact_paths
    )
    runtime_digests = {_object_sha256(runtime) for runtime in execution_runtimes}
    if len(runtime_digests) != 1:
        raise PaperEvaluationError(
            "physical artifacts from different execution runtimes cannot be pooled"
        )
    records = tuple(
        summarize_physical_robot_artifact(path, physics=physics) for path in artifact_paths
    )
    expected_stage_sha256 = sha256_file(stage_path)
    if any(record.parameters.get("stage_sha256") != expected_stage_sha256 for record in records):
        raise PaperEvaluationError(
            "one or more physical artifacts were produced from a different stage"
        )
    if len({record.seed for record in records}) != len(records):
        raise PaperEvaluationError("physical artifacts must have unique seeds")
    artifact_digests = {
        str(Path(path).expanduser().resolve()): sha256_file(path) for path in artifact_paths
    }
    bundle = ExperimentArtifactBundle(
        output_root,
        run_id,
        stage_path=stage_path,
        config_path=config_path,
        evidence_class=EvidenceClass.PHYSICAL_ROBOT_EXECUTION,
        metadata={
            "evaluation": "paper_physical_multi_operation",
            "physics_data_mode": physics.mode.value,
            "physics_data_class": physics.physics_data_class,
            "physics_bundle_sha256": physics.physics_bundle_sha256,
            "input_artifact_sha256": artifact_digests,
        },
        execution_runtime=execution_runtimes[0],
    )
    for record in records:
        bundle.record(record)
    summary = bundle.finalize()
    write_aggregate_metrics(summary.parent / "aggregate_metrics.json", records)
    manifest_path = summary.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["aggregate_metrics_sha256"] = sha256_file(summary.parent / "aggregate_metrics.json")
    ExperimentArtifactBundle._atomic_json(manifest_path, manifest)
    return summary
