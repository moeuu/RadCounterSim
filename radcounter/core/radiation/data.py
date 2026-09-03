"""Fail-closed loaders for versioned radiation physics data."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from radcounter.core.models.radiation import (
    DetectorSpec,
    EmissionLine,
    IsotopeSpec,
    MaterialSpec,
)
from radcounter.core.radiation.scatter import LoadedBuildupData, load_buildup_data

PhysicsDataKind = Literal["material", "isotope", "detector"]
PhysicsValue = MaterialSpec | IsotopeSpec | DetectorSpec


class PhysicsDataError(ValueError):
    """Raised when research physics data are missing, stale, or inconsistent."""


@dataclass(frozen=True)
class LoadedPhysicsData:
    """One validated physics object plus its auditable evidence."""

    kind: PhysicsDataKind
    data_id: str
    data_status: str
    provenance: dict[str, Any]
    payload_sha256: str
    source_path: Path
    value: PhysicsValue


@dataclass(frozen=True)
class ValidatedPhysicsBundle:
    """A mutually compatible, evidence-checked set of physics inputs.

    The tuple representation is intentional: callers cannot silently replace a
    validated entry after the bundle digest has been computed.
    """

    materials: tuple[LoadedPhysicsData, ...]
    isotopes: tuple[LoadedPhysicsData, ...]
    detectors: tuple[LoadedPhysicsData, ...]
    buildup_correction: LoadedBuildupData
    emission_energy_range_keV: tuple[float, float]
    bundle_sha256: str


def _mapping(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PhysicsDataError(f"physics data root must be a mapping: {source_path}")
    return source_path, payload


def _payload_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _evidence(
    source_path: Path,
    payload: dict[str, Any],
    numeric_payload: dict[str, Any],
    *,
    required_status: str | None,
) -> tuple[str, dict[str, Any], str]:
    status = str(payload.get("data_status", ""))
    if not status:
        raise PhysicsDataError(f"data_status is required: {source_path}")
    if required_status is not None and status != required_status:
        raise PhysicsDataError(
            f"{source_path} has data_status={status!r}; required {required_status!r}"
        )
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, dict):
        raise PhysicsDataError(f"provenance must be a mapping: {source_path}")
    actual_hash = _payload_hash(numeric_payload)
    expected_hash = str(provenance.get("payload_sha256", ""))
    if status != "synthetic_validation_only":
        for key in ("source_name", "retrieved_on", "payload_sha256"):
            if provenance.get(key) in (None, ""):
                raise PhysicsDataError(f"provenance.{key} is required: {source_path}")
        if expected_hash != actual_hash:
            raise PhysicsDataError(
                f"numeric payload digest mismatch for {source_path}: "
                f"expected {expected_hash}, computed {actual_hash}"
            )
    return status, provenance, actual_hash


def load_material_data(
    path: str | Path,
    *,
    required_status: str | None = None,
) -> LoadedPhysicsData:
    """Load a material table and verify provenance, units, and density conversion."""

    source_path, payload = _mapping(path)
    numeric_payload = {
        "energies_keV": payload.get("energies_keV"),
        "linear_attenuation_m_inv": payload.get("linear_attenuation_m_inv"),
    }
    status, provenance, digest = _evidence(
        source_path, payload, numeric_payload, required_status=required_status
    )
    energies = np.asarray(numeric_payload["energies_keV"], dtype=np.float64)
    attenuation = np.asarray(numeric_payload["linear_attenuation_m_inv"], dtype=np.float64)
    if status != "synthetic_validation_only" and np.any(attenuation <= 0.0):
        raise PhysicsDataError(f"reference attenuation values must be positive: {source_path}")
    mass_values = payload.get("mass_attenuation_cm2_g")
    density = provenance.get("density_g_cm3")
    if mass_values is not None and density is not None:
        converted = np.asarray(mass_values, dtype=np.float64) * float(density) * 100.0
        if not np.allclose(converted, attenuation, rtol=2.0e-7, atol=1.0e-9):
            raise PhysicsDataError(f"mass-to-linear attenuation conversion mismatch: {source_path}")
    spec = MaterialSpec(
        str(payload["material_id"]),
        energies,
        attenuation,
        str(payload.get("geometry_mode", "solid")),
        payload.get("explicit_thickness_m"),
    )
    return LoadedPhysicsData(
        "material", spec.material_id, status, provenance, digest, source_path, spec
    )


def load_isotope_data(
    path: str | Path,
    *,
    required_status: str | None = None,
) -> LoadedPhysicsData:
    """Load a versioned isotope emission-line dataset."""

    source_path, payload = _mapping(path)
    numeric_payload = {"emission_lines": payload.get("emission_lines")}
    status, provenance, digest = _evidence(
        source_path, payload, numeric_payload, required_status=required_status
    )
    lines = tuple(
        EmissionLine(float(line["energy_keV"]), float(line["photons_per_decay"]))
        for line in payload["emission_lines"]
    )
    spec = IsotopeSpec(str(payload["isotope_id"]), lines)
    return LoadedPhysicsData(
        "isotope", spec.isotope_id, status, provenance, digest, source_path, spec
    )


def load_detector_data(
    path: str | Path,
    *,
    required_status: str | None = None,
) -> LoadedPhysicsData:
    """Load an effective-area detector response without extrapolation fallbacks."""

    source_path, payload = _mapping(path)
    numeric_payload = {
        "energy_bin_edges_keV": payload.get("energy_bin_edges_keV"),
        "response_energy_keV": payload.get("response_energy_keV"),
        "effective_area_m2_per_bin": payload.get("effective_area_m2_per_bin"),
        "background_cps_per_bin": payload.get("background_cps_per_bin"),
        "dead_time_s": payload.get("dead_time_s", 0.0),
    }
    status, provenance, digest = _evidence(
        source_path, payload, numeric_payload, required_status=required_status
    )
    spec = DetectorSpec(
        str(payload["detector_id"]),
        np.asarray(numeric_payload["energy_bin_edges_keV"], dtype=np.float64),
        np.asarray(numeric_payload["response_energy_keV"], dtype=np.float64),
        np.asarray(numeric_payload["effective_area_m2_per_bin"], dtype=np.float64),
        np.asarray(numeric_payload["background_cps_per_bin"], dtype=np.float64),
        float(numeric_payload["dead_time_s"]),
        (
            None
            if payload.get("dose_conversion_sv_h_per_cps") is None
            else np.asarray(payload["dose_conversion_sv_h_per_cps"], dtype=np.float64)
        ),
    )
    return LoadedPhysicsData(
        "detector", spec.detector_id, status, provenance, digest, source_path, spec
    )


def _require_nonempty_unique(
    entries: tuple[LoadedPhysicsData, ...], *, kind: PhysicsDataKind
) -> None:
    if not entries:
        raise PhysicsDataError(f"at least one {kind} dataset is required")
    identifiers = [entry.data_id for entry in entries]
    duplicates = sorted({data_id for data_id in identifiers if identifiers.count(data_id) > 1})
    if duplicates:
        raise PhysicsDataError(f"duplicate {kind} identifiers: {', '.join(duplicates)}")


def validate_research_evaluation_data(
    *,
    material_paths: Iterable[str | Path],
    isotope_paths: Iterable[str | Path],
    detector_paths: Iterable[str | Path],
    buildup_path: str | Path,
    detector_required_status: str = "calibrated_measurement",
) -> ValidatedPhysicsBundle:
    """Load and cross-check every physics input for a research evaluation.

    This is the fail-closed entry point for paper experiments. Reference
    material and isotope data are mandatory, detector evidence must have the
    explicitly requested status, and no material or detector response is
    extrapolated beyond its tabulated energy range. A detector also must have
    non-zero sensitivity to every modeled emission line.
    """

    materials = tuple(
        load_material_data(path, required_status="authoritative_reference")
        for path in material_paths
    )
    isotopes = tuple(
        load_isotope_data(path, required_status="authoritative_reference")
        for path in isotope_paths
    )
    detectors = tuple(
        load_detector_data(path, required_status=detector_required_status)
        for path in detector_paths
    )
    try:
        buildup = load_buildup_data(
            buildup_path,
            required_status="reference_calibrated_correction",
        )
    except ValueError as error:
        raise PhysicsDataError(str(error)) from error
    _require_nonempty_unique(materials, kind="material")
    _require_nonempty_unique(isotopes, kind="isotope")
    _require_nonempty_unique(detectors, kind="detector")

    emission_energies: list[float] = []
    for entry in isotopes:
        isotope = entry.value
        if not isinstance(isotope, IsotopeSpec):  # pragma: no cover - construction invariant
            raise PhysicsDataError(f"invalid isotope entry: {entry.source_path}")
        emission_energies.extend(
            line.energy_keV for line in isotope.emission_lines if line.photons_per_decay > 0.0
        )
    if not emission_energies:
        raise PhysicsDataError("at least one positive-yield emission line is required")

    for entry in materials:
        material = entry.value
        if not isinstance(material, MaterialSpec):  # pragma: no cover - construction invariant
            raise PhysicsDataError(f"invalid material entry: {entry.source_path}")
        minimum = float(material.energies_keV[0])
        maximum = float(material.energies_keV[-1])
        uncovered = [energy for energy in emission_energies if energy < minimum or energy > maximum]
        if uncovered:
            formatted = ", ".join(f"{energy:g}" for energy in sorted(set(uncovered)))
            raise PhysicsDataError(
                f"material {material.material_id!r} does not cover emission energies "
                f"{formatted} keV; valid range is [{minimum:g}, {maximum:g}] keV"
            )

    for entry in detectors:
        detector = entry.value
        if not isinstance(detector, DetectorSpec):  # pragma: no cover - construction invariant
            raise PhysicsDataError(f"invalid detector entry: {entry.source_path}")
        for energy in emission_energies:
            try:
                effective_area = detector.effective_area_m2_at(energy)
            except ValueError as exc:
                raise PhysicsDataError(
                    f"detector {detector.detector_id!r} does not cover the "
                    f"{energy:g} keV emission line"
                ) from exc
            if not np.any(effective_area > 0.0):
                raise PhysicsDataError(
                    f"detector {detector.detector_id!r} has zero response at "
                    f"{energy:g} keV"
                )

    material_ids = {entry.data_id for entry in materials}
    buildup_ids = {surface.material_id for surface in buildup.surfaces}
    if buildup_ids != material_ids:
        missing = sorted(material_ids - buildup_ids)
        extra = sorted(buildup_ids - material_ids)
        raise PhysicsDataError(
            "buildup material coverage does not match transport materials: "
            f"missing={missing}, extra={extra}"
        )
    for surface in buildup.surfaces:
        uncovered = [
            energy
            for energy in emission_energies
            if energy < surface.energies_keV[0] or energy > surface.energies_keV[-1]
        ]
        if uncovered:
            raise PhysicsDataError(
                f"buildup data for {surface.material_id!r} do not cover all emission energies"
            )

    digest_payload = [
        {
            "kind": entry.kind,
            "data_id": entry.data_id,
            "data_status": entry.data_status,
            "payload_sha256": entry.payload_sha256,
        }
        for entry in (*materials, *isotopes, *detectors)
    ]
    digest_payload.append(
        {
            "kind": "buildup_correction",
            "data_status": buildup.data_status,
            "payload_sha256": buildup.payload_sha256,
        }
    )
    bundle_digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return ValidatedPhysicsBundle(
        materials=materials,
        isotopes=isotopes,
        detectors=detectors,
        buildup_correction=buildup,
        emission_energy_range_keV=(min(emission_energies), max(emission_energies)),
        bundle_sha256=bundle_digest,
    )
