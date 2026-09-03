import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from radcounter.core.models import DetectorSpec, IsotopeSpec, MaterialSpec
from radcounter.core.radiation import (
    PhysicsDataError,
    load_detector_data,
    load_isotope_data,
    load_material_data,
    validate_research_evaluation_data,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("material_id", ["concrete", "lead", "steel"])
def test_authoritative_material_tables_are_positive_and_digest_verified(material_id: str) -> None:
    loaded = load_material_data(
        ROOT / f"configs/materials/{material_id}.yaml",
        required_status="authoritative_reference",
    )
    assert isinstance(loaded.value, MaterialSpec)
    assert loaded.data_id == material_id
    assert np.all(loaded.value.linear_attenuation_m_inv > 0.0)
    assert len(loaded.payload_sha256) == 64


def test_cs137_reference_is_digest_verified() -> None:
    loaded = load_isotope_data(
        ROOT / "configs/isotopes/cs137.yaml",
        required_status="authoritative_reference",
    )
    assert isinstance(loaded.value, IsotopeSpec)
    assert loaded.value.emission_lines[0].energy_keV == 661.657
    assert loaded.value.emission_lines[0].photons_per_decay == 0.851


def test_research_status_rejects_synthetic_detector() -> None:
    with pytest.raises(PhysicsDataError, match="required 'calibrated_measurement'"):
        load_detector_data(
            ROOT / "configs/detectors/omni_counter.yaml",
            required_status="calibrated_measurement",
        )


def test_reference_digest_tampering_fails_closed(tmp_path: Path) -> None:
    source = ROOT / "configs/materials/lead.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["linear_attenuation_m_inv"][3] *= 1.01
    tampered = tmp_path / "lead.yaml"
    tampered.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(PhysicsDataError, match="digest mismatch"):
        load_material_data(tampered, required_status="authoritative_reference")


def test_synthetic_detector_uses_effective_area_matrix() -> None:
    loaded = load_detector_data(ROOT / "configs/detectors/omni_counter.yaml")
    assert isinstance(loaded.value, DetectorSpec)
    assert loaded.value.effective_area_m2_per_bin.shape == (3, 3)


def _write_calibrated_detector(
    path: Path, *, energies: list[float] | None = None, response: list[list[float]] | None = None
) -> Path:
    numeric_payload = {
        "energy_bin_edges_keV": [0.0, 800.0],
        "response_energy_keV": energies or [100.0, 1000.0],
        "effective_area_m2_per_bin": response or [[0.01], [0.02]],
        "background_cps_per_bin": [0.1],
        "dead_time_s": 0.0,
    }
    digest = hashlib.sha256(
        json.dumps(
            numeric_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "detector_id": "test_calibrated_detector",
        "data_status": "calibrated_measurement",
        "provenance": {
            "source_name": "unit-test controlled measurement",
            "retrieved_on": "2026-08-29",
            "payload_sha256": digest,
        },
        **numeric_payload,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_calibrated_buildup(path: Path, material_id: str = "lead") -> Path:
    numeric_payload = {
        "composition_rule": "product_by_material",
        "materials": [
            {
                "material_id": material_id,
                "energies_keV": [100.0, 1000.0],
                "optical_depths": [0.0, 1.0, 20.0],
                "factors_by_energy_depth": [
                    [1.0, 1.2, 2.0],
                    [1.0, 1.1, 1.5],
                ],
            }
        ],
    }
    digest = hashlib.sha256(
        json.dumps(
            numeric_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "data_status": "reference_calibrated_correction",
        "provenance": {
            "source_name": "unit-test independent photon reference",
            "retrieved_on": "2026-08-29",
            "payload_sha256": digest,
        },
        **numeric_payload,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_research_bundle_validates_status_energy_coverage_and_digest(tmp_path: Path) -> None:
    bundle = validate_research_evaluation_data(
        material_paths=[ROOT / "configs/materials/lead.yaml"],
        isotope_paths=[ROOT / "configs/isotopes/cs137.yaml"],
        detector_paths=[_write_calibrated_detector(tmp_path / "detector.yaml")],
        buildup_path=_write_calibrated_buildup(tmp_path / "buildup.yaml"),
    )
    assert bundle.emission_energy_range_keV == (661.657, 661.657)
    assert len(bundle.bundle_sha256) == 64
    assert bundle.detectors[0].data_status == "calibrated_measurement"


def test_research_bundle_rejects_detector_without_emission_coverage(tmp_path: Path) -> None:
    detector = _write_calibrated_detector(
        tmp_path / "detector.yaml",
        energies=[100.0, 600.0],
    )
    with pytest.raises(PhysicsDataError, match="does not cover the 661.657 keV"):
        validate_research_evaluation_data(
            material_paths=[ROOT / "configs/materials/lead.yaml"],
            isotope_paths=[ROOT / "configs/isotopes/cs137.yaml"],
            detector_paths=[detector],
            buildup_path=_write_calibrated_buildup(tmp_path / "buildup.yaml"),
        )


def test_research_bundle_rejects_zero_detector_response(tmp_path: Path) -> None:
    detector = _write_calibrated_detector(
        tmp_path / "detector.yaml",
        response=[[0.0], [0.0]],
    )
    with pytest.raises(PhysicsDataError, match="zero response"):
        validate_research_evaluation_data(
            material_paths=[ROOT / "configs/materials/lead.yaml"],
            isotope_paths=[ROOT / "configs/isotopes/cs137.yaml"],
            detector_paths=[detector],
            buildup_path=_write_calibrated_buildup(tmp_path / "buildup.yaml"),
        )


def test_research_bundle_rejects_missing_buildup_material(tmp_path: Path) -> None:
    with pytest.raises(PhysicsDataError, match="material coverage"):
        validate_research_evaluation_data(
            material_paths=[ROOT / "configs/materials/lead.yaml"],
            isotope_paths=[ROOT / "configs/isotopes/cs137.yaml"],
            detector_paths=[_write_calibrated_detector(tmp_path / "detector.yaml")],
            buildup_path=_write_calibrated_buildup(
                tmp_path / "buildup.yaml", material_id="concrete"
            ),
        )
