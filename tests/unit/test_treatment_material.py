from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from radcounter.core.treatment import load_treatment_material_model

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "configs/decontamination/concrete_surface.synthetic.yaml"


def _digest(path: Path = MODEL_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_treatment_material_is_versioned_and_material_specific() -> None:
    model = load_treatment_material_model(
        MODEL_PATH,
        expected_file_sha256=_digest(),
        expected_substrate_material_id="concrete",
    )
    assert model.status == "synthetic_validation_only"
    assert model.dry_contact.rate_constant_s_inv == pytest.approx(0.9)
    assert model.water_jet.removal_coefficient_m2_per_l == pytest.approx(0.8)
    assert len(model.numeric_sha256) == 64


def test_treatment_material_fails_closed_on_digest_status_and_substrate() -> None:
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_treatment_material_model(MODEL_PATH, expected_file_sha256="0" * 64)
    with pytest.raises(ValueError, match="does not satisfy"):
        load_treatment_material_model(
            MODEL_PATH,
            expected_file_sha256=_digest(),
            required_status="experimentally_calibrated",
        )
    with pytest.raises(ValueError, match="does not match"):
        load_treatment_material_model(
            MODEL_PATH,
            expected_file_sha256=_digest(),
            expected_substrate_material_id="steel",
        )
