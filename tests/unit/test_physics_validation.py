from pathlib import Path

from radcounter.validation.physics import (
    load_validation_dataset,
    validate_physics_dataset,
)


def _fixture() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2]
    base = root / "configs/validation/examples"
    return (
        base / "synthetic_cs137_lead.csv",
        base / "synthetic_cs137_lead.metadata.yaml",
    )


def test_analytic_source_and_shield_fixture_passes_numerical_gate() -> None:
    observations, metadata = load_validation_dataset(*_fixture())
    result = validate_physics_dataset(observations, metadata)
    assert result.passed
    assert result.evidence_kind == "synthetic_analytic"
    assert result.point_count == 7
    assert result.attenuation_mu_relative_error["lead"] < 0.01


def test_synthetic_fixture_cannot_qualify_as_measured_evidence() -> None:
    observations, metadata = load_validation_dataset(*_fixture())
    result = validate_physics_dataset(
        observations, metadata, require_measured=True
    )
    assert not result.passed
    assert any("physical evidence required" in reason for reason in result.reasons)
