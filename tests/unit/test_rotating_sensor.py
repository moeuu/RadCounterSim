from pathlib import Path

import numpy as np
import pytest

from radcounter.core.models import DetectorSpec
from radcounter.core.radiation.rng import SeedManager
from radcounter.core.sensors import (
    DoseRateMeter,
    RotatingShieldConfiguration,
    RotatingShieldCounter,
    RotatingShieldMode,
    ShieldProgram,
)

ROOT = Path(__file__).resolve().parents[2]


def _detector() -> DetectorSpec:
    return DetectorSpec(
        "rotating",
        np.array([0.0, 100.0, 200.0]),
        np.array([1.0, 200.0]),
        np.eye(2),
        np.array([0.0, 0.0]),
        dose_conversion_sv_h_per_cps=np.array([1.0e-9, 2.0e-9]),
    )


def test_response_mask_program_changes_expected_counts() -> None:
    sensor = RotatingShieldCounter(
        _detector(),
        SeedManager(3).generator("rotating"),
        mode=RotatingShieldMode.RESPONSE_MASK,
        response_mask_per_posture_bin=np.array([[1.0, 0.5], [0.25, 1.0]]),
    )
    measurement = sensor.measure_program(
        ShieldProgram(np.array([0.0, 90.0]), np.array([2.0, 2.0])),
        unshielded_rate_cps_per_bin=np.array([10.0, 20.0]),
    )
    assert np.array_equal(
        measurement.expected_counts_per_posture_bin,
        np.array([[20.0, 20.0], [5.0, 40.0]]),
    )


def test_physical_mode_calls_rate_provider_at_each_encoder_angle() -> None:
    sensor = RotatingShieldCounter(
        _detector(),
        SeedManager(4).generator("physical"),
        mode=RotatingShieldMode.PHYSICAL_GEOMETRY,
    )
    measurement = sensor.measure_program(
        ShieldProgram(np.array([0.0, 180.0]), np.ones(2)),
        physical_rate_provider=lambda angle_deg: np.array([10.0 + angle_deg / 180.0, 1.0]),
    )
    assert np.allclose(measurement.expected_counts_per_posture_bin[:, 0], [10.0, 11.0])
    assert np.array_equal(measurement.actual_angles_deg, [0.0, 180.0])


def test_physical_mode_separates_actual_posture_from_encoder_error() -> None:
    seed = 12
    reference = np.random.default_rng(seed)
    commanded = np.array([0.0, 90.0])
    expected_actual = commanded + reference.normal(0.0, 0.4, size=2)
    expected_encoder = expected_actual + reference.normal(0.0, 0.7, size=2)
    evaluated_angles: list[float] = []
    sensor = RotatingShieldCounter(
        _detector(),
        np.random.default_rng(seed),
        mode=RotatingShieldMode.PHYSICAL_GEOMETRY,
        posture_error_std_deg=0.4,
        encoder_noise_std_deg=0.7,
    )
    measurement = sensor.measure_program(
        ShieldProgram(commanded, np.ones(2)),
        physical_rate_provider=lambda angle_deg: (
            evaluated_angles.append(angle_deg) or np.array([1.0, 1.0])
        ),
    )
    np.testing.assert_allclose(measurement.actual_angles_deg, expected_actual)
    np.testing.assert_allclose(measurement.encoder_angles_deg, expected_encoder)
    np.testing.assert_allclose(evaluated_angles, expected_actual)


def test_physical_rotating_shield_configuration_loads_scene_binding() -> None:
    configuration = RotatingShieldConfiguration.from_yaml(
        ROOT / "configs/detectors/rotating_shield_counter.physical.synthetic.yaml"
    )
    assert configuration.mode is RotatingShieldMode.PHYSICAL_GEOMETRY
    assert configuration.data_status == "synthetic_validation_only"
    assert configuration.response_mask_per_posture_bin is None
    assert configuration.physical_geometry is not None
    assert configuration.physical_geometry.rotation_attribute == "xformOp:rotateZ"
    assert configuration.detector.energy_bin_count == 4


def test_response_mask_configuration_remains_a_separate_approximation() -> None:
    configuration = RotatingShieldConfiguration.from_yaml(
        ROOT / "configs/detectors/rotating_shield_counter.yaml"
    )
    assert configuration.mode is RotatingShieldMode.RESPONSE_MASK
    assert configuration.physical_geometry is None
    assert configuration.response_mask_per_posture_bin is not None


def test_physical_mode_rejects_a_hidden_response_mask() -> None:
    with pytest.raises(ValueError, match="cannot accept a response mask"):
        RotatingShieldCounter(
            _detector(),
            np.random.default_rng(5),
            mode=RotatingShieldMode.PHYSICAL_GEOMETRY,
            response_mask_per_posture_bin=np.ones((2, 2)),
        )


def test_dose_rate_meter_uses_bin_conversion() -> None:
    meter = DoseRateMeter(_detector())
    assert np.isclose(meter.dose_rate_sv_h(np.array([10.0, 20.0])), 5.0e-8)
