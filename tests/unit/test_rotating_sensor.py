import numpy as np

from radcounter.core.models import DetectorSpec
from radcounter.core.radiation.rng import SeedManager
from radcounter.core.sensors import (
    DoseRateMeter,
    RotatingShieldCounter,
    RotatingShieldMode,
    ShieldProgram,
)


def _detector() -> DetectorSpec:
    return DetectorSpec(
        "rotating",
        np.array([0.0, 100.0, 200.0]),
        np.array([1.0, 200.0]),
        np.array([1.0, 1.0]),
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


def test_dose_rate_meter_uses_bin_conversion() -> None:
    meter = DoseRateMeter(_detector())
    assert np.isclose(meter.dose_rate_sv_h(np.array([10.0, 20.0])), 5.0e-8)
