import numpy as np

from radcounter.core.models import DetectorSpec
from radcounter.core.radiation.rng import SeedManager
from radcounter.core.sensors import OmnidirectionalCounter


def _measure(seed: int) -> np.ndarray:
    detector = DetectorSpec(
        "d",
        np.array([0.0, 100.0]),
        np.array([1.0, 100.0]),
        np.array([1.0, 1.0]),
        np.array([0.0]),
    )
    sensor = OmnidirectionalCounter(detector, SeedManager(seed).generator("detector/d"))
    sensor.start_measurement(
        timestamp_sim_s=0.0,
        duration_s=1.0,
        position_world_m=np.zeros(3),
        orientation_world_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        expected_rate_cps_per_bin=np.array([100.0]),
        scene_revision=0,
    )
    sensor.update(1.0)
    measurement = sensor.get_latest()
    assert measurement is not None
    return measurement.counts_per_bin


def test_poisson_seed_is_reproducible() -> None:
    assert np.array_equal(_measure(7), _measure(7))
