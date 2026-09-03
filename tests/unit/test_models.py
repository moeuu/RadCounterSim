import numpy as np
import pytest

from radcounter.core.models import DetectorSpec, EmissionLine, IsotopeSpec, PointSourceState


def test_point_source_rejects_negative_activity() -> None:
    with pytest.raises(ValueError):
        PointSourceState("s", np.zeros(3), -1.0, "i")


def test_detector_requires_background_per_bin() -> None:
    with pytest.raises(ValueError):
        DetectorSpec(
            "d",
            np.array([0.0, 100.0, 200.0]),
            np.array([1.0, 200.0]),
            np.ones((2, 2)),
            np.array([0.0]),
        )


def test_detector_rejects_response_energy_extrapolation() -> None:
    detector = DetectorSpec(
        "d",
        np.array([0.0, 200.0]),
        np.array([10.0, 100.0]),
        np.ones((2, 1)),
        np.array([0.0]),
    )
    with pytest.raises(ValueError, match="outside detector response range"):
        detector.effective_area_m2_at(101.0)


def test_isotope_requires_emission_lines() -> None:
    with pytest.raises(ValueError):
        IsotopeSpec("empty", ())
    assert IsotopeSpec("ok", (EmissionLine(100.0, 1.0),)).isotope_id == "ok"
