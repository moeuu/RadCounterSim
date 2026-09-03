import numpy as np
import pytest

from radcounter.core.models import (
    DetectorSpec,
    EmissionLine,
    IsotopeSpec,
    MaterialSpec,
    PointSourceState,
)
from radcounter.core.radiation import (
    AnalyticSlab,
    AnalyticTransportBackend,
    MaterialTable,
    RadiationForwardModel,
)


def _detector() -> DetectorSpec:
    return DetectorSpec(
        "d",
        np.array([0.0, 200.0]),
        np.array([1.0, 100.0, 200.0]),
        np.ones((3, 1)),
        np.array([0.0]),
    )


def test_point_source_inverse_square() -> None:
    backend = AnalyticTransportBackend()
    backend.build_scene([], MaterialTable(()))
    model = RadiationForwardModel(backend)
    source = PointSourceState("s", np.zeros(3), 1.0e6, "i")
    isotope = {"i": IsotopeSpec("i", (EmissionLine(100.0, 1.0),))}
    prediction = model.predict_point_count_rate(
        np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), [source], isotope, _detector()
    )
    assert np.isclose(
        prediction.direct_count_rate_cps_per_bin[0, 0]
        / prediction.direct_count_rate_cps_per_bin[1, 0],
        4.0,
        rtol=1e-12,
    )


def test_detector_effective_area_closes_count_rate_units() -> None:
    backend = AnalyticTransportBackend()
    backend.build_scene([], MaterialTable(()))
    detector = DetectorSpec(
        "area",
        np.array([0.0, 200.0]),
        np.array([50.0, 150.0]),
        np.array([[0.02], [0.02]]),
        np.array([0.0]),
    )
    prediction = RadiationForwardModel(backend).predict_point_count_rate(
        np.array([[1.0, 0.0, 0.0]]),
        [PointSourceState("s", np.zeros(3), 1.0e6, "i")],
        {"i": IsotopeSpec("i", (EmissionLine(100.0, 1.0),))},
        detector,
    )
    assert np.isclose(prediction.direct_count_rate_cps_per_bin[0, 0], 1.0e6 * 0.02 / (4 * np.pi))


def test_single_slab_matches_exponential() -> None:
    material = MaterialSpec("m", np.array([10.0, 1000.0]), np.array([2.0, 2.0]))
    backend = AnalyticTransportBackend()
    backend.build_scene(
        [AnalyticSlab("slab", "m", np.zeros(3), np.array([1.0, 0.0, 0.0]), 0.5)],
        MaterialTable((material,)),
    )
    transmission = backend.trace_transmission(
        np.array([[-1.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]), np.array([100.0])
    )
    assert np.isclose(transmission[0, 0], np.exp(-1.0), rtol=1e-12)


def test_material_interpolation_is_positive() -> None:
    material = MaterialSpec("m", np.array([10.0, 1000.0]), np.array([10.0, 1.0]))
    value = MaterialTable((material,)).attenuation_m_inv("m", np.array([100.0]))
    assert np.isclose(value[0], np.sqrt(10.0))


def test_material_interpolation_refuses_extrapolation() -> None:
    material = MaterialSpec("m", np.array([10.0, 1000.0]), np.array([10.0, 1.0]))
    with pytest.raises(ValueError, match="does not cover"):
        MaterialTable((material,)).attenuation_m_inv("m", np.array([1001.0]))
