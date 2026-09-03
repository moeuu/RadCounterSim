import numpy as np
import pytest

from radcounter.core.models import DetectorSpec, EmissionLine, IsotopeSpec, MaterialSpec
from radcounter.core.radiation import AnalyticSlab, AnalyticTransportBackend, MaterialTable
from radcounter.core.radiation.backend import PathLengthBatch
from radcounter.core.radiation.sampled_forward import SampledRadiationForwardModel
from radcounter.core.radiation.sampling import point_sample_batch
from radcounter.core.radiation.scatter import (
    MaterialBuildupSurface,
    ReferenceCalibratedBuildupModel,
)


def _material_table() -> MaterialTable:
    return MaterialTable(
        (
            MaterialSpec(
                "test",
                np.asarray((100.0, 1000.0)),
                np.asarray((2.0, 2.0)),
            ),
        )
    )


def _buildup_model() -> ReferenceCalibratedBuildupModel:
    surface = MaterialBuildupSurface(
        "test",
        np.asarray((100.0, 1000.0)),
        np.asarray((0.0, 1.0, 2.0)),
        np.asarray(((1.0, 1.4, 1.8), (1.0, 1.2, 1.4))),
    )
    return ReferenceCalibratedBuildupModel(_material_table(), (surface,))


def test_buildup_interpolates_energy_and_optical_depth_without_extrapolation() -> None:
    energy = np.asarray((np.sqrt(100.0 * 1000.0),))
    paths = PathLengthBatch(
        ("test",),
        np.asarray(((0.0,), (0.5,))),
        np.asarray((False, False)),
    )
    factors = _buildup_model().factors(paths, energy)
    assert factors[:, 0] == pytest.approx((1.0, 1.3))

    outside = PathLengthBatch(
        ("test",),
        np.asarray(((1.1,),)),
        np.asarray((False,)),
    )
    with pytest.raises(ValueError, match="do not cover traced optical depth"):
        _buildup_model().factors(outside, energy)


def test_sampled_forward_keeps_primary_and_corrected_rates_separate() -> None:
    material_table = _material_table()
    backend = AnalyticTransportBackend()
    backend.build_scene(
        (
            AnalyticSlab(
                "slab",
                "test",
                np.asarray((0.0, 0.0, 0.0)),
                np.asarray((1.0, 0.0, 0.0)),
                0.5,
            ),
        ),
        material_table,
    )
    detector = DetectorSpec(
        "detector",
        np.asarray((0.0, 1000.0)),
        np.asarray((100.0, 1000.0)),
        np.ones((2, 1)),
        np.zeros(1),
    )
    source = point_sample_batch(
        np.asarray(((-1.0, 0.0, 0.0),)),
        np.asarray((1.0e6,)),
        np.asarray((0,)),
        np.asarray((0,)),
    )
    isotope = (
        IsotopeSpec("test", (EmissionLine(np.sqrt(100.0 * 1000.0), 1.0),)),
    )
    prediction = SampledRadiationForwardModel(
        backend,
        buildup_model=_buildup_model(),
    ).predict_count_rate(
        np.asarray(((1.0, 0.0, 0.0),)),
        source,
        isotope,
        detector,
    )
    assert prediction.count_rate_cps_per_bin[0, 0] == pytest.approx(
        prediction.direct_count_rate_cps_per_bin[0, 0] * 1.3
    )
    assert prediction.buildup_model_name == "reference_calibrated_optical_depth_buildup"
