import numpy as np

from radcounter.core.models import DetectorSpec, EmissionLine, IsotopeSpec, RevisionState
from radcounter.core.radiation import AnalyticTransportBackend, MaterialTable
from radcounter.core.radiation.dose_map import DoseMapEvaluator
from radcounter.core.radiation.sampled_forward import SampledRadiationForwardModel
from radcounter.core.radiation.sampling import point_sample_batch
from radcounter.core.radiation.transfer import (
    TransferMatrixCache,
    TransferMatrixKey,
    evaluate_linear_counts,
)


def _model_and_detector() -> tuple[SampledRadiationForwardModel, DetectorSpec]:
    backend = AnalyticTransportBackend()
    backend.build_scene([], MaterialTable(()))
    detector = DetectorSpec(
        "dose",
        np.array([0.0, 200.0]),
        np.array([1.0, 100.0, 200.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.0]),
        dose_conversion_sv_h_per_cps=np.array([1.0e-9]),
    )
    return SampledRadiationForwardModel(backend), detector


def _basis() -> object:
    return point_sample_batch(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([1000.0]),
        np.array([0]),
        np.array([0]),
    )


def test_transfer_matrix_linear_activity_update() -> None:
    model, detector = _model_and_detector()
    basis = _basis()
    isotopes = (IsotopeSpec("i", (EmissionLine(100.0, 1.0),)),)
    matrix = model.build_transfer_matrix(np.array([[1.0, 0.0, 0.0]]), basis, isotopes, detector)
    count_rate = evaluate_linear_counts(matrix, np.array([1000.0]))
    direct = model.predict_count_rate(
        np.array([[1.0, 0.0, 0.0]]), basis, isotopes, detector
    ).direct_count_rate_cps_per_bin.ravel()
    assert np.allclose(count_rate, direct)


def test_activity_revision_does_not_invalidate_transfer_key() -> None:
    detector_poses = np.array([[1.0, 0.0, 0.0]])
    basis_positions = np.array([[0.0, 0.0, 0.0]])
    energy = np.array([100.0])
    before = RevisionState(source_activity_revision=0)
    after = RevisionState(source_activity_revision=10)
    first = TransferMatrixKey.from_arrays(
        detector_poses=detector_poses,
        basis_positions=basis_positions,
        revision=before,
        energy_grid_keV=energy,
    )
    second = TransferMatrixKey.from_arrays(
        detector_poses=detector_poses,
        basis_positions=basis_positions,
        revision=after,
        energy_grid_keV=energy,
    )
    assert first == second


def test_geometry_revision_changes_key_and_cache_tracks_hits() -> None:
    poses = np.array([[1.0, 0.0, 0.0]])
    basis = np.array([[0.0, 0.0, 0.0]])
    energy = np.array([100.0])
    key = TransferMatrixKey.from_arrays(
        detector_poses=poses,
        basis_positions=basis,
        revision=RevisionState(),
        energy_grid_keV=energy,
    )
    changed = TransferMatrixKey.from_arrays(
        detector_poses=poses,
        basis_positions=basis,
        revision=RevisionState(geometry_revision=1),
        energy_grid_keV=energy,
    )
    cache = TransferMatrixCache(maximum_entries=1)
    cache.put(key, np.ones((1, 1)))
    assert cache.get(key) is not None
    assert cache.get(changed) is None
    assert cache.statistics.hits == 1
    assert cache.statistics.misses == 1


def test_planar_dose_map_obeys_inverse_square() -> None:
    model, detector = _model_and_detector()
    evaluator = DoseMapEvaluator(model)
    grid = evaluator.create_planar_grid((1.0, 2.0, 0.0, 0.0), z_m=0.0, resolution_m=1.0)
    dose_map = evaluator.evaluate(
        grid,
        _basis(),
        (IsotopeSpec("i", (EmissionLine(100.0, 1.0),)),),
        detector,
        RevisionState(),
        chunk_size=1,
    )
    assert np.isclose(dose_map.dose_rate_sv_h[0] / dose_map.dose_rate_sv_h[1], 4.0)
