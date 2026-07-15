import numpy as np

from radcounter.core.radiation.sampling import sample_surface_triangles


def _surface() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    triangles = np.array([[0, 1, 2], [0, 2, 3]])
    activity = np.array([20.0, 30.0])
    return vertices, triangles, activity


def test_centroid_sampling_preserves_triangle_activity() -> None:
    vertices, triangles, activity = _surface()
    samples = sample_surface_triangles(
        vertices_world_m=vertices,
        triangles=triangles,
        activity_bq_per_triangle=activity,
        isotope_index=0,
        source_id_index=0,
    )
    assert samples.sample_count == 2
    assert samples.total_activity_bq == 50.0
    assert np.allclose(samples.positions_world_m[0], [2.0 / 3.0, 1.0 / 3.0, 0.0])


def test_stratified_sampling_is_reproducible_and_conservative() -> None:
    vertices, triangles, activity = _surface()
    arguments = dict(
        vertices_world_m=vertices,
        triangles=triangles,
        activity_bq_per_triangle=activity,
        isotope_index=0,
        source_id_index=0,
        mode="stratified",
        samples_per_triangle=5,
    )
    first = sample_surface_triangles(**arguments, rng=np.random.default_rng(9))
    second = sample_surface_triangles(**arguments, rng=np.random.default_rng(9))
    assert first.sample_count == 10
    assert np.array_equal(first.positions_world_m, second.positions_world_m)
    assert np.isclose(first.total_activity_bq, activity.sum())


def test_attached_source_transform_updates_positions_only() -> None:
    vertices, triangles, activity = _surface()
    samples = sample_surface_triangles(
        vertices_world_m=vertices,
        triangles=triangles,
        activity_bq_per_triangle=activity,
        isotope_index=0,
        source_id_index=0,
    )
    transform = np.eye(4)
    transform[:3, 3] = [2.0, -1.0, 0.5]
    moved = samples.transformed(transform)
    assert np.allclose(moved.positions_world_m, samples.positions_world_m + [2.0, -1.0, 0.5])
    assert np.array_equal(moved.activity_bq, samples.activity_bq)
