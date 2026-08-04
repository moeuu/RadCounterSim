import numpy as np

from radcounter.core.radiation.partial_cache import (
    AxisAlignedBounds,
    SelectiveRayResponseCache,
    TransferRayLayout,
)


def test_aabb_patch_recomputes_only_intersecting_segments() -> None:
    origins = np.array([[-1.0, 0.0, 0.0], [-1.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    targets = np.array([[1.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    cache = SelectiveRayResponseCache(origins, targets, np.ones((3, 2)))
    calls: list[int] = []

    def recompute(selected_origins: np.ndarray, _selected_targets: np.ndarray) -> np.ndarray:
        calls.append(len(selected_origins))
        return np.full((len(selected_origins), 2), 0.25)

    indexes = cache.patch_bounds(
        AxisAlignedBounds(np.array([-0.1, -0.1, -0.1]), np.array([0.1, 0.1, 0.1])),
        recompute,
        geometry_revision=1,
    )
    assert indexes.tolist() == [0]
    assert calls == [1]
    np.testing.assert_allclose(cache.response[0], 0.25)
    np.testing.assert_allclose(cache.response[1:], 1.0)


def test_activity_projection_uses_cached_transfer_without_recompute() -> None:
    layout, origins, targets = TransferRayLayout.build_segments(
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
    )
    response = np.array([[1.0], [0.5], [0.25], [2.0]])
    cache = SelectiveRayResponseCache(origins, targets, response, layout=layout)
    first = cache.expected_detector_response(np.array([10.0, 20.0]))
    second = cache.expected_detector_response(np.array([5.0, 4.0]))
    np.testing.assert_allclose(first[:, 0], [20.0, 42.5])
    np.testing.assert_allclose(second[:, 0], [7.0, 9.25])
    assert cache.patch_count == 0
