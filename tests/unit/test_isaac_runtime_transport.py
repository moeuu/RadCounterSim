import importlib
from pathlib import Path

import numpy as np
import pytest

import radcounter

ROOT = Path(__file__).resolve().parents[2]
EXTENSION_NAMESPACE = str(ROOT / "source/extensions/radcounter.isaac/radcounter")
if EXTENSION_NAMESPACE not in radcounter.__path__:
    radcounter.__path__.append(EXTENSION_NAMESPACE)

_runtime = importlib.import_module("radcounter.isaac.runtime.simulation")
_ensure_material_columns = _runtime._ensure_material_columns
RuntimeConfiguration = _runtime.RuntimeConfiguration
RuntimeDetector = _runtime.RuntimeDetector
NativeStageTransport = _runtime.NativeStageTransport
IsaacRadiationSimulation = _runtime.IsaacRadiationSimulation
RuntimeSource = _runtime.RuntimeSource
_usd_row_matrix_values = importlib.import_module(
    "radcounter.isaac.usd.environment"
)._usd_row_matrix_values


def test_absent_configured_materials_are_zero_padded() -> None:
    paths = _ensure_material_columns(np.asarray([[0.4, 0.2]]), material_count=3)
    np.testing.assert_allclose(paths, [[0.4, 0.2, 0.0]])


def test_native_scene_cannot_return_more_materials_than_configured() -> None:
    with pytest.raises(ValueError, match="4 material columns"):
        _ensure_material_columns(np.ones((2, 4)), material_count=3)


def test_native_usd_reference_transposes_column_vector_transform() -> None:
    matrix = np.asarray(
        [
            [0.0, -0.001, 0.0, -35.0],
            [0.001, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.001, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    rows = np.asarray(_usd_row_matrix_values(matrix))
    np.testing.assert_allclose(rows, matrix.T)
    np.testing.assert_allclose(rows[3, :3], (-35.0, 2.0, 3.0))


def test_runtime_detector_uses_binned_effective_area_and_rejects_extrapolation() -> None:
    detector = RuntimeDetector(
        detector_id="d",
        energy_bin_edges_keV=np.asarray((0.0, 500.0, 1000.0)),
        response_energy_keV=np.asarray((100.0, 1000.0)),
        effective_area_m2_per_bin=np.asarray(((0.02, 0.0), (0.0, 0.01))),
        background_cps_per_bin=np.asarray((0.1, 0.2)),
        dead_time_s=0.0,
    )
    np.testing.assert_allclose(detector.effective_area_m2(np.asarray((550.0,))), [[0.01, 0.005]])
    with pytest.raises(ValueError, match="does not cover"):
        detector.effective_area_m2(np.asarray((1001.0,)))


def test_vertical_slice_runtime_uses_canonical_detector_response_schema() -> None:
    configuration = RuntimeConfiguration.from_json(
        ROOT / "configs/scenarios/vertical_slice.runtime.json"
    )
    detector = configuration.detectors["gamma-counter"]
    assert detector.energy_bin_count == 4
    assert detector.effective_area_m2_per_bin.shape == (4, 4)
    assert configuration.buildup_model.model_name == "primary_only"


def _assert_closed_outward(vertices: np.ndarray, triangles: np.ndarray) -> None:
    edge_counts: dict[tuple[int, int], int] = {}
    for triangle in triangles:
        for first, second in zip(triangle, np.roll(triangle, -1), strict=True):
            edge = tuple(sorted((int(first), int(second))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    assert set(edge_counts.values()) == {2}
    points = vertices[triangles]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    centroids = points.mean(axis=1)
    assert np.all(np.einsum("ij,ij->i", normals, centroids) > 0.0)


def test_sphere_transport_tessellation_is_closed_and_outward() -> None:
    vertices, triangles = NativeStageTransport._sphere_mesh_values(
        1.25, radial_segments=16, polar_segments=8
    )
    _assert_closed_outward(vertices, triangles)
    np.testing.assert_allclose(vertices.min(axis=0), (-1.25, -1.25, -1.25), atol=1e-12)
    np.testing.assert_allclose(vertices.max(axis=0), (1.25, 1.25, 1.25), atol=1e-12)


@pytest.mark.parametrize(
    ("axis", "expected_extent"),
    [("X", (2.0, 0.5, 0.5)), ("Y", (0.5, 2.0, 0.5)), ("Z", (0.5, 0.5, 2.0))],
)
def test_cylinder_transport_tessellation_tracks_usd_axis(
    axis: str, expected_extent: tuple[float, float, float]
) -> None:
    vertices, triangles = NativeStageTransport._cylinder_mesh_values(
        0.5, 4.0, radial_segments=16, axis=axis
    )
    _assert_closed_outward(vertices, triangles)
    np.testing.assert_allclose(vertices.max(axis=0), expected_extent, atol=1e-12)


def test_runtime_source_common_representation_conserves_activity() -> None:
    source = RuntimeSource(
        prim_path="/World/Source",
        isotope_id="Cs-137",
        source_type="volume",
        positions_m=np.asarray(((0.0, 0.0, 0.0), (0.1, 0.0, 0.0))),
        activity_bq=np.asarray((3.0, 7.0)),
        element_indices=np.asarray((0, 1)),
        hidden_from_estimator=False,
    )
    assert source.total_activity_bq == 10.0


def test_volume_voxels_must_lie_inside_visible_primitive() -> None:
    points = np.asarray(((0.0, 0.0, 0.0), (0.9, 0.0, 0.0)))
    assert IsaacRadiationSimulation._volume_points_inside(points, ("sphere", (1.0,)))
    assert not IsaacRadiationSimulation._volume_points_inside(points, ("sphere", (0.5,)))
    assert IsaacRadiationSimulation._volume_points_inside(
        np.asarray(((0.0, 1.5, 0.2),)), ("cylinder", (0.5, 2.0, 1.0))
    )
