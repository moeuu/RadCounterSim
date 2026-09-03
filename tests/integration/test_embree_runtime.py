import importlib
from pathlib import Path

import numpy as np
import pytest

import radcounter
from radcounter.core.radiation.embree_native import (
    EmbreeNativeScene,
    TriangleMesh,
    native_embree_available,
)

ROOT = Path(__file__).resolve().parents[2]
EXTENSION_NAMESPACE = str(ROOT / "source/extensions/radcounter.isaac/radcounter")
if EXTENSION_NAMESPACE not in radcounter.__path__:
    radcounter.__path__.append(EXTENSION_NAMESPACE)
NativeStageTransport = importlib.import_module(
    "radcounter.isaac.runtime.simulation"
).NativeStageTransport

pytestmark = pytest.mark.skipif(
    not native_embree_available(),
    reason="the optional Embree native module is not built",
)


def _unit_cube() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    )
    triangles = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
        ],
        dtype=np.uint32,
    )
    return vertices, triangles


def test_dynamic_solid_transform_and_removal() -> None:
    vertices, triangles = _unit_cube()
    scene = EmbreeNativeScene()
    geometry_id = scene.add_mesh(TriangleMesh(vertices, triangles, 0, mesh_id="cube"))
    scene.commit()
    origin = np.array([[-1.0, 0.5, 0.5]])
    target = np.array([[2.0, 0.5, 0.5]])
    assert scene.trace_path_lengths(origin, target)[0, 0] == pytest.approx(1.0, abs=2e-4)
    translation = np.eye(4)
    translation[1, 3] = 2.0
    scene.update_instance_transform(geometry_id, translation)
    scene.commit()
    assert scene.trace_path_lengths(origin, target)[0, 0] == pytest.approx(0.0)
    scene.remove_geometry(geometry_id)
    scene.commit()
    assert scene.trace_path_lengths(origin, target).shape == (1, 1)
    assert scene.trace_path_lengths(origin, target)[0, 0] == pytest.approx(0.0)


def test_thin_sheet_uses_angle_corrected_explicit_thickness() -> None:
    vertices = np.array([[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 1.0, 1.0], [0.0, -1.0, 1.0]])
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    scene = EmbreeNativeScene()
    scene.add_mesh(
        TriangleMesh(
            vertices,
            triangles,
            0,
            geometry_mode="thin_sheet",
            explicit_thickness_m=0.1,
        )
    )
    scene.commit()
    normal_path = scene.trace_path_lengths(
        np.array([[-1.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])
    )[0, 0]
    oblique_path = scene.trace_path_lengths(
        np.array([[-1.0, -0.5, 0.0]]), np.array([[1.0, 0.5, 0.0]])
    )[0, 0]
    assert normal_path == pytest.approx(0.1, abs=1e-5)
    assert oblique_path == pytest.approx(0.1 * np.sqrt(1.25), rel=2e-4)


def test_packet_trace_handles_multiple_full_packets_and_tail() -> None:
    vertices, triangles = _unit_cube()
    scene = EmbreeNativeScene()
    scene.add_mesh(TriangleMesh(vertices, triangles, 0, mesh_id="packet-cube"))
    scene.commit()
    coordinates = np.linspace(0.1, 0.9, 17)
    origins = np.column_stack((np.full(17, -1.0), coordinates, np.full(17, 0.5)))
    targets = np.column_stack((np.full(17, 2.0), coordinates, np.full(17, 0.5)))
    paths = scene.trace_path_lengths(origins, targets)
    assert paths.shape == (17, 1)
    assert paths[:, 0] == pytest.approx(np.ones(17), abs=2e-4)


@pytest.mark.parametrize("shape", ["sphere", "cylinder"])
def test_generated_usd_primitive_meshes_trace_outside_and_inside_paths(shape: str) -> None:
    if shape == "sphere":
        vertices, triangles = NativeStageTransport._sphere_mesh_values(1.0)
    else:
        vertices, triangles = NativeStageTransport._cylinder_mesh_values(1.0, 2.0)
    scene = EmbreeNativeScene()
    scene.add_mesh(TriangleMesh(vertices, triangles, 0, mesh_id=shape))
    scene.commit()
    paths = scene.trace_path_lengths(
        np.array([[-2.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        np.array([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )
    assert paths[:, 0] == pytest.approx((2.0, 1.0), abs=3e-4)


def test_generated_sphere_respects_nonuniform_instance_transform() -> None:
    vertices, triangles = NativeStageTransport._sphere_mesh_values(1.0)
    transform = np.diag((2.0, 1.0, 0.5, 1.0))
    scene = EmbreeNativeScene()
    scene.add_mesh(
        TriangleMesh(vertices, triangles, 0, mesh_id="ellipsoid", world_transform=transform)
    )
    scene.commit()
    path = scene.trace_path_lengths(
        np.array([[-3.0, 0.0, 0.0]]), np.array([[3.0, 0.0, 0.0]])
    )[0, 0]
    assert path == pytest.approx(4.0, abs=5e-4)
