import sys
from types import SimpleNamespace

import numpy as np
import pytest

from radcounter.core.models import MaterialSpec
from radcounter.core.radiation.embree_native import (
    EmbreeNativeScene,
    EmbreeNativeUnavailable,
    EmbreeTransportBackend,
    RadiationTriangleMesh,
    TriangleMesh,
    native_embree_available,
)
from radcounter.core.radiation.materials import MaterialTable


def test_triangle_mesh_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="outside vertices"):
        TriangleMesh(
            np.zeros((3, 3)),
            np.array([[0, 1, 3]], dtype=np.uint32),
            0,
        )


def test_native_boundary_fails_explicitly_when_embree_is_unavailable() -> None:
    if native_embree_available():
        pytest.skip("native Embree module is available on this host")
    with pytest.raises(EmbreeNativeUnavailable, match="build_native"):
        EmbreeNativeScene()


def test_embree_backend_matches_transport_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeScene:
        def __init__(self) -> None:
            self.revision = 0

        def add_triangle_mesh(
            self,
            vertices_m: np.ndarray,
            triangles: np.ndarray,
            material_index: int,
        ) -> int:
            del vertices_m, triangles, material_index
            return 0

        def commit(self) -> None:
            self.revision += 1

        def trace_transmission(
            self,
            origins_m: np.ndarray,
            targets_m: np.ndarray,
            attenuation_per_m: np.ndarray,
        ) -> np.ndarray:
            del targets_m
            path_lengths_m = np.full(
                (len(origins_m), attenuation_per_m.shape[0]),
                0.5,
            )
            return np.exp(-(path_lengths_m @ attenuation_per_m))

    monkeypatch.setitem(sys.modules, "_radcounter_embree", SimpleNamespace(Scene=FakeScene))
    mesh = RadiationTriangleMesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([[0, 1, 2]], dtype=np.uint32),
        "test-material",
    )
    backend = EmbreeTransportBackend()
    backend.build_scene(
        [mesh],
        MaterialTable(
            (
                MaterialSpec(
                    "test-material",
                    np.array([10.0, 1000.0]),
                    np.array([2.0, 2.0]),
                ),
            )
        ),
    )
    origins = np.array([[-1.0, 0.0, 0.0]])
    targets = np.array([[1.0, 0.0, 0.0]])
    assert backend.commit_updates() == 1
    assert backend.trace_path_lengths(origins, targets).lengths_m[0, 0] == pytest.approx(0.5)
    assert backend.trace_transmission(origins, targets, np.array([100.0]))[0, 0] == pytest.approx(
        np.exp(-1.0)
    )
