"""Optional Embree 4 Python boundary with strict array and unit validation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType

import numpy as np
from numpy.typing import NDArray

from radcounter.core.radiation.backend import PathLengthBatch
from radcounter.core.radiation.materials import MaterialTable


class EmbreeNativeUnavailable(RuntimeError):
    """Raised when the optional native Embree module is not importable."""


def _load_native() -> ModuleType:
    try:
        return import_module("_radcounter_embree")
    except ImportError as error:
        raise EmbreeNativeUnavailable(
            "_radcounter_embree is unavailable; build it with "
            "scripts/build_native.sh and add build/native/python to PYTHONPATH"
        ) from error


def native_embree_available() -> bool:
    """Return whether the compiled module can be loaded in this process."""

    try:
        _load_native()
    except EmbreeNativeUnavailable:
        return False
    return True


@dataclass(frozen=True)
class TriangleMesh:
    """One closed triangle mesh with a material-table row index."""

    vertices_m: NDArray[np.float64]
    triangles: NDArray[np.uint32]
    material_index: int

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_m, dtype=np.float64)
        triangles = np.asarray(self.triangles, dtype=np.uint32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            raise ValueError("vertices_m must have nonempty shape (N, 3)")
        if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) == 0:
            raise ValueError("triangles must have nonempty shape (M, 3)")
        if int(np.max(triangles)) >= len(vertices):
            raise ValueError("triangle index is outside vertices_m")
        if self.material_index < 0:
            raise ValueError("material_index must be nonnegative")
        object.__setattr__(self, "vertices_m", vertices)
        object.__setattr__(self, "triangles", triangles)


@dataclass(frozen=True)
class RadiationTriangleMesh:
    """Triangle mesh tagged by the public material identifier."""

    vertices_m: NDArray[np.float64]
    triangles: NDArray[np.uint32]
    material_id: str

    def __post_init__(self) -> None:
        if not self.material_id:
            raise ValueError("material_id must not be empty")
        validated = TriangleMesh(self.vertices_m, self.triangles, 0)
        object.__setattr__(self, "vertices_m", validated.vertices_m)
        object.__setattr__(self, "triangles", validated.triangles)


class EmbreeNativeScene:
    """Own a native Embree scene and evaluate segment transmission."""

    def __init__(self) -> None:
        native = _load_native()
        self._scene = native.Scene()
        self._committed = False

    @property
    def revision(self) -> int:
        return int(self._scene.revision)

    def add_mesh(self, mesh: TriangleMesh) -> int:
        """Copy one mesh into native storage and return its geometry ID."""

        geometry_id = self._scene.add_triangle_mesh(
            np.asarray(mesh.vertices_m, dtype=np.float32, order="C"),
            np.asarray(mesh.triangles, dtype=np.uint32, order="C"),
            mesh.material_index,
        )
        self._committed = False
        return int(geometry_id)

    def commit(self) -> None:
        self._scene.commit()
        self._committed = True

    def trace_transmission(
        self,
        origins_m: NDArray[np.float64],
        targets_m: NDArray[np.float64],
        attenuation_per_m: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return transmission with shape ``(segment, energy_bin)``."""

        if not self._committed:
            raise RuntimeError("commit() must be called before tracing")
        origins = np.asarray(origins_m, dtype=np.float64, order="C")
        targets = np.asarray(targets_m, dtype=np.float64, order="C")
        attenuation = np.asarray(attenuation_per_m, dtype=np.float64, order="C")
        if origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("origins_m must have shape (N, 3)")
        if targets.shape != origins.shape:
            raise ValueError("targets_m must have the same shape as origins_m")
        if attenuation.ndim != 2 or attenuation.shape[1] == 0:
            raise ValueError("attenuation_per_m must have shape (material, energy_bin)")
        if np.any(attenuation < 0.0) or not np.all(np.isfinite(attenuation)):
            raise ValueError("attenuation_per_m must be finite and nonnegative")
        return np.asarray(
            self._scene.trace_transmission(origins, targets, attenuation),
            dtype=np.float64,
        )


class EmbreeTransportBackend:
    """Drop-in ``RayTransportBackend`` implemented by Embree 4."""

    def __init__(self) -> None:
        self._scene: EmbreeNativeScene | None = None
        self._materials = MaterialTable(())
        self._material_ids: tuple[str, ...] = ()

    def build_scene(
        self,
        meshes: Sequence[object],
        material_table: MaterialTable,
    ) -> None:
        """Build and commit a native scene from material-tagged meshes."""

        if any(not isinstance(mesh, RadiationTriangleMesh) for mesh in meshes):
            raise TypeError("EmbreeTransportBackend accepts RadiationTriangleMesh objects only")
        typed_meshes = tuple(mesh for mesh in meshes if isinstance(mesh, RadiationTriangleMesh))
        material_ids = tuple(dict.fromkeys(mesh.material_id for mesh in typed_meshes))
        for material_id in material_ids:
            material_table.get(material_id)
        material_index = {material_id: index for index, material_id in enumerate(material_ids)}
        scene = EmbreeNativeScene()
        for mesh in typed_meshes:
            scene.add_mesh(
                TriangleMesh(
                    mesh.vertices_m,
                    mesh.triangles,
                    material_index[mesh.material_id],
                )
            )
        scene.commit()
        self._scene = scene
        self._materials = material_table
        self._material_ids = material_ids

    def commit_updates(self) -> int:
        """Return the committed native scene revision."""

        if self._scene is None:
            raise RuntimeError("build_scene() must be called before commit_updates()")
        return self._scene.revision

    def trace_path_lengths(
        self,
        origins_m: NDArray[np.float64],
        targets_m: NDArray[np.float64],
    ) -> PathLengthBatch:
        """Return path length through every configured material."""

        if self._scene is None:
            raise RuntimeError("build_scene() must be called before tracing")
        origins = np.asarray(origins_m, dtype=np.float64)
        targets = np.asarray(targets_m, dtype=np.float64)
        if not self._material_ids:
            if origins.ndim != 2 or origins.shape[1:] != (3,) or targets.shape != origins.shape:
                raise ValueError("origins_m and targets_m must both have shape (R, 3)")
            return PathLengthBatch(
                (),
                np.zeros((len(origins), 0), dtype=np.float64),
                np.zeros(len(origins), dtype=np.bool_),
            )
        identity_attenuation = np.eye(len(self._material_ids), dtype=np.float64)
        material_transmission = self._scene.trace_transmission(
            origins,
            targets,
            identity_attenuation,
        )
        lengths_m = -np.log(np.clip(material_transmission, np.finfo(np.float64).tiny, 1.0))
        return PathLengthBatch(
            self._material_ids,
            lengths_m,
            np.zeros(len(origins), dtype=np.bool_),
        )

    def trace_transmission(
        self,
        origins_m: NDArray[np.float64],
        targets_m: NDArray[np.float64],
        energies_keV: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate material attenuation at each requested energy."""

        energies = np.asarray(energies_keV, dtype=np.float64)
        if energies.ndim != 1 or np.any(energies <= 0.0) or not np.all(np.isfinite(energies)):
            raise ValueError("energies_keV must be a finite positive 1-D array")
        batch = self.trace_path_lengths(origins_m, targets_m)
        optical_depth = np.zeros((len(batch.lengths_m), len(energies)), dtype=np.float64)
        for index, material_id in enumerate(batch.material_ids):
            optical_depth += (
                batch.lengths_m[:, index, None]
                * self._materials.attenuation_m_inv(material_id, energies)[None, :]
            )
        return np.exp(-np.minimum(optical_depth, 700.0))
