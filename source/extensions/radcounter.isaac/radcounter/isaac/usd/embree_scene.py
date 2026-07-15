"""Convert world-space USD meshes into the native Embree backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from radcounter.core.radiation.embree_native import (
    EmbreeNativeScene,
    TriangleMesh,
)


class UsdMeshConversionUnavailable(RuntimeError):
    """Raised when USD conversion is called outside Isaac Sim."""


def _usd_types() -> tuple[Any, Any]:
    try:
        from pxr import Usd, UsdGeom  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise UsdMeshConversionUnavailable(
            "USD mesh conversion requires the Isaac Sim pxr modules"
        ) from error
    return Usd, UsdGeom


def _triangulate(face_counts: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    offset = 0
    for count in face_counts:
        count_int = int(count)
        polygon = face_indices[offset : offset + count_int]
        offset += count_int
        if count_int < 3:
            continue
        for local_index in range(1, count_int - 1):
            triangles.append(
                (int(polygon[0]), int(polygon[local_index]), int(polygon[local_index + 1]))
            )
    return np.asarray(triangles, dtype=np.uint32).reshape((-1, 3))


def extract_triangle_meshes(
    stage: Any,
    material_index_by_id: Mapping[str, int],
) -> tuple[TriangleMesh, ...]:
    """Extract every tagged USD mesh in world coordinates."""

    usd, usd_geom = _usd_types()
    xform_cache = usd_geom.XformCache(usd.TimeCode.Default())
    meshes: list[TriangleMesh] = []
    for prim in stage.Traverse():
        if not prim.IsA(usd_geom.Mesh):
            continue
        material_attribute = prim.GetAttribute("radcounter:materialId")
        if not material_attribute.IsValid() or not material_attribute.HasAuthoredValue():
            continue
        material_id = str(material_attribute.Get())
        if material_id not in material_index_by_id:
            raise KeyError(f"unregistered radiation material: {material_id}")
        mesh = usd_geom.Mesh(prim)
        points = mesh.GetPointsAttr().Get(usd.TimeCode.Default())
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        triangles = _triangulate(counts, indices)
        if len(points) == 0 or len(triangles) == 0:
            continue
        transform = xform_cache.GetLocalToWorldTransform(prim)
        vertices_m = np.asarray(
            [tuple(transform.Transform(point)) for point in points],
            dtype=np.float64,
        )
        meshes.append(
            TriangleMesh(
                vertices_m=vertices_m,
                triangles=triangles,
                material_index=material_index_by_id[material_id],
            )
        )
    return tuple(meshes)


class UsdEmbreeSceneAdapter:
    """Rebuild an Embree scene after a USD geometry/material revision."""

    def __init__(self) -> None:
        self._scene: EmbreeNativeScene | None = None

    @property
    def revision(self) -> int:
        return 0 if self._scene is None else self._scene.revision

    def rebuild(self, stage: Any, material_index_by_id: Mapping[str, int]) -> int:
        scene = EmbreeNativeScene()
        for mesh in extract_triangle_meshes(stage, material_index_by_id):
            scene.add_mesh(mesh)
        scene.commit()
        self._scene = scene
        return scene.revision

    def trace_transmission(
        self,
        origins_m: np.ndarray,
        targets_m: np.ndarray,
        attenuation_per_m: np.ndarray,
    ) -> np.ndarray:
        if self._scene is None:
            raise RuntimeError("rebuild() must be called before tracing")
        return self._scene.trace_transmission(
            origins_m,
            targets_m,
            attenuation_per_m,
        )
