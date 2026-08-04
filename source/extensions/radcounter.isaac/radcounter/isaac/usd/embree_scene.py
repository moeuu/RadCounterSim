"""Convert world-space USD meshes into the native Embree backend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from radcounter.core.radiation.embree_native import (
    EmbreeNativeScene,
    TriangleMesh,
)
from radcounter.core.scene import MaterialMode, UsdRadiationAttributes


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
        material_attribute = prim.GetAttribute(UsdRadiationAttributes.MATERIAL_ID)
        if not material_attribute.IsValid() or not material_attribute.HasAuthoredValue():
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
        transform_rows = np.asarray(
            [[float(transform[row][column]) for column in range(4)] for row in range(4)],
            dtype=np.float64,
        )
        world_transform = transform_rows.T
        mode_value = prim.GetAttribute(UsdRadiationAttributes.MATERIAL_MODE)
        geometry_mode = (
            str(mode_value.Get())
            if mode_value.IsValid() and mode_value.HasAuthoredValue()
            else MaterialMode.SOLID.value
        )
        thickness_attribute = prim.GetAttribute(UsdRadiationAttributes.MATERIAL_THICKNESS_M)
        thickness_m = (
            float(thickness_attribute.Get())
            if thickness_attribute.IsValid() and thickness_attribute.HasAuthoredValue()
            else None
        )
        meshes.append(
            TriangleMesh(
                vertices_m=np.asarray(points, dtype=np.float64),
                triangles=triangles,
                material_index=material_index_by_id[material_id],
                geometry_mode=geometry_mode,
                explicit_thickness_m=thickness_m,
                mesh_id=str(prim.GetPath()),
                world_transform=world_transform,
            )
        )
    return tuple(meshes)


class UsdEmbreeSceneAdapter:
    """Rebuild an Embree scene after a USD geometry/material revision."""

    def __init__(self) -> None:
        self._scene: EmbreeNativeScene | None = None
        self._geometry_id_by_prim_path: dict[str, int] = {}

    @property
    def revision(self) -> int:
        return 0 if self._scene is None else self._scene.revision

    def rebuild(self, stage: Any, material_index_by_id: Mapping[str, int]) -> int:
        scene = EmbreeNativeScene()
        geometry_id_by_prim_path: dict[str, int] = {}
        for mesh in extract_triangle_meshes(stage, material_index_by_id):
            if mesh.mesh_id is None:
                raise RuntimeError("USD-derived radiation mesh has no prim path")
            geometry_id_by_prim_path[mesh.mesh_id] = scene.add_mesh(mesh)
        scene.commit()
        self._scene = scene
        self._geometry_id_by_prim_path = geometry_id_by_prim_path
        return scene.revision

    def update_prim_transform(self, stage: Any, prim_path: str) -> None:
        if self._scene is None:
            raise RuntimeError("rebuild() must be called before updates")
        try:
            geometry_id = self._geometry_id_by_prim_path[prim_path]
        except KeyError as error:
            raise KeyError(f"USD prim is absent from the Embree scene: {prim_path}") from error
        usd, usd_geom = _usd_types()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise KeyError(f"USD prim does not exist: {prim_path}")
        matrix = usd_geom.XformCache(usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
        rows = np.asarray(
            [[float(matrix[row][column]) for column in range(4)] for row in range(4)],
            dtype=np.float64,
        )
        self._scene.update_instance_transform(geometry_id, rows.T)

    def remove_prim(self, prim_path: str) -> None:
        if self._scene is None:
            raise RuntimeError("rebuild() must be called before updates")
        try:
            geometry_id = self._geometry_id_by_prim_path.pop(prim_path)
        except KeyError as error:
            raise KeyError(f"USD prim is absent from the Embree scene: {prim_path}") from error
        self._scene.remove_geometry(geometry_id)

    def commit_updates(self) -> int:
        if self._scene is None:
            raise RuntimeError("rebuild() must be called before updates")
        self._scene.commit()
        return self._scene.revision

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
