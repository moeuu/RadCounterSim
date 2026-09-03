"""Explicit kinematic scene edits for radiological sensitivity experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from radcounter.core.experiments import EvidenceClass


class IsaacUsdUnavailable(RuntimeError):
    """Raised when scene-edit operations are requested outside Isaac Sim."""


def _usd_modules() -> tuple[Any, Any, Any]:
    try:
        import omni.usd  # type: ignore[import-not-found]
        from pxr import Gf, Sdf, UsdGeom  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise IsaacUsdUnavailable("USD scene edits require the Isaac Sim runtime") from error
    return omni.usd, Gf, (Sdf, UsdGeom)


@dataclass(frozen=True)
class Pose3D:
    """World pose with quaternion stored in xyzw order."""

    position_m: np.ndarray
    orientation_xyzw: np.ndarray

    def __post_init__(self) -> None:
        position = np.asarray(self.position_m, dtype=np.float64)
        orientation = np.asarray(self.orientation_xyzw, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("position_m must contain three finite values")
        if orientation.shape != (4,) or not np.all(np.isfinite(orientation)):
            raise ValueError("orientation_xyzw must contain four finite values")
        norm = float(np.linalg.norm(orientation))
        if norm <= 1.0e-12:
            raise ValueError("orientation_xyzw must have nonzero norm")
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "orientation_xyzw", orientation / norm)


class SceneEditOperation(StrEnum):
    SET_GEOMETRY_POSE = "set_geometry_pose"
    SET_SOURCE_POSE = "set_source_pose"
    SET_SOURCE_PRESENCE = "set_source_presence"


@dataclass(frozen=True, slots=True)
class SceneEditRecord:
    """Auditable evidence that must never be presented as robot execution."""

    target_prim_path: str
    operation: SceneEditOperation
    revision: int
    evidence_class: EvidenceClass = EvidenceClass.KINEMATIC_SCENE_EDIT


class UsdSceneStateEditor:
    """Apply explicit scene-state edits without pretending a robot executed them."""

    def __init__(self) -> None:
        usd_module, gf_module, usd_types = _usd_modules()
        self._context = usd_module.get_context()
        self._gf = gf_module
        self._sdf, self._usd_geom = usd_types

    def _stage(self) -> Any:
        stage = self._context.get_stage()
        if stage is None:
            raise RuntimeError("no USD stage is open")
        return stage

    def _prim(self, prim_path: str) -> Any:
        stage = self._stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise KeyError(f"USD prim does not exist: {prim_path}")
        return prim

    def _set_world_pose(
        self, prim_path: str, pose: Pose3D, operation: SceneEditOperation
    ) -> SceneEditRecord:
        prim = self._prim(prim_path)
        x, y, z, w = pose.orientation_xyzw
        rotation = self._gf.Quatd(float(w), float(x), float(y), float(z))
        matrix = self._gf.Matrix4d(1.0)
        matrix.SetRotate(rotation)
        matrix.SetTranslateOnly(self._gf.Vec3d(*pose.position_m.tolist()))
        transformable = self._usd_geom.Xformable(prim)
        transformable.ClearXformOpOrder()
        transformable.AddTransformOp().Set(matrix)
        return self._mark_edit(prim, operation)

    def set_geometry_pose(self, prim_path: str, pose: Pose3D) -> SceneEditRecord:
        prim = self._prim(prim_path)
        material = prim.GetAttribute("rad:material:id")
        if not material or not material.HasAuthoredValueOpinion():
            raise ValueError(f"geometry scene edit requires rad:material:id on {prim_path}")
        return self._set_world_pose(prim_path, pose, SceneEditOperation.SET_GEOMETRY_POSE)

    def set_source_pose(self, prim_path: str, pose: Pose3D) -> SceneEditRecord:
        prim = self._prim(prim_path)
        source_type = prim.GetAttribute("rad:source:type")
        if not source_type or not source_type.HasAuthoredValueOpinion():
            raise ValueError(f"source scene edit requires rad:source:type on {prim_path}")
        return self._set_world_pose(prim_path, pose, SceneEditOperation.SET_SOURCE_POSE)

    def set_source_presence(self, prim_path: str, *, present: bool) -> SceneEditRecord:
        prim = self._prim(prim_path)
        source_type = prim.GetAttribute("rad:source:type")
        if not source_type or not source_type.HasAuthoredValueOpinion():
            raise ValueError(f"source scene edit requires rad:source:type on {prim_path}")
        enabled = prim.GetAttribute("rad:source:enabled")
        if not enabled:
            enabled = prim.CreateAttribute(
                "rad:source:enabled", self._sdf.ValueTypeNames.Bool, custom=True
            )
        enabled.Set(bool(present))
        return self._mark_edit(prim, SceneEditOperation.SET_SOURCE_PRESENCE)

    def _mark_edit(self, prim: Any, operation: SceneEditOperation) -> SceneEditRecord:
        evidence = prim.GetAttribute("rad:evidence:class")
        if not evidence:
            evidence = prim.CreateAttribute(
                "rad:evidence:class",
                self._sdf.ValueTypeNames.String,
                custom=True,
            )
        evidence.Set(EvidenceClass.KINEMATIC_SCENE_EDIT.value)
        operation_attribute = prim.GetAttribute("rad:sceneEdit:lastOperation")
        if not operation_attribute:
            operation_attribute = prim.CreateAttribute(
                "rad:sceneEdit:lastOperation",
                self._sdf.ValueTypeNames.String,
                custom=True,
            )
        operation_attribute.Set(operation.value)
        revision_attribute = prim.GetAttribute("rad:sceneEdit:revision")
        if not revision_attribute:
            revision_attribute = prim.CreateAttribute(
                "rad:sceneEdit:revision", self._sdf.ValueTypeNames.Int64, custom=True
            )
        revision = int(revision_attribute.Get() or 0) + 1
        revision_attribute.Set(revision)
        return SceneEditRecord(str(prim.GetPath()), operation, revision)
