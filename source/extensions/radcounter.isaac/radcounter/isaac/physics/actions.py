"""Apply three-dimensional countermeasure poses to an Isaac USD stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class IsaacPhysicsUnavailable(RuntimeError):
    """Raised when USD/PhysX operations are requested outside Isaac Sim."""


def _usd_modules() -> tuple[Any, Any, Any]:
    try:
        import omni.usd  # type: ignore[import-not-found]
        from pxr import Gf, Sdf, UsdGeom  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        raise IsaacPhysicsUnavailable(
            "USD physics actions require the Isaac Sim runtime"
        ) from error
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


class UsdPhysicsActionExecutor:
    """Execute shield and object transforms as full 3-D USD poses."""

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

    def set_world_pose(self, prim_path: str, pose: Pose3D) -> None:
        stage = self._stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise KeyError(f"USD prim does not exist: {prim_path}")
        x, y, z, w = pose.orientation_xyzw
        rotation = self._gf.Quatd(float(w), float(x), float(y), float(z))
        matrix = self._gf.Matrix4d(1.0)
        matrix.SetRotate(rotation)
        matrix.SetTranslateOnly(self._gf.Vec3d(*pose.position_m.tolist()))
        transformable = self._usd_geom.Xformable(prim)
        transformable.ClearXformOpOrder()
        transformable.AddTransformOp().Set(matrix)
        self._mark_action(prim, "pose_applied")

    def place_shield(self, shield_prim_path: str, station_pose: Pose3D) -> None:
        self.set_world_pose(shield_prim_path, station_pose)

    def move_object(self, object_prim_path: str, destination_pose: Pose3D) -> None:
        self.set_world_pose(object_prim_path, destination_pose)

    def remove_object(self, object_prim_path: str) -> None:
        stage = self._stage()
        path = self._sdf.Path(object_prim_path)
        if not stage.GetPrimAtPath(path).IsValid():
            raise KeyError(f"USD prim does not exist: {object_prim_path}")
        if not stage.RemovePrim(path):
            raise RuntimeError(f"failed to remove USD prim: {object_prim_path}")

    def _mark_action(self, prim: Any, status: str) -> None:
        attribute = prim.GetAttribute("radcounter:actionStatus")
        if not attribute.IsValid():
            attribute = prim.CreateAttribute(
                "radcounter:actionStatus",
                self._sdf.ValueTypeNames.String,
                custom=True,
            )
        attribute.Set(status)
