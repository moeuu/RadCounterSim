"""Selected finite-ray overlay with material-specific path-length metadata."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np


class RayDebugVisualizer:
    def __init__(
        self,
        stage: Any,
        path: str = "/World/RadInterActVisualization/SelectedRay",
    ) -> None:
        self.stage = stage
        self.path = path

    @staticmethod
    def _color(material_lengths_m: Mapping[str, float]) -> tuple[float, float, float]:
        active = sorted(name for name, length in material_lengths_m.items() if length > 0.0)
        if not active:
            return (0.90, 0.90, 0.90)
        digest = hashlib.blake2b("|".join(active).encode(), digest_size=3).digest()
        return tuple(0.25 + 0.65 * value / 255.0 for value in digest)

    def author(
        self,
        origin_world_m: object,
        target_world_m: object,
        material_lengths_m: Mapping[str, float],
    ) -> str:
        from pxr import Gf, Sdf, UsdGeom

        origin = np.asarray(origin_world_m, dtype=np.float64)
        target = np.asarray(target_world_m, dtype=np.float64)
        if (
            origin.shape != (3,)
            or target.shape != (3,)
            or not np.all(np.isfinite(np.concatenate((origin, target))))
        ):
            raise ValueError("ray endpoints must be finite 3-vectors")
        if np.linalg.norm(target - origin) <= 0.0:
            raise ValueError("ray endpoints must be distinct")
        if any(not np.isfinite(length) or length < 0.0 for length in material_lengths_m.values()):
            raise ValueError("material path lengths must be finite and nonnegative")
        curve = UsdGeom.BasisCurves.Define(self.stage, self.path)
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateCurveVertexCountsAttr([2])
        curve.CreatePointsAttr([Gf.Vec3f(*origin), Gf.Vec3f(*target)])
        curve.CreateWidthsAttr([0.025])
        curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        curve.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(
            [Gf.Vec3f(*self._color(material_lengths_m))]
        )
        prim = curve.GetPrim()
        prim.CreateAttribute(
            "rad:visualization:dataClass", Sdf.ValueTypeNames.String, custom=True
        ).Set("selected_finite_transport_ray")
        material_ids = tuple(sorted(material_lengths_m))
        prim.CreateAttribute(
            "rad:visualization:materialIds", Sdf.ValueTypeNames.StringArray, custom=True
        ).Set(list(material_ids))
        prim.CreateAttribute(
            "rad:visualization:materialPathLengthsM",
            Sdf.ValueTypeNames.DoubleArray,
            custom=True,
        ).Set([float(material_lengths_m[name]) for name in material_ids])
        prim.CreateAttribute(
            "rad:visualization:rayLengthM", Sdf.ValueTypeNames.Double, custom=True
        ).Set(float(np.linalg.norm(target - origin)))
        return self.path
