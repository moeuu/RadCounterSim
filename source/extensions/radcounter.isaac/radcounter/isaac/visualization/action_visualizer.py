"""Annotation-only route, tool-path, and target overlays for physical actions."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._common import author_points, finite_points


class ActionVisualizer:
    def __init__(
        self, stage: Any, root_path: str = "/World/RadInterActVisualization/Action"
    ) -> None:
        self.stage = stage
        root = root_path.rstrip("/")
        self.route_path = f"{root}/BaseRoute"
        self.tool_path = f"{root}/ToolPath"
        self.target_path = f"{root}/Target"

    def _curve(
        self,
        path: str,
        points_world_m: object,
        *,
        color: tuple[float, float, float],
        data_class: str,
    ) -> str:
        from pxr import Gf, Sdf, UsdGeom

        points = finite_points(points_world_m)
        if len(points) < 2:
            raise ValueError("an action path requires at least two points")
        curve = UsdGeom.BasisCurves.Define(self.stage, path)
        curve.CreateTypeAttr(UsdGeom.Tokens.linear)
        curve.CreateCurveVertexCountsAttr([len(points)])
        curve.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
        curve.CreateWidthsAttr([0.035])
        curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        curve.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(*color)])
        prim = curve.GetPrim()
        prim.CreateAttribute(
            "rad:visualization:dataClass", Sdf.ValueTypeNames.String, custom=True
        ).Set(data_class)
        prim.CreateAttribute(
            "rad:visualization:annotationOnly", Sdf.ValueTypeNames.Bool, custom=True
        ).Set(True)
        return path

    def author(
        self,
        *,
        base_route_world_m: object | None = None,
        tool_path_world_m: object | None = None,
        target_world_m: object | None = None,
    ) -> tuple[str | None, str | None, str | None]:
        route_path = None
        tool_path = None
        target_path = None
        if base_route_world_m is not None:
            route_path = self._curve(
                self.route_path,
                base_route_world_m,
                color=(0.12, 0.68, 0.98),
                data_class="planned_base_route_annotation",
            )
        if tool_path_world_m is not None:
            tool_path = self._curve(
                self.tool_path,
                tool_path_world_m,
                color=(0.98, 0.62, 0.12),
                data_class="planned_tool_path_annotation",
            )
        if target_world_m is not None:
            target = np.asarray(target_world_m, dtype=np.float64)
            if target.shape != (3,) or not np.all(np.isfinite(target)):
                raise ValueError("action target must be a finite 3-vector")
            prim = author_points(
                self.stage,
                self.target_path,
                target[None, :],
                colors_rgb=np.asarray(((1.0, 0.15, 0.08),)),
                widths_m=np.asarray((0.28,)),
                data_class="action_target_annotation",
            )
            from pxr import Sdf

            prim.CreateAttribute(
                "rad:visualization:annotationOnly", Sdf.ValueTypeNames.Bool, custom=True
            ).Set(True)
            target_path = self.target_path
        return route_path, tool_path, target_path

    def clear(self) -> None:
        root = self.route_path.rsplit("/", 1)[0]
        if self.stage.GetPrimAtPath(root).IsValid():
            self.stage.RemovePrim(root)
