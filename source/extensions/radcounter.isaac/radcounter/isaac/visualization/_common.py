"""Small USD authoring helpers shared by operator visualization layers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def finite_points(values: object, *, name: str = "points") -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must be a finite N x 3 array")
    return points


def finite_values(values: object, count: int, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (count,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite length-{count} array")
    return result


def author_points(
    stage: Any,
    path: str,
    positions_m: np.ndarray,
    *,
    colors_rgb: np.ndarray,
    widths_m: np.ndarray,
    data_class: str,
) -> Any:
    from pxr import Gf, Sdf, UsdGeom

    positions = finite_points(positions_m)
    colors = np.asarray(colors_rgb, dtype=np.float64)
    widths = np.asarray(widths_m, dtype=np.float64)
    if colors.shape != positions.shape or not np.all(np.isfinite(colors)):
        raise ValueError("point colors must be a finite N x 3 array")
    if widths.shape != (len(positions),) or np.any(widths < 0.0):
        raise ValueError("point widths must be a nonnegative length-N array")
    points = UsdGeom.Points.Define(stage, path)
    points.CreatePointsAttr([Gf.Vec3f(*row) for row in positions])
    points.CreateWidthsAttr([float(value) for value in widths])
    points.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(
        [Gf.Vec3f(*np.clip(row, 0.0, 1.0)) for row in colors]
    )
    points.GetPrim().CreateAttribute(
        "rad:visualization:dataClass", Sdf.ValueTypeNames.String, custom=True
    ).Set(data_class)
    return points.GetPrim()


def set_visibility(stage: Any, path: str, visible: bool) -> None:
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return
    imageable = UsdGeom.Imageable(prim)
    if imageable:
        imageable.MakeVisible() if visible else imageable.MakeInvisible()


def sequential_color(values: np.ndarray, *, logarithmic: bool = False) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if logarithmic:
        if np.any(data < 0.0):
            raise ValueError("logarithmic color data must be nonnegative")
        data = np.log10(np.maximum(data, np.finfo(np.float64).tiny))
    lower = float(np.min(data)) if len(data) else 0.0
    upper = float(np.max(data)) if len(data) else 0.0
    normalized = (
        np.zeros_like(data)
        if math.isclose(lower, upper)
        else np.clip((data - lower) / (upper - lower), 0.0, 1.0)
    )
    return np.column_stack(
        (
            np.minimum(1.0, 2.0 * normalized),
            1.0 - np.abs(2.0 * normalized - 1.0),
            np.minimum(1.0, 2.0 * (1.0 - normalized)),
        )
    )


def diverging_color(values: np.ndarray, *, limit: float | None = None) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    bound = float(np.max(np.abs(data))) if limit is None and len(data) else float(limit or 1.0)
    bound = max(bound, np.finfo(np.float64).eps)
    normalized = np.clip(data / bound, -1.0, 1.0)
    neutral = 1.0 - np.abs(normalized)
    return np.column_stack(
        (
            neutral + np.maximum(normalized, 0.0),
            neutral,
            neutral + np.maximum(-normalized, 0.0),
        )
    )
