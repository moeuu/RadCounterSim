"""Separate predicted, observed, and normalized-residual layers."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._common import author_points, diverging_color, finite_points, finite_values


class ResidualVisualizer:
    def __init__(self, stage: Any, root_path: str = "/World/RadInterActVisualization") -> None:
        self.stage = stage
        root = root_path.rstrip("/")
        self.predicted_path = f"{root}/PredictedPostAction"
        self.observed_path = f"{root}/ObservedPostAction"
        self.residual_path = f"{root}/NormalizedResidual"

    def author(
        self,
        positions_world_m: object,
        predicted_values: object,
        observed_values: object,
        normalized_residuals: object,
    ) -> tuple[str, str, str]:
        positions = finite_points(positions_world_m)
        count = len(positions)
        predicted = finite_values(predicted_values, count, name="predicted values")
        observed = finite_values(observed_values, count, name="observed values")
        residual = finite_values(normalized_residuals, count, name="normalized residuals")
        if np.any(predicted < 0.0) or np.any(observed < 0.0):
            raise ValueError("predicted and observed values must be nonnegative")
        layers = (
            (
                self.predicted_path,
                predicted,
                np.tile((0.18, 0.52, 0.95), (count, 1)),
                "predicted_post_action",
            ),
            (
                self.observed_path,
                observed,
                np.tile((0.20, 0.88, 0.42), (count, 1)),
                "observed_post_action",
            ),
            (
                self.residual_path,
                residual,
                diverging_color(residual),
                "normalized_residual",
            ),
        )
        from pxr import Sdf

        for path, values, colors, data_class in layers:
            prim = author_points(
                self.stage,
                path,
                positions,
                colors_rgb=colors,
                widths_m=np.full(count, 0.16, dtype=np.float64),
                data_class=data_class,
            )
            prim.CreateAttribute(
                "rad:visualization:values", Sdf.ValueTypeNames.DoubleArray, custom=True
            ).Set(values.tolist())
        return self.predicted_path, self.observed_path, self.residual_path
