"""Belief-source and position-uncertainty visualization layers."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._common import author_points, finite_points, finite_values, sequential_color


class SourceEstimateVisualizer:
    def __init__(
        self,
        stage: Any,
        *,
        estimate_path: str = "/World/RadInterActVisualization/BeliefSources",
        uncertainty_path: str = "/World/RadInterActVisualization/SourceUncertainty",
    ) -> None:
        self.stage = stage
        self.estimate_path = estimate_path
        self.uncertainty_path = uncertainty_path

    def author(
        self,
        positions_world_m: object,
        activity_bq: object,
        *,
        position_standard_deviation_m: object | None = None,
        activity_standard_deviation_bq: object | None = None,
    ) -> tuple[str, str | None]:
        positions = finite_points(positions_world_m)
        activity = finite_values(activity_bq, len(positions), name="source activity")
        if np.any(activity < 0.0):
            raise ValueError("source activity must be nonnegative")
        colors = sequential_color(activity, logarithmic=True)
        widths = np.full(len(positions), 0.18, dtype=np.float64)
        estimate = author_points(
            self.stage,
            self.estimate_path,
            positions,
            colors_rgb=colors,
            widths_m=widths,
            data_class="belief_source_activity_bq",
        )
        from pxr import Sdf

        estimate.CreateAttribute(
            "rad:visualization:activityBq", Sdf.ValueTypeNames.DoubleArray, custom=True
        ).Set(activity.tolist())
        if position_standard_deviation_m is not None and activity_standard_deviation_bq is not None:
            raise ValueError("choose position or activity uncertainty, not both")
        if position_standard_deviation_m is None and activity_standard_deviation_bq is None:
            if self.stage.GetPrimAtPath(self.uncertainty_path).IsValid():
                self.stage.RemovePrim(self.uncertainty_path)
            return self.estimate_path, None
        if position_standard_deviation_m is not None:
            standard_deviation = np.asarray(position_standard_deviation_m, dtype=np.float64)
            if standard_deviation.shape == (len(positions), 3):
                radial = np.linalg.norm(standard_deviation, axis=1)
            elif standard_deviation.shape == (len(positions),):
                radial = standard_deviation
            else:
                raise ValueError("position uncertainty must be N or N x 3")
            data_class = "belief_position_uncertainty_2sigma_m"
            attribute_name = "rad:visualization:standardDeviationM"
            uncertainty_widths = np.maximum(0.04, 2.0 * radial)
        else:
            radial = finite_values(
                activity_standard_deviation_bq,
                len(positions),
                name="activity uncertainty",
            )
            data_class = "belief_activity_uncertainty_bq"
            attribute_name = "rad:visualization:activityStandardDeviationBq"
            maximum = max(float(np.max(radial)), np.finfo(np.float64).eps)
            uncertainty_widths = 0.08 + 0.42 * radial / maximum
        if np.any(~np.isfinite(radial)) or np.any(radial < 0.0):
            raise ValueError("source uncertainty must be finite and nonnegative")
        uncertainty = author_points(
            self.stage,
            self.uncertainty_path,
            positions,
            colors_rgb=np.tile((0.95, 0.72, 0.18), (len(positions), 1)),
            widths_m=uncertainty_widths,
            data_class=data_class,
        )
        uncertainty.CreateAttribute(
            attribute_name,
            Sdf.ValueTypeNames.DoubleArray,
            custom=True,
        ).Set(radial.tolist())
        return self.estimate_path, self.uncertainty_path
