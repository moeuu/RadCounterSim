"""Batched dose or count-rate map visualization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from ._common import author_points, finite_points, finite_values, sequential_color


class ColorRangeMode(StrEnum):
    FIXED = "fixed"
    PERCENTILE = "percentile"
    LOGARITHMIC = "logarithmic"


@dataclass(frozen=True, slots=True)
class DoseMapStyle:
    mode: ColorRangeMode = ColorRangeMode.LOGARITHMIC
    fixed_range: tuple[float, float] | None = None
    percentile_range: tuple[float, float] = (2.0, 98.0)
    minimum_width_m: float = 0.08
    maximum_width_m: float = 0.16

    def __post_init__(self) -> None:
        if self.minimum_width_m <= 0.0 or self.maximum_width_m < self.minimum_width_m:
            raise ValueError("dose-map widths must be positive and ordered")
        if not 0.0 <= self.percentile_range[0] < self.percentile_range[1] <= 100.0:
            raise ValueError("percentile range must be ordered inside [0, 100]")
        if self.mode == ColorRangeMode.FIXED and (
            self.fixed_range is None or self.fixed_range[1] <= self.fixed_range[0]
        ):
            raise ValueError("fixed color mode requires an ordered fixed_range")


class DoseMapVisualizer:
    def __init__(self, stage: Any, path: str = "/World/RadiationDoseProxy") -> None:
        self.stage = stage
        self.path = path

    @staticmethod
    def _normalize(values: np.ndarray, style: DoseMapStyle) -> np.ndarray:
        transformed = values.copy()
        if style.mode == ColorRangeMode.LOGARITHMIC:
            if np.any(transformed < 0.0):
                raise ValueError("logarithmic dose-map values must be nonnegative")
            transformed = np.log10(np.maximum(transformed, np.finfo(np.float64).tiny))
            lower, upper = float(np.min(transformed)), float(np.max(transformed))
        elif style.mode == ColorRangeMode.PERCENTILE:
            lower, upper = np.percentile(transformed, style.percentile_range)
        else:
            assert style.fixed_range is not None
            lower, upper = style.fixed_range
        if upper <= lower:
            return np.zeros_like(transformed)
        return np.clip((transformed - lower) / (upper - lower), 0.0, 1.0)

    def author(
        self,
        positions_world_m: object,
        values: object,
        *,
        style: DoseMapStyle | None = None,
        data_class: str = "dose_rate_sv_h",
    ) -> str:
        positions = finite_points(positions_world_m)
        samples = finite_values(values, len(positions), name="dose-map values")
        if np.any(samples < 0.0):
            raise ValueError("dose-map values must be nonnegative")
        selected_style = style or DoseMapStyle()
        normalized = self._normalize(samples, selected_style)
        colors = sequential_color(normalized)
        widths = selected_style.minimum_width_m + normalized * (
            selected_style.maximum_width_m - selected_style.minimum_width_m
        )
        prim = author_points(
            self.stage,
            self.path,
            positions,
            colors_rgb=colors,
            widths_m=widths,
            data_class=data_class,
        )
        from pxr import Sdf

        prim.CreateAttribute(
            "rad:visualization:colorRangeMode", Sdf.ValueTypeNames.String, custom=True
        ).Set(selected_style.mode.value)
        prim.CreateAttribute(
            "rad:visualization:values", Sdf.ValueTypeNames.DoubleArray, custom=True
        ).Set(samples.tolist())
        return self.path
