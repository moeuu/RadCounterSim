"""Dose-dependent camera image formation for radiation environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from radcounter.core.rendering.models import HighDoseCameraConfig


@dataclass(frozen=True)
class CameraFrameResult:
    image: np.ndarray
    dropped: bool
    diagnostics: dict[str, float]


class HighDoseCameraModel:
    """Stateful CMOS/CCD degradation approximation.

    It separates transient ionizing events from cumulative sensor damage.  The
    model is deliberately image-space and bounded, so it can run after an RTX
    RGB render without adding geometry or ray-tracing cost.
    """

    def __init__(self, config: HighDoseCameraConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.random_seed)
        self._cumulative_dose_gy = 0.0
        self._hot_pixels: np.ndarray | None = None
        self._previous: np.ndarray | None = None
        self._shape: tuple[int, int] | None = None

    @property
    def cumulative_dose_gy(self) -> float:
        return self._cumulative_dose_gy

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.config.random_seed)
        self._cumulative_dose_gy = 0.0
        self._hot_pixels = None
        self._previous = None
        self._shape = None

    def process(
        self,
        image: np.ndarray,
        *,
        dose_rate_gy_h: float,
        exposure_s: float,
    ) -> CameraFrameResult:
        source = np.asarray(image)
        if source.ndim != 3 or source.shape[2] not in {3, 4}:
            raise ValueError("camera image must have shape (height, width, 3|4)")
        if dose_rate_gy_h < 0.0 or exposure_s <= 0.0:
            raise ValueError("dose_rate_gy_h must be non-negative and exposure_s positive")
        if not self.config.enabled:
            return CameraFrameResult(source.copy(), False, {"dose_rate_gy_h": dose_rate_gy_h})

        rgb, alpha, scale = self._to_float(source)
        height, width = rgb.shape[:2]
        if self._shape != (height, width):
            self._shape = (height, width)
            self._hot_pixels = np.zeros(height * width, dtype=bool)
            self._previous = None

        frame_dose = dose_rate_gy_h * exposure_s / 3600.0
        self._cumulative_dose_gy += frame_dose
        permanent_count = self._update_permanent_damage(height * width)

        onset_ratio = max(0.0, dose_rate_gy_h / self.config.onset_dose_rate_gy_h - 1.0)
        drop_probability = 1.0 - np.exp(-self.config.drop_rate_per_s * onset_ratio * exposure_s)
        dropped = bool(self._rng.random() < drop_probability)
        if dropped:
            held = self._previous if self._previous is not None else np.zeros_like(source)
            return CameraFrameResult(
                held.copy(),
                True,
                {
                    "dose_rate_gy_h": dose_rate_gy_h,
                    "frame_dose_gy": frame_dose,
                    "cumulative_dose_gy": self._cumulative_dose_gy,
                    "drop_probability": float(drop_probability),
                    "transient_hits": 0.0,
                    "permanent_hot_pixels": float(permanent_count),
                },
            )

        hit_expectation = (
            self.config.transient_hits_per_mpix_s_per_gy_h
            * (height * width / 1_000_000.0)
            * exposure_s
            * dose_rate_gy_h
        )
        transient_hits = min(
            int(self._rng.poisson(min(hit_expectation, 1_000_000.0))),
            self.config.max_transient_hits_per_frame,
        )
        if transient_hits:
            self._apply_transient_hits(rgb, transient_hits)
        self._apply_permanent_hot_pixels(rgb)

        half_rate = self.config.desaturation_half_dose_rate_gy_h
        color_fraction = 1.0 / (1.0 + dose_rate_gy_h / half_rate)
        luminance = np.sum(rgb * np.asarray([0.2126, 0.7152, 0.0722]), axis=2, keepdims=True)
        rgb = luminance + color_fraction * (rgb - luminance)
        bloom = self.config.bloom_gain * np.log1p(dose_rate_gy_h / self.config.onset_dose_rate_gy_h)
        rgb += bloom * np.maximum(rgb - 0.75, 0.0)
        noise_std = self.config.read_noise_std * np.sqrt(1.0 + onset_ratio)
        if noise_std:
            rgb += self._rng.normal(0.0, noise_std, size=rgb.shape)
        rgb = np.clip(rgb, 0.0, 1.0)
        output = self._from_float(rgb, alpha, source.dtype, scale)
        self._previous = output.copy()
        return CameraFrameResult(
            output,
            False,
            {
                "dose_rate_gy_h": dose_rate_gy_h,
                "frame_dose_gy": frame_dose,
                "cumulative_dose_gy": self._cumulative_dose_gy,
                "drop_probability": float(drop_probability),
                "transient_hits": float(transient_hits),
                "permanent_hot_pixels": float(permanent_count),
                "color_fraction": float(color_fraction),
            },
        )

    def _update_permanent_damage(self, pixel_count: int) -> int:
        assert self._hot_pixels is not None
        target_fraction = min(
            self.config.max_permanent_hot_pixel_fraction,
            self._cumulative_dose_gy * self.config.permanent_hot_pixel_fraction_per_gy,
        )
        target = int(round(pixel_count * target_fraction))
        current = int(np.count_nonzero(self._hot_pixels))
        missing = target - current
        if missing > 0:
            available = np.flatnonzero(~self._hot_pixels)
            selected = self._rng.choice(available, size=min(missing, len(available)), replace=False)
            self._hot_pixels[selected] = True
        return int(np.count_nonzero(self._hot_pixels))

    def _apply_permanent_hot_pixels(self, rgb: np.ndarray) -> None:
        assert self._hot_pixels is not None
        indices = np.flatnonzero(self._hot_pixels)
        if not len(indices):
            return
        width = rgb.shape[1]
        rows, columns = np.divmod(indices, width)
        rgb[rows, columns] = self._rng.uniform(0.85, 1.0, size=(len(indices), 3))

    def _apply_transient_hits(self, rgb: np.ndarray, count: int) -> None:
        height, width = rgb.shape[:2]
        indices = self._rng.integers(0, height * width, size=count)
        rows, columns = np.divmod(indices, width)
        amplitudes = self._rng.uniform(0.6, 1.8, size=count)
        rgb[rows, columns] += amplitudes[:, None]
        max_length = min(height, int(max(4.0, self.config.streak_decay_px * 6.0)))
        for row, column, amplitude in zip(rows, columns, amplitudes, strict=True):
            length = min(max_length, height - int(row))
            offsets = np.arange(length)
            decay = amplitude * np.exp(-offsets / self.config.streak_decay_px)
            rgb[int(row) : int(row) + length, int(column)] += decay[:, None]

    @staticmethod
    def _to_float(image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, float]:
        scale = float(np.iinfo(image.dtype).max) if np.issubdtype(image.dtype, np.integer) else 1.0
        rgb = image[..., :3].astype(np.float32) / scale
        alpha = image[..., 3:4].copy() if image.shape[2] == 4 else None
        return rgb, alpha, scale

    @staticmethod
    def _from_float(
        rgb: np.ndarray,
        alpha: np.ndarray | None,
        dtype: np.dtype[Any],
        scale: float,
    ) -> np.ndarray:
        converted = rgb * scale
        if np.issubdtype(dtype, np.integer):
            converted = np.rint(converted)
        converted = converted.astype(dtype)
        return np.concatenate((converted, alpha), axis=2) if alpha is not None else converted
