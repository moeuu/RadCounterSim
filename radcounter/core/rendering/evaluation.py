"""Quantitative comparison of simulated and real facility sensor frames."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RgbComparison:
    mae: float
    rmse: float
    psnr_db: float
    global_ssim: float
    edge_mae: float
    histogram_js_divergence: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class DepthComparison:
    valid_fraction: float
    mae_m: float
    rmse_m: float
    absolute_relative_error: float
    delta_1_25: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _rgb_float(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)[..., :3]
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("RGB input must have shape (height, width, 3|4)")
    if np.issubdtype(value.dtype, np.integer):
        return value.astype(np.float64) / np.iinfo(value.dtype).max
    return np.clip(value.astype(np.float64), 0.0, 1.0)


def _global_ssim(reference: np.ndarray, simulated: np.ndarray) -> float:
    c1, c2 = 0.01**2, 0.03**2
    scores = []
    for channel in range(3):
        left = reference[..., channel]
        right = simulated[..., channel]
        mu_left, mu_right = float(left.mean()), float(right.mean())
        variance_left, variance_right = float(left.var()), float(right.var())
        covariance = float(np.mean((left - mu_left) * (right - mu_right)))
        numerator = (2.0 * mu_left * mu_right + c1) * (2.0 * covariance + c2)
        denominator = (mu_left**2 + mu_right**2 + c1) * (variance_left + variance_right + c2)
        scores.append(numerator / denominator)
    return float(np.mean(scores))


def _edge_magnitude(image: np.ndarray) -> np.ndarray:
    luminance = np.sum(image * np.asarray([0.2126, 0.7152, 0.0722]), axis=2)
    dx = np.diff(luminance, axis=1, append=luminance[:, -1:])
    dy = np.diff(luminance, axis=0, append=luminance[-1:, :])
    return np.hypot(dx, dy)


def _histogram_js(reference: np.ndarray, simulated: np.ndarray, bins: int = 64) -> float:
    values = []
    epsilon = 1.0e-12
    for channel in range(3):
        left, _ = np.histogram(reference[..., channel], bins=bins, range=(0.0, 1.0))
        right, _ = np.histogram(simulated[..., channel], bins=bins, range=(0.0, 1.0))
        p = left.astype(np.float64) + epsilon
        q = right.astype(np.float64) + epsilon
        p /= p.sum()
        q /= q.sum()
        midpoint = 0.5 * (p + q)
        values.append(
            0.5 * np.sum(p * np.log(p / midpoint)) + 0.5 * np.sum(q * np.log(q / midpoint))
        )
    return float(np.mean(values))


def compare_rgb(reference: np.ndarray, simulated: np.ndarray) -> RgbComparison:
    left = _rgb_float(reference)
    right = _rgb_float(simulated)
    if left.shape != right.shape:
        raise ValueError(f"RGB shape mismatch: {left.shape} != {right.shape}")
    difference = right - left
    mse = float(np.mean(difference**2))
    return RgbComparison(
        mae=float(np.mean(np.abs(difference))),
        rmse=float(np.sqrt(mse)),
        psnr_db=float("inf") if mse == 0.0 else float(10.0 * np.log10(1.0 / mse)),
        global_ssim=_global_ssim(left, right),
        edge_mae=float(np.mean(np.abs(_edge_magnitude(left) - _edge_magnitude(right)))),
        histogram_js_divergence=_histogram_js(left, right),
    )


def compare_depth(reference_m: np.ndarray, simulated_m: np.ndarray) -> DepthComparison:
    left = np.asarray(reference_m, dtype=np.float64)
    right = np.asarray(simulated_m, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"depth shape mismatch: {left.shape} != {right.shape}")
    valid = np.isfinite(left) & np.isfinite(right) & (left > 0.0) & (right > 0.0)
    valid_fraction = float(np.mean(valid))
    if not np.any(valid):
        return DepthComparison(valid_fraction, float("nan"), float("nan"), float("nan"), 0.0)
    difference = right[valid] - left[valid]
    ratio = np.maximum(right[valid] / left[valid], left[valid] / right[valid])
    return DepthComparison(
        valid_fraction=valid_fraction,
        mae_m=float(np.mean(np.abs(difference))),
        rmse_m=float(np.sqrt(np.mean(difference**2))),
        absolute_relative_error=float(np.mean(np.abs(difference) / left[valid])),
        delta_1_25=float(np.mean(ratio < 1.25)),
    )
