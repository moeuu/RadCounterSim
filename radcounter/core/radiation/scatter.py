"""Explicit scatter and model-mismatch plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class ScatterModel(Protocol):
    """Named plugin that augments direct count-rate predictions."""

    model_name: str

    def add_scatter(self, direct_rate_cps_per_bin: FloatArray) -> FloatArray: ...


@dataclass(frozen=True)
class NoScatterModel:
    """Explicitly configured no-scatter model."""

    model_name: str = "none"

    def add_scatter(self, direct_rate_cps_per_bin: FloatArray) -> FloatArray:
        return np.asarray(direct_rate_cps_per_bin, dtype=np.float64).copy()


@dataclass(frozen=True)
class EmpiricalBuildupModel:
    """Apply a nonnegative bin-wise empirical buildup factor."""

    buildup_factor_per_bin: FloatArray
    model_name: str = "empirical_buildup"

    def add_scatter(self, direct_rate_cps_per_bin: FloatArray) -> FloatArray:
        direct = np.asarray(direct_rate_cps_per_bin, dtype=np.float64)
        factor = np.asarray(self.buildup_factor_per_bin, dtype=np.float64)
        if factor.shape != direct.shape[-1:] or np.any(factor < 1.0):
            raise ValueError("buildup factor must match bins and be at least one")
        return direct * factor


@dataclass(frozen=True)
class TruthOnlyBiasModel:
    """Inject reproducible truth-only gain/background mismatch."""

    gain_per_bin: FloatArray
    background_drift_cps_per_bin: FloatArray
    model_name: str = "truth_only_bias"

    def add_scatter(self, direct_rate_cps_per_bin: FloatArray) -> FloatArray:
        direct = np.asarray(direct_rate_cps_per_bin, dtype=np.float64)
        gain = np.asarray(self.gain_per_bin, dtype=np.float64)
        drift = np.asarray(self.background_drift_cps_per_bin, dtype=np.float64)
        if gain.shape != direct.shape[-1:] or drift.shape != direct.shape[-1:]:
            raise ValueError("truth-only bias arrays must match energy bins")
        if np.any(gain < 0) or np.any(drift < 0):
            raise ValueError("truth-only gain/background must be nonnegative")
        return direct * gain + drift
