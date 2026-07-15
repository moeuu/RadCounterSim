"""Poisson inverse-problem validation and objective derivatives."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PoissonInverseProblem:
    """Stacked count observations with a linear nonnegative response."""

    observed_counts: FloatArray
    response_counts_per_bq: FloatArray
    background_counts: FloatArray

    def __post_init__(self) -> None:
        observed = np.asarray(self.observed_counts, dtype=np.float64)
        response = np.asarray(self.response_counts_per_bq, dtype=np.float64)
        background = np.asarray(self.background_counts, dtype=np.float64)
        if observed.ndim != 1 or response.ndim != 2:
            raise ValueError("observed counts must be 1-D and response matrix 2-D")
        if response.shape[0] != len(observed) or background.shape != observed.shape:
            raise ValueError("Poisson problem row dimensions are inconsistent")
        if (
            np.any(observed < 0)
            or np.any(response < 0)
            or np.any(background < 0)
            or not np.all(np.isfinite(response))
        ):
            raise ValueError("Poisson observations/response/background must be finite nonnegative")
        object.__setattr__(self, "observed_counts", observed)
        object.__setattr__(self, "response_counts_per_bq", response)
        object.__setattr__(self, "background_counts", background)

    @property
    def candidate_count(self) -> int:
        return self.response_counts_per_bq.shape[1]

    @classmethod
    def stack(
        cls,
        observed_blocks: Sequence[FloatArray],
        response_blocks: Sequence[FloatArray],
        background_blocks: Sequence[FloatArray],
    ) -> PoissonInverseProblem:
        """Stack pose/time/bin measurement blocks."""

        if not (len(observed_blocks) == len(response_blocks) == len(background_blocks) > 0):
            raise ValueError("measurement block sequences must be equal and non-empty")
        return cls(
            np.concatenate(observed_blocks),
            np.vstack(response_blocks),
            np.concatenate(background_blocks),
        )


def poisson_nll_and_gradient(
    source_strength_bq: FloatArray,
    problem: PoissonInverseProblem,
    *,
    lambda_l1: float = 0.0,
    minimum_expected_count: float = 1e-12,
) -> tuple[float, FloatArray]:
    """Return Poisson negative log likelihood and analytic gradient."""

    strength = np.asarray(source_strength_bq, dtype=np.float64)
    if strength.shape != (problem.candidate_count,) or np.any(strength < 0):
        raise ValueError("source strength must be a nonnegative candidate vector")
    if lambda_l1 < 0 or minimum_expected_count <= 0:
        raise ValueError("regularization/count floor are invalid")
    expected = np.maximum(
        problem.response_counts_per_bq @ strength + problem.background_counts,
        minimum_expected_count,
    )
    objective = float(
        np.sum(expected - problem.observed_counts * np.log(expected)) + lambda_l1 * strength.sum()
    )
    gradient = problem.response_counts_per_bq.T @ (1.0 - problem.observed_counts / expected)
    gradient += lambda_l1
    return objective, gradient


def fisher_covariance(
    source_strength_bq: FloatArray,
    problem: PoissonInverseProblem,
    *,
    active_threshold_bq: float = 1e-9,
    regularization: float = 1e-9,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Compute active-set Fisher pseudo-inverse in full candidate coordinates."""

    strength = np.asarray(source_strength_bq, dtype=np.float64)
    active = strength > active_threshold_bq
    covariance = np.zeros((problem.candidate_count, problem.candidate_count), dtype=np.float64)
    if not np.any(active):
        return covariance, active
    expected = np.maximum(
        problem.response_counts_per_bq @ strength + problem.background_counts, 1e-12
    )
    active_response = problem.response_counts_per_bq[:, active]
    fisher = active_response.T @ (active_response / expected[:, None])
    fisher += regularization * np.eye(fisher.shape[0])
    active_covariance = np.linalg.pinv(fisher, hermitian=True)
    covariance[np.ix_(active, active)] = active_covariance
    return covariance, active
