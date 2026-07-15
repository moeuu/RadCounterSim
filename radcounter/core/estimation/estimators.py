"""Nonnegative Poisson sparse and surface-TV estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize

from radcounter.core.estimation.basis import CandidateBasis
from radcounter.core.estimation.poisson import (
    PoissonInverseProblem,
    fisher_covariance,
    poisson_nll_and_gradient,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PointHypothesis:
    """Connected active component summarized as one source hypothesis."""

    position_world_m: FloatArray
    source_strength_bq: float
    basis_indexes: NDArray[np.int64]


@dataclass(frozen=True)
class SourceEstimate:
    """Serializable estimator output."""

    estimate_id: str
    basis_activity_bq: FloatArray
    covariance_diag_bq2: FloatArray
    covariance_bq2: FloatArray
    point_hypotheses: tuple[PointHypothesis, ...]
    predicted_measurements: FloatArray
    objective_value: float
    converged: bool
    diagnostics: dict[str, Any]


class SourceEstimator(Protocol):
    """Estimator contract intentionally excludes simulator truth."""

    def fit(
        self,
        problem: PoissonInverseProblem,
        basis: CandidateBasis,
        initial_strength_bq: FloatArray | None = None,
    ) -> SourceEstimate: ...


def _initial_strength(
    problem: PoissonInverseProblem, initial_strength_bq: FloatArray | None
) -> FloatArray:
    if initial_strength_bq is not None:
        initial = np.asarray(initial_strength_bq, dtype=np.float64)
        if initial.shape != (problem.candidate_count,) or np.any(initial < 0):
            raise ValueError("initial source strength is invalid")
        return initial.copy()
    corrected = np.maximum(problem.observed_counts - problem.background_counts, 0.0)
    least_squares, *_ = np.linalg.lstsq(problem.response_counts_per_bq, corrected, rcond=None)
    return np.maximum(least_squares, 0.0)


def _point_hypotheses(
    basis: CandidateBasis, strength_bq: FloatArray, active_threshold_bq: float
) -> tuple[PointHypothesis, ...]:
    relative_threshold = max(active_threshold_bq, float(strength_bq.max(initial=0.0)) * 1e-6)
    components = basis.connected_components(strength_bq > relative_threshold)
    hypotheses: list[PointHypothesis] = []
    for indexes in components:
        weights = strength_bq[indexes]
        total = float(weights.sum())
        position = np.average(basis.positions_world_m[indexes], axis=0, weights=weights)
        hypotheses.append(PointHypothesis(position, total, indexes))
    return tuple(hypotheses)


class GridPoissonSparseEstimator:
    """Nonnegative Poisson MLE with optional L1 sparsity."""

    def __init__(
        self,
        *,
        lambda_l1: float = 0.0,
        active_threshold_bq: float = 1e-6,
        maximum_iterations: int = 1000,
    ) -> None:
        if min(lambda_l1, active_threshold_bq) < 0 or maximum_iterations < 1:
            raise ValueError("estimator settings are invalid")
        self.lambda_l1 = lambda_l1
        self.active_threshold_bq = active_threshold_bq
        self.maximum_iterations = maximum_iterations

    def _objective_gradient(
        self, strength_bq: FloatArray, problem: PoissonInverseProblem, basis: CandidateBasis
    ) -> tuple[float, FloatArray]:
        del basis
        return poisson_nll_and_gradient(strength_bq, problem, lambda_l1=self.lambda_l1)

    def fit(
        self,
        problem: PoissonInverseProblem,
        basis: CandidateBasis,
        initial_strength_bq: FloatArray | None = None,
    ) -> SourceEstimate:
        """Fit source strengths without source-count truth."""

        if problem.candidate_count != basis.size:
            raise ValueError("response matrix columns must match candidate basis")
        initial = _initial_strength(problem, initial_strength_bq)

        def objective(value: FloatArray) -> tuple[float, FloatArray]:
            return self._objective_gradient(value, problem, basis)

        result: OptimizeResult = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, None)] * basis.size,
            options={"maxiter": self.maximum_iterations, "ftol": 1e-12, "gtol": 1e-8},
        )
        strength = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
        covariance, active = fisher_covariance(
            strength, problem, active_threshold_bq=self.active_threshold_bq
        )
        predicted = problem.response_counts_per_bq @ strength + problem.background_counts
        return SourceEstimate(
            str(uuid4()),
            strength,
            np.diag(covariance),
            covariance,
            _point_hypotheses(basis, strength, self.active_threshold_bq),
            predicted,
            float(result.fun),
            bool(result.success),
            {
                "message": str(result.message),
                "iterations": int(result.nit),
                "active_candidates": int(active.sum()),
                "lambda_l1": self.lambda_l1,
                "solver": "L-BFGS-B",
            },
        )

    def bootstrap(
        self,
        estimate: SourceEstimate,
        problem: PoissonInverseProblem,
        basis: CandidateBasis,
        *,
        replicates: int,
        rng: np.random.Generator,
    ) -> FloatArray:
        """Parametric Poisson bootstrap of estimated source strengths."""

        if replicates < 1:
            raise ValueError("bootstrap replicates must be positive")
        samples = np.zeros((replicates, basis.size), dtype=np.float64)
        for index in range(replicates):
            bootstrap_problem = PoissonInverseProblem(
                rng.poisson(estimate.predicted_measurements),
                problem.response_counts_per_bq,
                problem.background_counts,
            )
            samples[index] = self.fit(
                bootstrap_problem, basis, estimate.basis_activity_bq
            ).basis_activity_bq
        return samples


class SurfacePoissonTVEstimator(GridPoissonSparseEstimator):
    """Poisson estimator with smooth graph-TV approximation."""

    def __init__(
        self,
        *,
        lambda_l1: float = 0.0,
        lambda_tv: float = 0.0,
        tv_smoothing_bq: float = 1e-6,
        active_threshold_bq: float = 1e-6,
        maximum_iterations: int = 1000,
    ) -> None:
        super().__init__(
            lambda_l1=lambda_l1,
            active_threshold_bq=active_threshold_bq,
            maximum_iterations=maximum_iterations,
        )
        if lambda_tv < 0 or tv_smoothing_bq <= 0:
            raise ValueError("TV settings are invalid")
        self.lambda_tv = lambda_tv
        self.tv_smoothing_bq = tv_smoothing_bq

    def _objective_gradient(
        self, strength_bq: FloatArray, problem: PoissonInverseProblem, basis: CandidateBasis
    ) -> tuple[float, FloatArray]:
        objective, gradient = poisson_nll_and_gradient(
            strength_bq, problem, lambda_l1=self.lambda_l1
        )
        if self.lambda_tv == 0 or len(basis.adjacency_edges) == 0:
            return objective, gradient
        left = basis.adjacency_edges[:, 0]
        right = basis.adjacency_edges[:, 1]
        difference = strength_bq[left] - strength_bq[right]
        smooth_norm = np.sqrt(difference**2 + self.tv_smoothing_bq**2)
        objective += self.lambda_tv * float(smooth_norm.sum())
        edge_gradient = self.lambda_tv * difference / smooth_norm
        np.add.at(gradient, left, edge_gradient)
        np.add.at(gradient, right, -edge_gradient)
        return objective, gradient

    def fit(
        self,
        problem: PoissonInverseProblem,
        basis: CandidateBasis,
        initial_strength_bq: FloatArray | None = None,
    ) -> SourceEstimate:
        estimate = super().fit(problem, basis, initial_strength_bq)
        estimate.diagnostics.update(
            {
                "lambda_tv": self.lambda_tv,
                "tv_approximation": "smooth_graph_total_variation",
                "tv_smoothing_bq": self.tv_smoothing_bq,
            }
        )
        return estimate
