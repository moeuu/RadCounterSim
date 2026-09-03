"""Poisson source estimation with an explicit nonparalyzable dead-time model."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize

from radcounter.core.estimation.basis import CandidateBasis
from radcounter.core.estimation.estimators import PointHypothesis, SourceEstimate

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DeadTimePoissonInverseProblem:
    """Count observations and a linear pre-electronics source response.

    For detector row ``i``, the expected count is

    ``duration * rate / (1 + dead_time * rate)``

    where ``rate = background + response @ activity``. This retains the
    detector nonlinearity instead of treating a dead-time-affected spectrum as
    a linear response matrix.
    """

    observed_counts: FloatArray
    source_rate_cps_per_bq: FloatArray
    background_rate_cps: FloatArray
    duration_s: FloatArray
    dead_time_s: FloatArray

    def __post_init__(self) -> None:
        observed = np.asarray(self.observed_counts, dtype=np.float64)
        response = np.asarray(self.source_rate_cps_per_bq, dtype=np.float64)
        background = np.asarray(self.background_rate_cps, dtype=np.float64)
        duration = np.asarray(self.duration_s, dtype=np.float64)
        dead_time = np.asarray(self.dead_time_s, dtype=np.float64)
        if observed.ndim != 1 or response.ndim != 2:
            raise ValueError("dead-time observations must be 1-D and response must be 2-D")
        if response.shape[0] != len(observed):
            raise ValueError("dead-time response rows must match observations")
        if any(array.shape != observed.shape for array in (background, duration, dead_time)):
            raise ValueError("dead-time background, duration, and electronics rows must match")
        if (
            not np.all(np.isfinite(observed))
            or not np.all(np.isfinite(response))
            or not np.all(np.isfinite(background))
            or not np.all(np.isfinite(duration))
            or not np.all(np.isfinite(dead_time))
            or np.any(observed < 0.0)
            or np.any(response < 0.0)
            or np.any(background < 0.0)
            or np.any(duration <= 0.0)
            or np.any(dead_time < 0.0)
        ):
            raise ValueError("dead-time inverse-problem values are invalid")
        object.__setattr__(self, "observed_counts", observed)
        object.__setattr__(self, "source_rate_cps_per_bq", response)
        object.__setattr__(self, "background_rate_cps", background)
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "dead_time_s", dead_time)

    @property
    def candidate_count(self) -> int:
        return self.source_rate_cps_per_bq.shape[1]

    def expected_counts(self, source_strength_bq: FloatArray) -> FloatArray:
        strength = np.asarray(source_strength_bq, dtype=np.float64)
        if strength.shape != (self.candidate_count,) or np.any(strength < 0.0):
            raise ValueError("source strength must match candidates and be nonnegative")
        incident_rate = self.background_rate_cps + self.source_rate_cps_per_bq @ strength
        return self.duration_s * incident_rate / (1.0 + self.dead_time_s * incident_rate)

    def count_jacobian(self, source_strength_bq: FloatArray) -> FloatArray:
        strength = np.asarray(source_strength_bq, dtype=np.float64)
        incident_rate = self.background_rate_cps + self.source_rate_cps_per_bq @ strength
        electronics_denominator = (1.0 + self.dead_time_s * incident_rate) ** 2
        return (
            self.duration_s[:, None]
            * self.source_rate_cps_per_bq
            / electronics_denominator[:, None]
        )


class DeadTimePoissonEstimator:
    """Nonnegative Poisson MLE evaluated through detector dead time."""

    def __init__(
        self,
        *,
        lambda_l1: float = 0.0,
        maximum_iterations: int = 1000,
        active_threshold_bq: float = 1.0e-6,
    ) -> None:
        if lambda_l1 < 0.0 or maximum_iterations < 1 or active_threshold_bq < 0.0:
            raise ValueError("dead-time estimator settings are invalid")
        self.lambda_l1 = lambda_l1
        self.maximum_iterations = maximum_iterations
        self.active_threshold_bq = active_threshold_bq

    @staticmethod
    def _initial_strength(problem: DeadTimePoissonInverseProblem) -> FloatArray:
        observed_rate = problem.observed_counts / problem.duration_s
        corrected_rate = observed_rate.copy()
        active_dead_time = problem.dead_time_s > 0.0
        if np.any(active_dead_time):
            product = problem.dead_time_s[active_dead_time] * observed_rate[active_dead_time]
            product = np.minimum(product, 1.0 - 1.0e-9)
            corrected_rate[active_dead_time] = observed_rate[active_dead_time] / (1.0 - product)
        source_rate = np.maximum(corrected_rate - problem.background_rate_cps, 0.0)
        initial, *_ = np.linalg.lstsq(
            problem.source_rate_cps_per_bq,
            source_rate,
            rcond=None,
        )
        return np.maximum(initial, 0.0)

    def fit(
        self,
        problem: DeadTimePoissonInverseProblem,
        basis: CandidateBasis,
        initial_strength_bq: FloatArray | None = None,
    ) -> SourceEstimate:
        if basis.size != problem.candidate_count:
            raise ValueError("dead-time response columns must match candidate basis")
        if initial_strength_bq is None:
            initial = self._initial_strength(problem)
        else:
            initial = np.asarray(initial_strength_bq, dtype=np.float64)
            if initial.shape != (basis.size,) or np.any(initial < 0.0):
                raise ValueError("initial source strength is invalid")

        def objective(strength: FloatArray) -> tuple[float, FloatArray]:
            expected = np.maximum(problem.expected_counts(strength), 1.0e-12)
            jacobian = problem.count_jacobian(strength)
            value = float(
                np.sum(expected - problem.observed_counts * np.log(expected))
                + self.lambda_l1 * np.sum(strength)
            )
            gradient = jacobian.T @ (1.0 - problem.observed_counts / expected)
            gradient += self.lambda_l1
            return value, gradient

        result: OptimizeResult = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, None)] * basis.size,
            options={
                "maxiter": self.maximum_iterations,
                "ftol": 1.0e-12,
                "gtol": 1.0e-8,
            },
        )
        strength = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
        predicted = problem.expected_counts(strength)
        jacobian = problem.count_jacobian(strength)
        fisher = jacobian.T @ (jacobian / np.maximum(predicted, 1.0e-12)[:, None])
        covariance = np.linalg.pinv(
            fisher + np.eye(basis.size) * 1.0e-12,
            hermitian=True,
        )
        relative_threshold = max(
            self.active_threshold_bq,
            float(strength.max(initial=0.0)) * 1.0e-6,
        )
        hypotheses: list[PointHypothesis] = []
        for indexes in basis.connected_components(strength > relative_threshold):
            weights = strength[indexes]
            hypotheses.append(
                PointHypothesis(
                    np.average(basis.positions_world_m[indexes], axis=0, weights=weights),
                    float(np.sum(weights)),
                    indexes,
                )
            )
        return SourceEstimate(
            estimate_id=str(uuid4()),
            basis_activity_bq=strength,
            covariance_diag_bq2=np.diag(covariance),
            covariance_bq2=covariance,
            point_hypotheses=tuple(hypotheses),
            predicted_measurements=predicted,
            objective_value=float(result.fun),
            converged=bool(result.success),
            diagnostics={
                "message": str(result.message),
                "iterations": int(result.nit),
                "solver": "nonparalyzable_dead_time_poisson_L-BFGS-B",
                "lambda_l1": self.lambda_l1,
            },
        )
