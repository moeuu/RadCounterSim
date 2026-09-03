"""Continuous-coordinate point-source refinement and particle-assisted MLE."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from radcounter.core.estimation.basis import CandidateBasis
from radcounter.core.estimation.estimators import (
    GridPoissonSparseEstimator,
    PointHypothesis,
    SourceEstimate,
)
from radcounter.core.estimation.poisson import PoissonInverseProblem


class ContinuousPointResponseModel(Protocol):
    """Public response provider for a unit point source at a continuous position."""

    measurement_count: int

    def response_counts_per_bq(self, position_world_m: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class InverseSquarePointResponseModel:
    """Analytic unit response for tests and explicitly selected free-space studies."""

    detector_positions_world_m: np.ndarray
    sensitivity_counts_m2_per_bq: np.ndarray
    minimum_distance_m: float = 0.01

    def __post_init__(self) -> None:
        positions = np.asarray(self.detector_positions_world_m, dtype=np.float64)
        sensitivity = np.asarray(self.sensitivity_counts_m2_per_bq, dtype=np.float64)
        if (
            positions.ndim != 2
            or positions.shape[1:] != (3,)
            or sensitivity.shape != (len(positions),)
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(sensitivity))
            or np.any(sensitivity < 0.0)
            or self.minimum_distance_m <= 0.0
        ):
            raise ValueError("continuous inverse-square response configuration is invalid")
        object.__setattr__(self, "detector_positions_world_m", positions)
        object.__setattr__(self, "sensitivity_counts_m2_per_bq", sensitivity)

    @property
    def measurement_count(self) -> int:
        return len(self.detector_positions_world_m)

    def response_counts_per_bq(self, position_world_m: np.ndarray) -> np.ndarray:
        position = np.asarray(position_world_m, dtype=np.float64)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("continuous source position must be a finite 3-vector")
        distance = np.maximum(
            np.linalg.norm(self.detector_positions_world_m - position[None, :], axis=1),
            self.minimum_distance_m,
        )
        return self.sensitivity_counts_m2_per_bq / (4.0 * math.pi * distance**2)


@dataclass(frozen=True, slots=True)
class RefinedPointHypothesis:
    position_world_m: np.ndarray
    source_strength_bq: float
    parameter_covariance: np.ndarray
    basis_indexes: np.ndarray
    objective_value: float
    converged: bool
    diagnostics: dict[str, object]


class ContinuousPointMLERefiner:
    """Refine a connected grid component in continuous XYZ/activity coordinates."""

    def __init__(
        self,
        response_model: ContinuousPointResponseModel,
        bounds_xyz_m: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        *,
        maximum_iterations: int = 500,
        finite_difference_step_m: float = 1.0e-4,
    ) -> None:
        if maximum_iterations < 1 or finite_difference_step_m <= 0.0:
            raise ValueError("continuous refinement settings are invalid")
        if any(
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or upper <= lower
            for lower, upper in bounds_xyz_m
        ):
            raise ValueError("continuous refinement bounds are invalid")
        self.response_model = response_model
        self.bounds_xyz_m = bounds_xyz_m
        self.maximum_iterations = maximum_iterations
        self.finite_difference_step_m = finite_difference_step_m

    @staticmethod
    def _poisson_nll(observed: np.ndarray, expected: np.ndarray) -> float:
        expected = np.maximum(expected, 1.0e-12)
        return float(np.sum(expected - observed * np.log(expected)))

    def refine_point(
        self,
        observed_counts: np.ndarray,
        base_counts: np.ndarray,
        starting_position_world_m: np.ndarray,
        starting_strength_bq: float,
        basis_indexes: np.ndarray,
    ) -> RefinedPointHypothesis:
        observed = np.asarray(observed_counts, dtype=np.float64)
        base = np.asarray(base_counts, dtype=np.float64)
        start_position = np.asarray(starting_position_world_m, dtype=np.float64)
        if (
            observed.shape != (self.response_model.measurement_count,)
            or base.shape != observed.shape
            or np.any(observed < 0.0)
            or np.any(base < 0.0)
            or start_position.shape != (3,)
            or starting_strength_bq <= 0.0
        ):
            raise ValueError("continuous point-refinement inputs are invalid")
        clipped_start = np.asarray(
            [
                np.clip(value, lower, upper)
                for value, (lower, upper) in zip(
                    start_position, self.bounds_xyz_m, strict=True
                )
            ]
        )
        log_strength_bounds = (
            math.log(max(starting_strength_bq * 1.0e-6, 1.0e-12)),
            math.log(max(starting_strength_bq * 1.0e3, 1.0)),
        )

        def objective(parameters: np.ndarray) -> float:
            position = parameters[:3]
            strength = math.exp(float(parameters[3]))
            response = self.response_model.response_counts_per_bq(position)
            return self._poisson_nll(observed, base + response * strength)

        result: OptimizeResult = minimize(
            objective,
            np.concatenate((clipped_start, (math.log(starting_strength_bq),))),
            method="L-BFGS-B",
            bounds=(*self.bounds_xyz_m, log_strength_bounds),
            options={"maxiter": self.maximum_iterations, "ftol": 1e-12, "gtol": 1e-8},
        )
        position = np.asarray(result.x[:3], dtype=np.float64)
        strength = math.exp(float(result.x[3]))
        covariance = self._fisher_covariance(position, strength, base)
        return RefinedPointHypothesis(
            position,
            strength,
            covariance,
            np.asarray(basis_indexes, dtype=np.int64),
            float(result.fun),
            bool(result.success),
            {"message": str(result.message), "iterations": int(result.nit)},
        )

    def _fisher_covariance(
        self, position: np.ndarray, strength: float, base_counts: np.ndarray
    ) -> np.ndarray:
        response = self.response_model.response_counts_per_bq(position)
        jacobian = np.empty((len(response), 4), dtype=np.float64)
        for axis in range(3):
            step = min(
                self.finite_difference_step_m,
                (self.bounds_xyz_m[axis][1] - self.bounds_xyz_m[axis][0]) * 1.0e-3,
            )
            plus = position.copy()
            minus = position.copy()
            plus[axis] = min(plus[axis] + step, self.bounds_xyz_m[axis][1])
            minus[axis] = max(minus[axis] - step, self.bounds_xyz_m[axis][0])
            denominator = plus[axis] - minus[axis]
            if denominator <= 0.0:
                jacobian[:, axis] = 0.0
            else:
                jacobian[:, axis] = strength * (
                    self.response_model.response_counts_per_bq(plus)
                    - self.response_model.response_counts_per_bq(minus)
                ) / denominator
        jacobian[:, 3] = response
        expected = np.maximum(base_counts + response * strength, 1.0e-12)
        fisher = jacobian.T @ (jacobian / expected[:, None])
        return np.linalg.pinv(fisher + np.eye(4) * 1.0e-12, hermitian=True)

    def refine(
        self,
        problem: PoissonInverseProblem,
        basis: CandidateBasis,
        coarse_estimate: SourceEstimate,
    ) -> tuple[RefinedPointHypothesis, ...]:
        if self.response_model.measurement_count != len(problem.observed_counts):
            raise ValueError("continuous response row count does not match observations")
        refined: list[RefinedPointHypothesis] = []
        for hypothesis in coarse_estimate.point_hypotheses:
            component_counts = (
                problem.response_counts_per_bq[:, hypothesis.basis_indexes]
                @ coarse_estimate.basis_activity_bq[hypothesis.basis_indexes]
            )
            base = np.maximum(coarse_estimate.predicted_measurements - component_counts, 0.0)
            refined.append(
                self.refine_point(
                    problem.observed_counts,
                    base,
                    hypothesis.position_world_m,
                    hypothesis.source_strength_bq,
                    hypothesis.basis_indexes,
                )
            )
        return tuple(refined)


class PFPlusMLEEstimator:
    """Particle exploration followed by continuous-coordinate Poisson MLE."""

    def __init__(
        self,
        response_model: ContinuousPointResponseModel,
        bounds_xyz_m: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        *,
        particle_count: int = 256,
        particle_rounds: int = 4,
        position_jitter_m: float = 0.25,
        strength_log_jitter: float = 0.35,
        random_seed: int = 0,
        grid_estimator: GridPoissonSparseEstimator | None = None,
    ) -> None:
        if (
            particle_count < 8
            or particle_rounds < 1
            or position_jitter_m <= 0.0
            or strength_log_jitter <= 0.0
        ):
            raise ValueError("particle-assisted estimator settings are invalid")
        self.response_model = response_model
        self.refiner = ContinuousPointMLERefiner(response_model, bounds_xyz_m)
        self.particle_count = particle_count
        self.particle_rounds = particle_rounds
        self.position_jitter_m = position_jitter_m
        self.strength_log_jitter = strength_log_jitter
        self.random_seed = random_seed
        self.grid_estimator = grid_estimator or GridPoissonSparseEstimator()

    def _particle_start(
        self,
        observed: np.ndarray,
        base: np.ndarray,
        hypothesis: PointHypothesis,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float, float]:
        positions = rng.normal(
            hypothesis.position_world_m,
            self.position_jitter_m,
            size=(self.particle_count, 3),
        )
        for axis, (lower, upper) in enumerate(self.refiner.bounds_xyz_m):
            positions[:, axis] = np.clip(positions[:, axis], lower, upper)
        strengths = np.exp(
            rng.normal(
                math.log(max(hypothesis.source_strength_bq, 1.0e-12)),
                self.strength_log_jitter,
                size=self.particle_count,
            )
        )
        effective_sample_size = 0.0
        for round_index in range(self.particle_rounds):
            log_weights = np.empty(self.particle_count, dtype=np.float64)
            for index in range(self.particle_count):
                expected = base + self.response_model.response_counts_per_bq(
                    positions[index]
                ) * strengths[index]
                log_weights[index] = -ContinuousPointMLERefiner._poisson_nll(
                    observed, expected
                )
            log_weights -= np.max(log_weights)
            weights = np.exp(log_weights)
            weights /= np.sum(weights)
            effective_sample_size = float(1.0 / np.sum(weights**2))
            indexes = rng.choice(
                self.particle_count,
                size=self.particle_count,
                replace=True,
                p=weights,
            )
            positions = positions[indexes]
            strengths = strengths[indexes]
            shrink = 0.55 ** (round_index + 1)
            positions += rng.normal(
                0.0,
                self.position_jitter_m * shrink,
                size=positions.shape,
            )
            strengths *= np.exp(
                rng.normal(
                    0.0,
                    self.strength_log_jitter * shrink,
                    size=self.particle_count,
                )
            )
            for axis, (lower, upper) in enumerate(self.refiner.bounds_xyz_m):
                positions[:, axis] = np.clip(positions[:, axis], lower, upper)
        scores = np.asarray(
            [
                ContinuousPointMLERefiner._poisson_nll(
                    observed,
                    base
                    + self.response_model.response_counts_per_bq(position) * strength,
                )
                for position, strength in zip(positions, strengths, strict=True)
            ]
        )
        best = int(np.argmin(scores))
        return positions[best], float(strengths[best]), effective_sample_size

    def fit(
        self,
        problem: PoissonInverseProblem,
        basis: CandidateBasis,
        initial_strength_bq: np.ndarray | None = None,
    ) -> SourceEstimate:
        coarse = self.grid_estimator.fit(problem, basis, initial_strength_bq)
        rng = np.random.default_rng(self.random_seed)
        refined: list[RefinedPointHypothesis] = []
        effective_sample_sizes: list[float] = []
        for hypothesis in coarse.point_hypotheses:
            component = (
                problem.response_counts_per_bq[:, hypothesis.basis_indexes]
                @ coarse.basis_activity_bq[hypothesis.basis_indexes]
            )
            base = np.maximum(coarse.predicted_measurements - component, 0.0)
            start_position, start_strength, effective_size = self._particle_start(
                problem.observed_counts,
                base,
                hypothesis,
                rng,
            )
            effective_sample_sizes.append(effective_size)
            refined.append(
                self.refiner.refine_point(
                    problem.observed_counts,
                    base,
                    start_position,
                    start_strength,
                    hypothesis.basis_indexes,
                )
            )
        adjusted_basis = coarse.basis_activity_bq.copy()
        point_hypotheses: list[PointHypothesis] = []
        for original, result in zip(coarse.point_hypotheses, refined, strict=True):
            weights = adjusted_basis[original.basis_indexes]
            if np.sum(weights) > 0.0:
                adjusted_basis[original.basis_indexes] = (
                    weights / np.sum(weights) * result.source_strength_bq
                )
            point_hypotheses.append(
                PointHypothesis(
                    result.position_world_m,
                    result.source_strength_bq,
                    result.basis_indexes,
                )
            )
        predicted = problem.background_counts.copy()
        active_indexes = set()
        for result in refined:
            predicted += (
                self.response_model.response_counts_per_bq(result.position_world_m)
                * result.source_strength_bq
            )
            active_indexes.update(int(index) for index in result.basis_indexes)
        unused = np.asarray(
            [index for index in range(basis.size) if index not in active_indexes],
            dtype=np.int64,
        )
        if len(unused):
            predicted += problem.response_counts_per_bq[:, unused] @ adjusted_basis[unused]
        objective = ContinuousPointMLERefiner._poisson_nll(
            problem.observed_counts, predicted
        )
        diagnostics = dict(coarse.diagnostics)
        diagnostics.update(
            {
                "solver": "particle_exploration_plus_continuous_L-BFGS-B",
                "particle_count": self.particle_count,
                "particle_rounds": self.particle_rounds,
                "effective_sample_size": effective_sample_sizes,
                "continuous_point_covariance": [
                    result.parameter_covariance.tolist() for result in refined
                ],
            }
        )
        return SourceEstimate(
            str(uuid4()),
            adjusted_basis,
            coarse.covariance_diag_bq2,
            coarse.covariance_bq2,
            tuple(point_hypotheses),
            predicted,
            objective,
            coarse.converged and all(result.converged for result in refined),
            diagnostics,
        )
