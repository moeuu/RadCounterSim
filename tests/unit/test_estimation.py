import ast
from pathlib import Path

import numpy as np
import pytest

from radcounter.core.estimation import (
    CandidateBasis,
    ContinuousPointMLERefiner,
    DeadTimePoissonEstimator,
    DeadTimePoissonInverseProblem,
    GridPoissonSparseEstimator,
    InverseSquarePointResponseModel,
    PFPlusMLEEstimator,
    PoissonInverseProblem,
    SurfacePoissonTVEstimator,
    fisher_covariance,
    poisson_nll_and_gradient,
)


def _two_candidate_basis() -> CandidateBasis:
    return CandidateBasis.surface(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([[0, 1]]),
    )


def test_poisson_gradient_matches_finite_difference() -> None:
    problem = PoissonInverseProblem(
        np.array([8.0, 5.0, 2.0]),
        np.array([[1.0, 0.2], [0.3, 1.1], [0.5, 0.4]]),
        np.array([0.5, 0.5, 0.5]),
    )
    point = np.array([3.0, 2.0])
    objective, gradient = poisson_nll_and_gradient(point, problem, lambda_l1=0.1)
    epsilon = 1e-6
    numerical = np.zeros_like(point)
    for index in range(len(point)):
        offset = np.zeros_like(point)
        offset[index] = epsilon
        plus = poisson_nll_and_gradient(point + offset, problem, lambda_l1=0.1)[0]
        minus = poisson_nll_and_gradient(point - offset, problem, lambda_l1=0.1)[0]
        numerical[index] = (plus - minus) / (2.0 * epsilon)
    assert np.isfinite(objective)
    assert np.allclose(gradient, numerical, rtol=1e-5, atol=1e-6)


def test_noiseless_two_source_recovery() -> None:
    response = np.array([[1.0, 0.1], [0.2, 1.0], [0.8, 0.3], [0.3, 0.9]], dtype=np.float64)
    truth = np.array([12.0, 5.0])
    background = np.full(4, 0.25)
    problem = PoissonInverseProblem(response @ truth + background, response, background)
    estimate = GridPoissonSparseEstimator().fit(problem, _two_candidate_basis())
    assert estimate.converged
    assert np.allclose(estimate.basis_activity_bq, truth, rtol=1e-5, atol=1e-5)
    assert len(estimate.point_hypotheses) == 1


def test_l1_estimator_suppresses_unused_candidate() -> None:
    response = np.array([[1.0, 0.0], [0.5, 0.1], [0.2, 1.0]])
    truth = np.array([20.0, 0.0])
    background = np.full(3, 1.0)
    problem = PoissonInverseProblem(response @ truth + background, response, background)
    estimate = GridPoissonSparseEstimator(lambda_l1=0.05).fit(problem, _two_candidate_basis())
    assert estimate.basis_activity_bq[0] > 10.0
    assert estimate.basis_activity_bq[1] < 1e-6


def test_surface_tv_estimator_is_nonnegative_and_reports_regularization() -> None:
    basis = CandidateBasis.surface(
        np.column_stack((np.arange(4, dtype=float), np.zeros((4, 2)))),
        np.array([[0, 1], [1, 2], [2, 3]]),
    )
    response = np.eye(4)
    background = np.full(4, 0.1)
    observed = np.array([0.1, 10.1, 10.1, 0.1])
    estimate = SurfacePoissonTVEstimator(lambda_l1=0.01, lambda_tv=0.1).fit(
        PoissonInverseProblem(observed, response, background), basis
    )
    assert np.all(estimate.basis_activity_bq >= 0)
    assert estimate.diagnostics["tv_approximation"] == "smooth_graph_total_variation"
    assert estimate.basis_activity_bq[1] > estimate.basis_activity_bq[0]


def test_fisher_covariance_is_symmetric_positive_semidefinite() -> None:
    problem = PoissonInverseProblem(
        np.array([5.0, 7.0]),
        np.array([[1.0, 0.2], [0.1, 1.0]]),
        np.ones(2),
    )
    covariance, active = fisher_covariance(np.array([4.0, 6.0]), problem)
    assert np.all(active)
    assert np.allclose(covariance, covariance.T)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-10


def test_bootstrap_is_reproducible() -> None:
    basis = _two_candidate_basis()
    response = np.eye(2)
    problem = PoissonInverseProblem(np.array([11.0, 6.0]), response, np.ones(2))
    estimator = GridPoissonSparseEstimator()
    estimate = estimator.fit(problem, basis)
    first = estimator.bootstrap(
        estimate, problem, basis, replicates=3, rng=np.random.default_rng(20)
    )
    second = estimator.bootstrap(
        estimate, problem, basis, replicates=3, rng=np.random.default_rng(20)
    )
    assert np.array_equal(first, second)


def test_estimation_modules_do_not_reference_truth_state() -> None:
    package = Path(__file__).resolve().parents[2] / "radcounter/core/estimation"
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert "TruthState" not in identifiers
        assert all("TruthState" not in name for name in imported)


def _continuous_problem() -> tuple[
    PoissonInverseProblem,
    CandidateBasis,
    InverseSquarePointResponseModel,
    np.ndarray,
]:
    detector_positions = np.asarray(
        (
            (2.0, 0.0, 0.0),
            (-2.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.0, -2.0, 0.0),
            (0.0, 0.0, 2.0),
            (0.0, 0.0, -2.0),
            (2.0, 2.0, 0.0),
            (-2.0, -2.0, 0.0),
        )
    )
    model = InverseSquarePointResponseModel(
        detector_positions,
        np.full(len(detector_positions), 10.0),
    )
    true_position = np.asarray((0.23, -0.17, 0.12))
    true_strength = 1000.0
    background = np.full(len(detector_positions), 2.0)
    basis = CandidateBasis.regular_grid((-0.5, 0.5, -0.5, 0.5, -0.5, 0.5), spacing_m=0.5)
    response = np.column_stack(
        [model.response_counts_per_bq(position) for position in basis.positions_world_m]
    )
    observed = background + model.response_counts_per_bq(true_position) * true_strength
    return PoissonInverseProblem(observed, response, background), basis, model, true_position


def test_continuous_point_mle_refines_beyond_grid_resolution() -> None:
    problem, basis, model, true_position = _continuous_problem()
    coarse = GridPoissonSparseEstimator(lambda_l1=0.001).fit(problem, basis)
    refined = ContinuousPointMLERefiner(
        model,
        ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)),
    ).refine(problem, basis, coarse)
    assert len(refined) == 1
    coarse_error = np.linalg.norm(coarse.point_hypotheses[0].position_world_m - true_position)
    refined_error = np.linalg.norm(refined[0].position_world_m - true_position)
    assert refined[0].converged
    assert refined_error < coarse_error * 0.01
    assert np.isclose(refined[0].source_strength_bq, 1000.0, rtol=1e-5)
    assert refined[0].parameter_covariance.shape == (4, 4)


def test_particle_assisted_mle_is_reproducible_and_continuous() -> None:
    problem, basis, model, true_position = _continuous_problem()
    estimator = PFPlusMLEEstimator(
        model,
        ((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)),
        particle_count=128,
        particle_rounds=3,
        random_seed=4,
        grid_estimator=GridPoissonSparseEstimator(lambda_l1=0.001),
    )
    first = estimator.fit(problem, basis)
    second = estimator.fit(problem, basis)
    assert first.converged and second.converged
    np.testing.assert_allclose(
        first.point_hypotheses[0].position_world_m,
        second.point_hypotheses[0].position_world_m,
    )
    assert np.linalg.norm(first.point_hypotheses[0].position_world_m - true_position) < 1e-4
    assert first.diagnostics["solver"] == ("particle_exploration_plus_continuous_L-BFGS-B")


def test_dead_time_poisson_estimator_recovers_nonlinear_activity() -> None:
    basis = _two_candidate_basis()
    response = np.asarray(((2.0e-5, 0.5e-5), (0.4e-5, 1.8e-5), (1.2e-5, 0.8e-5)))
    truth = np.asarray((1.2e6, 0.7e6))
    problem_template = DeadTimePoissonInverseProblem(
        np.zeros(3),
        response,
        np.asarray((2.0, 3.0, 2.5)),
        np.asarray((10.0, 12.0, 8.0)),
        np.asarray((2.0e-3, 1.0e-3, 1.5e-3)),
    )
    problem = DeadTimePoissonInverseProblem(
        problem_template.expected_counts(truth),
        response,
        problem_template.background_rate_cps,
        problem_template.duration_s,
        problem_template.dead_time_s,
    )
    estimate = DeadTimePoissonEstimator().fit(problem, basis)
    assert estimate.converged
    np.testing.assert_allclose(estimate.basis_activity_bq, truth, rtol=2.0e-5)
    np.testing.assert_allclose(estimate.predicted_measurements, problem.observed_counts)
    assert np.min(np.linalg.eigvalsh(estimate.covariance_bq2)) >= -1.0e-8
    assert estimate.diagnostics["solver"] == ("nonparalyzable_dead_time_poisson_L-BFGS-B")


def test_dead_time_problem_rejects_inconsistent_rows() -> None:
    with pytest.raises(ValueError, match="rows"):
        DeadTimePoissonInverseProblem(
            np.ones(2),
            np.ones((3, 1)),
            np.ones(2),
            np.ones(2),
            np.zeros(2),
        )
