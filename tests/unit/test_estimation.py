import ast
from pathlib import Path

import numpy as np

from radcounter.core.estimation import (
    CandidateBasis,
    GridPoissonSparseEstimator,
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
