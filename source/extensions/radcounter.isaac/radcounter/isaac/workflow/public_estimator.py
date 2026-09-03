"""Public-measurement source estimation against the synchronized Isaac scene."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from radcounter.core.estimation import (
    BasisKind,
    CandidateBasis,
    DeadTimePoissonEstimator,
    DeadTimePoissonInverseProblem,
)
from radcounter.core.models.state import BeliefState, RevisionState


class IsaacPublicPoissonEstimator:
    """Estimate public source candidates without reading activity truth fields."""

    def __init__(
        self,
        simulation: object,
        *,
        source_paths: Sequence[str] | None = None,
        maximum_iterations: int = 1000,
    ) -> None:
        self.simulation = simulation
        self.source_paths = None if source_paths is None else tuple(source_paths)
        self.estimator = DeadTimePoissonEstimator(maximum_iterations=maximum_iterations)
        self.last_audit: dict[str, object] | None = None

    def __call__(
        self,
        measurements: tuple[object, ...],
        previous: BeliefState | None,
    ) -> BeliefState:
        if not measurements:
            raise ValueError("public source estimation requires measurements")
        response = self.simulation.public_source_response(source_paths=self.source_paths)
        by_path = {str(item.detector_path): item for item in measurements}
        if set(by_path) != set(response.detector_paths):
            raise ValueError(
                "public measurements must exactly match the configured response detectors"
            )
        ordered = [by_path[path] for path in response.detector_paths]
        durations = np.asarray([float(item.duration_s) for item in ordered])
        observed = np.asarray([int(item.counts) for item in ordered], dtype=np.float64)
        problem = DeadTimePoissonInverseProblem(
            observed_counts=observed,
            source_rate_cps_per_bq=response.source_rate_cps_per_bq,
            background_rate_cps=response.background_rate_cps,
            duration_s=durations,
            dead_time_s=response.dead_time_s,
        )
        basis = CandidateBasis(
            basis_ids=response.source_paths,
            positions_world_m=response.source_positions_world_m,
            adjacency_edges=np.empty((0, 2), dtype=np.int64),
            kind=BasisKind.GRID_3D,
        )
        initial = None
        if previous is not None and previous.basis_ids == response.source_paths:
            initial = previous.source_strength_bq
        estimate = self.estimator.fit(problem, basis, initial)
        if not estimate.converged:
            raise RuntimeError(
                "public source estimator did not converge: "
                f"{estimate.diagnostics.get('message', 'unknown solver status')}"
            )
        self.last_audit = {
            "estimate_id": estimate.estimate_id,
            "solver": estimate.diagnostics["solver"],
            "objective_value": estimate.objective_value,
            "observed_counts": observed.tolist(),
            "predicted_counts": estimate.predicted_measurements.tolist(),
            "source_paths": list(response.source_paths),
            "detector_paths": list(response.detector_paths),
            "template_kinds": list(response.template_kinds),
            "source_rate_cps_per_bq": response.source_rate_cps_per_bq.tolist(),
            "background_rate_cps": response.background_rate_cps.tolist(),
            "dead_time_s": response.dead_time_s.tolist(),
            "activity_covariance_trace_bq2": float(np.trace(estimate.covariance_bq2)),
        }
        return BeliefState(
            basis_ids=response.source_paths,
            source_strength_bq=estimate.basis_activity_bq,
            covariance=estimate.covariance_bq2,
            revision=(RevisionState() if previous is None else previous.revision.copy()),
            remaining_resources=({} if previous is None else dict(previous.remaining_resources)),
            action_effect_parameters=(
                {} if previous is None else dict(previous.action_effect_parameters)
            ),
        )
