"""Post-action residual construction, diagnosis, and belief updates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, minimize_scalar

from radcounter.core.estimation.basis import CandidateBasis
from radcounter.core.estimation.estimators import GridPoissonSparseEstimator
from radcounter.core.estimation.poisson import PoissonInverseProblem
from radcounter.core.models.state import BeliefState

FloatArray = NDArray[np.float64]


def _poisson_nll(observed_counts: FloatArray, expected_counts: FloatArray) -> float:
    expected = np.maximum(np.asarray(expected_counts, dtype=np.float64), 1e-12)
    observed = np.asarray(observed_counts, dtype=np.float64)
    return float(np.sum(expected - observed * np.log(expected)))


@dataclass(frozen=True)
class VerificationResidual:
    """Raw and normalized post-action innovation."""

    observed_counts: FloatArray
    predicted_counts: FloatArray
    raw_residual_counts: FloatArray
    normalized_residual: FloatArray
    model_variance: FloatArray

    @classmethod
    def create(
        cls,
        observed_counts: FloatArray,
        predicted_counts: FloatArray,
        model_variance: FloatArray | None = None,
    ) -> VerificationResidual:
        """Construct `observed - predicted` and variance-normalized residual."""

        observed = np.asarray(observed_counts, dtype=np.float64)
        predicted = np.asarray(predicted_counts, dtype=np.float64)
        variance = (
            np.zeros_like(predicted)
            if model_variance is None
            else np.asarray(model_variance, dtype=np.float64)
        )
        if (
            observed.ndim != 1
            or predicted.shape != observed.shape
            or variance.shape != observed.shape
        ):
            raise ValueError("verification arrays must be equal one-dimensional vectors")
        if np.any(observed < 0) or np.any(predicted < 0) or np.any(variance < 0):
            raise ValueError("verification counts/variance must be nonnegative")
        raw = observed - predicted
        normalized = raw / np.sqrt(np.maximum(predicted + variance, 1.0))
        return cls(observed, predicted, raw, normalized, variance)


@dataclass(frozen=True)
class ResidualContext:
    """Public prediction models available to residual hypotheses."""

    residual: VerificationResidual
    decon_base_counts: FloatArray | None = None
    decon_target_pre_action_counts: FloatArray | None = None
    nominal_decon_retention: float | None = None
    shield_candidate_predictions: Mapping[str, FloatArray] = field(default_factory=dict)
    shield_candidate_parameters: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    hidden_source_response_counts_per_bq: FloatArray | None = None
    hidden_source_basis: CandidateBasis | None = None
    localization_jacobian_counts_per_m: FloatArray | None = None
    localization_parameter_names: tuple[str, ...] = ()
    localization_bound_m: float = 1.0

    def __post_init__(self) -> None:
        measurement_count = len(self.residual.observed_counts)
        vector_fields = (self.decon_base_counts, self.decon_target_pre_action_counts)
        for vector in vector_fields:
            if vector is not None and np.asarray(vector).shape != (measurement_count,):
                raise ValueError("decontamination prediction component has wrong shape")
        for prediction in self.shield_candidate_predictions.values():
            if np.asarray(prediction).shape != (measurement_count,):
                raise ValueError("shield candidate prediction has wrong shape")
        if self.hidden_source_response_counts_per_bq is not None:
            response = np.asarray(self.hidden_source_response_counts_per_bq)
            if response.ndim != 2 or response.shape[0] != measurement_count:
                raise ValueError("hidden-source response has wrong shape")
            if (
                self.hidden_source_basis is None
                or response.shape[1] != self.hidden_source_basis.size
            ):
                raise ValueError("hidden-source response and basis are inconsistent")
        if self.localization_jacobian_counts_per_m is not None:
            jacobian = np.asarray(self.localization_jacobian_counts_per_m)
            if jacobian.ndim != 2 or jacobian.shape[0] != measurement_count:
                raise ValueError("localization Jacobian has wrong shape")
            if jacobian.shape[1] != len(self.localization_parameter_names):
                raise ValueError("localization parameter names do not match Jacobian")
        if self.localization_bound_m <= 0:
            raise ValueError("localization bound must be positive")


@dataclass(frozen=True)
class HypothesisFit:
    """One fitted residual explanation."""

    hypothesis_id: str
    predicted_counts: FloatArray
    parameters: Mapping[str, Any]
    parameter_count: int
    negative_log_likelihood: float
    bic: float
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


class ResidualHypothesisUnavailable(RuntimeError):
    """Raised only when a context lacks data required by one hypothesis."""


class ResidualHypothesis(Protocol):
    hypothesis_id: str

    def fit(self, context: ResidualContext) -> HypothesisFit: ...


def _fit_record(
    hypothesis_id: str,
    context: ResidualContext,
    predicted_counts: FloatArray,
    parameters: Mapping[str, Any],
    parameter_count: int,
    diagnostics: Mapping[str, Any] | None = None,
) -> HypothesisFit:
    predicted = np.maximum(np.asarray(predicted_counts, dtype=np.float64), 1e-12)
    nll = _poisson_nll(context.residual.observed_counts, predicted)
    sample_count = len(predicted)
    bic = 2.0 * nll + parameter_count * np.log(max(sample_count, 1))
    return HypothesisFit(
        hypothesis_id,
        predicted,
        dict(parameters),
        parameter_count,
        nll,
        float(bic),
        {} if diagnostics is None else dict(diagnostics),
    )


class DeconResidualHypothesis:
    """Fit residual activity retention in the treated region."""

    hypothesis_id = "decon_residual"

    def fit(self, context: ResidualContext) -> HypothesisFit:
        if context.decon_base_counts is None or context.decon_target_pre_action_counts is None:
            raise ResidualHypothesisUnavailable("decontamination components are absent")
        base = np.asarray(context.decon_base_counts, dtype=np.float64)
        target = np.asarray(context.decon_target_pre_action_counts, dtype=np.float64)

        def objective(retention: float) -> float:
            return _poisson_nll(context.residual.observed_counts, base + retention * target)

        result = minimize_scalar(objective, bounds=(0.0, 2.0), method="bounded")
        retention = float(result.x)
        nominal = context.nominal_decon_retention
        return _fit_record(
            self.hypothesis_id,
            context,
            base + retention * target,
            {
                "retention_factor": retention,
                "nominal_retention_factor": nominal,
            },
            1,
            {"converged": bool(result.success)},
        )


class ShieldPoseErrorHypothesis:
    """Select the highest-likelihood prediction from local shield-pose candidates."""

    hypothesis_id = "shield_pose_error"

    def fit(self, context: ResidualContext) -> HypothesisFit:
        if not context.shield_candidate_predictions:
            raise ResidualHypothesisUnavailable("shield-pose candidates are absent")
        best_id, best_prediction = min(
            context.shield_candidate_predictions.items(),
            key=lambda item: _poisson_nll(context.residual.observed_counts, item[1]),
        )
        parameters: dict[str, Any] = {"candidate_id": best_id}
        parameters.update(context.shield_candidate_parameters.get(best_id, {}))
        return _fit_record(
            self.hypothesis_id,
            context,
            best_prediction,
            parameters,
            max(len(parameters) - 1, 1),
        )


class HiddenSourceHypothesis:
    """Fit positive residual on an unused candidate basis."""

    hypothesis_id = "hidden_source"

    def __init__(self, *, lambda_l1: float = 0.01) -> None:
        self._estimator = GridPoissonSparseEstimator(lambda_l1=lambda_l1)

    def fit(self, context: ResidualContext) -> HypothesisFit:
        response = context.hidden_source_response_counts_per_bq
        basis = context.hidden_source_basis
        if response is None or basis is None:
            raise ResidualHypothesisUnavailable("hidden-source basis/response are absent")
        problem = PoissonInverseProblem(
            context.residual.observed_counts,
            response,
            context.residual.predicted_counts,
        )
        estimate = self._estimator.fit(problem, basis)
        active = estimate.basis_activity_bq > 1e-6
        return _fit_record(
            self.hypothesis_id,
            context,
            estimate.predicted_measurements,
            {
                "basis_ids": list(basis.basis_ids),
                "source_strength_bq": estimate.basis_activity_bq.tolist(),
                "covariance_diag_bq2": estimate.covariance_diag_bq2.tolist(),
            },
            max(int(active.sum()), 1),
            {"estimator_converged": estimate.converged},
        )


class GlobalGainBackgroundHypothesis:
    """Separate detector-wide gain/background mismatch from source changes."""

    hypothesis_id = "global_gain_background"

    def fit(self, context: ResidualContext) -> HypothesisFit:
        nominal = context.residual.predicted_counts
        observed = context.residual.observed_counts

        def objective_gradient(parameters: FloatArray) -> tuple[float, FloatArray]:
            gain, offset = parameters
            expected = np.maximum(gain * nominal + offset, 1e-12)
            residual_factor = 1.0 - observed / expected
            gradient = np.array([float(nominal @ residual_factor), float(residual_factor.sum())])
            return _poisson_nll(observed, expected), gradient

        result = minimize(
            objective_gradient,
            np.array([1.0, 0.0]),
            method="L-BFGS-B",
            jac=True,
            bounds=((0.0, None), (0.0, None)),
        )
        gain, offset = np.maximum(result.x, 0.0)
        return _fit_record(
            self.hypothesis_id,
            context,
            gain * nominal + offset,
            {"gain": float(gain), "background_offset_counts": float(offset)},
            2,
            {"converged": bool(result.success)},
        )


class SourceLocalizationErrorHypothesis:
    """Fit a bounded first-order source-position correction."""

    hypothesis_id = "source_localization_error"

    def fit(self, context: ResidualContext) -> HypothesisFit:
        jacobian = context.localization_jacobian_counts_per_m
        if jacobian is None:
            raise ResidualHypothesisUnavailable("localization Jacobian is absent")
        matrix = np.asarray(jacobian, dtype=np.float64)
        nominal = context.residual.predicted_counts
        observed = context.residual.observed_counts

        def objective_gradient(delta_m: FloatArray) -> tuple[float, FloatArray]:
            expected = np.maximum(nominal + matrix @ delta_m, 1e-12)
            factor = 1.0 - observed / expected
            return _poisson_nll(observed, expected), matrix.T @ factor

        parameter_count = matrix.shape[1]
        result = minimize(
            objective_gradient,
            np.zeros(parameter_count),
            method="L-BFGS-B",
            jac=True,
            bounds=[(-context.localization_bound_m, context.localization_bound_m)]
            * parameter_count,
        )
        parameters = {
            name: float(value)
            for name, value in zip(context.localization_parameter_names, result.x, strict=True)
        }
        return _fit_record(
            self.hypothesis_id,
            context,
            nominal + matrix @ result.x,
            parameters,
            parameter_count,
            {"converged": bool(result.success)},
        )


@dataclass(frozen=True)
class ResidualDiagnosis:
    """Ranked hypothesis selection and normalized evidence weights."""

    selected: HypothesisFit
    ranked_fits: tuple[HypothesisFit, ...]
    confidence: float
    unavailable_hypotheses: Mapping[str, str]


class ResidualDiagnosisEngine:
    """Fit every applicable hypothesis and select minimum BIC."""

    def __init__(self, hypotheses: Sequence[ResidualHypothesis] | None = None) -> None:
        self._hypotheses = tuple(
            hypotheses
            if hypotheses is not None
            else (
                DeconResidualHypothesis(),
                ShieldPoseErrorHypothesis(),
                HiddenSourceHypothesis(),
                GlobalGainBackgroundHypothesis(),
                SourceLocalizationErrorHypothesis(),
            )
        )

    def diagnose(self, context: ResidualContext) -> ResidualDiagnosis:
        """Return a BIC-ranked diagnosis; missing model inputs are explicit."""

        fits: list[HypothesisFit] = []
        unavailable: dict[str, str] = {}
        for hypothesis in self._hypotheses:
            try:
                fits.append(hypothesis.fit(context))
            except ResidualHypothesisUnavailable as exc:
                unavailable[hypothesis.hypothesis_id] = str(exc)
        if not fits:
            raise RuntimeError("no residual hypothesis had sufficient context")
        ranked = tuple(sorted(fits, key=lambda fit: fit.bic))
        bic = np.asarray([fit.bic for fit in ranked], dtype=np.float64)
        weights = np.exp(-0.5 * (bic - bic.min()))
        confidence = float(weights[0] / weights.sum())
        return ResidualDiagnosis(ranked[0], ranked, confidence, unavailable)


class BeliefUpdater:
    """Apply selected public diagnosis parameters to an immutable belief."""

    def update(self, belief: BeliefState, diagnosis: ResidualDiagnosis) -> BeliefState:
        """Return a new belief without reading simulator truth."""

        selected = diagnosis.selected
        parameters = dict(belief.action_effect_parameters)
        strengths = belief.source_strength_bq.copy()
        covariance = belief.covariance.copy()
        basis_ids = belief.basis_ids
        if selected.hypothesis_id == "hidden_source":
            candidate_ids = tuple(str(item) for item in selected.parameters["basis_ids"])
            candidate_strengths = np.asarray(
                selected.parameters["source_strength_bq"], dtype=np.float64
            )
            candidate_variances = np.asarray(
                selected.parameters["covariance_diag_bq2"], dtype=np.float64
            )
            append_mask = np.asarray(
                [candidate_id not in basis_ids for candidate_id in candidate_ids], dtype=np.bool_
            )
            appended_ids = tuple(
                candidate_id
                for candidate_id, append in zip(candidate_ids, append_mask, strict=True)
                if append
            )
            if appended_ids:
                old_size = len(basis_ids)
                appended_strengths = candidate_strengths[append_mask]
                appended_variances = candidate_variances[append_mask]
                new_size = old_size + len(appended_ids)
                expanded_covariance = np.zeros((new_size, new_size), dtype=np.float64)
                expanded_covariance[:old_size, :old_size] = covariance
                expanded_covariance[old_size:, old_size:] = np.diag(appended_variances)
                basis_ids += appended_ids
                strengths = np.concatenate((strengths, appended_strengths))
                covariance = expanded_covariance
        else:
            for name, value in selected.parameters.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    parameters[f"{selected.hypothesis_id}.{name}"] = float(value)
        parameters["residual.selected_bic"] = float(selected.bic)
        parameters["residual.confidence"] = diagnosis.confidence
        return BeliefState(
            basis_ids,
            strengths,
            covariance,
            belief.revision.copy(),
            dict(belief.remaining_resources),
            parameters,
        )


class NominalActionPreviewer:
    """Apply nominal action effects to a belief clone before execution."""

    @staticmethod
    def preview_decontamination(
        belief: BeliefState,
        affected_basis_indexes: NDArray[np.int64],
        nominal_removal_fraction: FloatArray,
    ) -> BeliefState:
        """Scale only selected belief strengths; never inspect actual removal."""

        indexes = np.asarray(affected_basis_indexes, dtype=np.int64)
        removal = np.asarray(nominal_removal_fraction, dtype=np.float64)
        if indexes.ndim != 1 or removal.shape != indexes.shape:
            raise ValueError("preview indexes/removal arrays must match")
        if np.any(indexes < 0) or np.any(indexes >= len(belief.basis_ids)):
            raise ValueError("preview basis index out of range")
        if np.any(removal < 0) or np.any(removal > 1):
            raise ValueError("nominal removal fractions must be in [0,1]")
        strength = belief.source_strength_bq.copy()
        strength[indexes] *= 1.0 - removal
        parameters = dict(belief.action_effect_parameters)
        parameters["preview.decon_mean_removal_fraction"] = float(removal.mean())
        return BeliefState(
            belief.basis_ids,
            strength,
            belief.covariance.copy(),
            belief.revision.copy(),
            dict(belief.remaining_resources),
            parameters,
        )
