import ast
from pathlib import Path

import numpy as np

from radcounter.core.estimation import (
    BeliefUpdater,
    CandidateBasis,
    DeconResidualHypothesis,
    GlobalGainBackgroundHypothesis,
    HiddenSourceHypothesis,
    NominalActionPreviewer,
    ResidualContext,
    ResidualDiagnosisEngine,
    ShieldPoseErrorHypothesis,
    SourceLocalizationErrorHypothesis,
    VerificationResidual,
)
from radcounter.core.models import BeliefState, RevisionState


def _belief() -> BeliefState:
    return BeliefState(
        ("known-0",),
        np.array([10.0]),
        np.array([[1.0]]),
        RevisionState(),
        {"robot_s": 100.0},
    )


def test_verification_residual_normalization() -> None:
    residual = VerificationResidual.create(
        np.array([12.0, 7.0]), np.array([10.0, 5.0]), np.array([2.0, 3.0])
    )
    assert np.array_equal(residual.raw_residual_counts, [2.0, 2.0])
    assert np.allclose(residual.normalized_residual, [2.0 / np.sqrt(12.0), 2.0 / np.sqrt(8.0)])


def test_decon_residual_is_selected_and_updates_belief_parameter() -> None:
    base = np.array([5.0, 7.0, 9.0, 4.0])
    target = np.array([20.0, 3.0, 8.0, 15.0])
    observed = base + 0.4 * target
    nominal = base + 0.1 * target
    context = ResidualContext(
        VerificationResidual.create(observed, nominal),
        decon_base_counts=base,
        decon_target_pre_action_counts=target,
        nominal_decon_retention=0.1,
    )
    diagnosis = ResidualDiagnosisEngine(
        [DeconResidualHypothesis(), GlobalGainBackgroundHypothesis()]
    ).diagnose(context)
    assert diagnosis.selected.hypothesis_id == "decon_residual"
    assert np.isclose(diagnosis.selected.parameters["retention_factor"], 0.4, atol=1e-4)
    updated = BeliefUpdater().update(_belief(), diagnosis)
    assert np.isclose(
        updated.action_effect_parameters["decon_residual.retention_factor"], 0.4, atol=1e-4
    )


def test_shield_pose_candidate_recovers_translation() -> None:
    nominal = np.array([10.0, 30.0, 8.0, 18.0])
    shifted = np.array([12.0, 20.0, 15.0, 10.0])
    context = ResidualContext(
        VerificationResidual.create(shifted, nominal),
        shield_candidate_predictions={"nominal": nominal, "dx+5cm": shifted},
        shield_candidate_parameters={"dx+5cm": {"translation_x_m": 0.05}},
    )
    diagnosis = ResidualDiagnosisEngine(
        [ShieldPoseErrorHypothesis(), GlobalGainBackgroundHypothesis()]
    ).diagnose(context)
    assert diagnosis.selected.hypothesis_id == "shield_pose_error"
    assert diagnosis.selected.parameters["candidate_id"] == "dx+5cm"


def test_hidden_source_is_detected_and_appended_to_belief() -> None:
    nominal = np.array([3.0, 3.0, 3.0, 3.0])
    response = np.array([[1.0, 0.0], [0.5, 0.1], [0.1, 0.5], [0.0, 1.0]])
    hidden_strength = np.array([8.0, 0.0])
    observed = nominal + response @ hidden_strength
    hidden_basis = CandidateBasis.surface(
        np.array([[1.0, 0.0, 0.0], [4.0, 0.0, 0.0]]), np.empty((0, 2), dtype=int)
    )
    context = ResidualContext(
        VerificationResidual.create(observed, nominal),
        hidden_source_response_counts_per_bq=response,
        hidden_source_basis=hidden_basis,
    )
    diagnosis = ResidualDiagnosisEngine(
        [HiddenSourceHypothesis(lambda_l1=0.0), GlobalGainBackgroundHypothesis()]
    ).diagnose(context)
    assert diagnosis.selected.hypothesis_id == "hidden_source"
    updated = BeliefUpdater().update(_belief(), diagnosis)
    assert "surface-0" in updated.basis_ids
    assert len(updated.basis_ids) == 3


def test_global_gain_bias_is_not_forced_into_localization() -> None:
    nominal = np.array([5.0, 10.0, 20.0, 8.0, 14.0])
    observed = 1.3 * nominal + 2.0
    context = ResidualContext(
        VerificationResidual.create(observed, nominal),
        localization_jacobian_counts_per_m=np.array([[1.0], [-1.0], [0.5], [-0.5], [0.2]]),
        localization_parameter_names=("source_0_dx_m",),
    )
    diagnosis = ResidualDiagnosisEngine(
        [GlobalGainBackgroundHypothesis(), SourceLocalizationErrorHypothesis()]
    ).diagnose(context)
    assert diagnosis.selected.hypothesis_id == "global_gain_background"
    assert np.isclose(diagnosis.selected.parameters["gain"], 1.3, atol=1e-4)
    assert np.isclose(diagnosis.selected.parameters["background_offset_counts"], 2.0, atol=1e-4)


def test_localization_error_recovers_bounded_shift() -> None:
    nominal = np.array([10.0, 15.0, 20.0, 12.0])
    jacobian = np.array([[8.0], [-3.0], [2.0], [-5.0]])
    observed = nominal + jacobian[:, 0] * 0.2
    context = ResidualContext(
        VerificationResidual.create(observed, nominal),
        localization_jacobian_counts_per_m=jacobian,
        localization_parameter_names=("source_0_dx_m",),
        localization_bound_m=0.5,
    )
    diagnosis = ResidualDiagnosisEngine(
        [SourceLocalizationErrorHypothesis(), GlobalGainBackgroundHypothesis()]
    ).diagnose(context)
    assert diagnosis.selected.hypothesis_id == "source_localization_error"
    assert np.isclose(diagnosis.selected.parameters["source_0_dx_m"], 0.2, atol=1e-4)


def test_mixed_failure_reports_nonunit_confidence() -> None:
    nominal = np.array([10.0, 20.0, 15.0, 8.0])
    observed = 1.1 * nominal + np.array([3.0, 0.0, 2.0, 0.0])
    response = np.array([[1.0], [0.1], [0.8], [0.0]])
    basis = CandidateBasis.surface(np.array([[1.0, 0.0, 0.0]]), np.empty((0, 2), dtype=int))
    context = ResidualContext(
        VerificationResidual.create(observed, nominal),
        hidden_source_response_counts_per_bq=response,
        hidden_source_basis=basis,
    )
    diagnosis = ResidualDiagnosisEngine(
        [HiddenSourceHypothesis(lambda_l1=0.0), GlobalGainBackgroundHypothesis()]
    ).diagnose(context)
    assert 0.5 <= diagnosis.confidence < 1.0
    assert len(diagnosis.ranked_fits) == 2


def test_nominal_preview_updates_clone_only() -> None:
    belief = _belief()
    preview = NominalActionPreviewer.preview_decontamination(
        belief, np.array([0]), np.array([0.25])
    )
    assert np.isclose(preview.source_strength_bq[0], 7.5)
    assert np.isclose(belief.source_strength_bq[0], 10.0)


def test_residual_module_does_not_reference_truth_state() -> None:
    path = Path(__file__).resolve().parents[2] / "radcounter/core/estimation/residual.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    identifiers = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "TruthState" not in identifiers
