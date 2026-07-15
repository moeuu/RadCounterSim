"""Deterministic experiment cases covering the countermeasure workflow."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from radcounter.core.experiments.artifacts import CaseResult


def _result(
    *,
    metrics: dict[str, Any],
    measurements: list[dict[str, Any]],
    estimates: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    resources: list[dict[str, Any]],
    maps: dict[str, dict[str, np.ndarray]],
) -> CaseResult:
    return CaseResult(
        metrics=metrics,
        tables={
            "measurements": measurements,
            "estimates": estimates,
            "actions": actions,
            "resources": resources,
        },
        maps=maps,
        events=(),
    )


def _inverse_square_response(
    detector_positions_m: np.ndarray,
    source_position_m: np.ndarray,
    minimum_distance_m: float = 0.25,
) -> np.ndarray:
    squared_distance = np.sum(
        (detector_positions_m - source_position_m[None, :]) ** 2,
        axis=1,
    )
    return 1.0 / np.maximum(squared_distance, minimum_distance_m**2)


@dataclass(frozen=True)
class DecontaminationPrimitiveCase:
    """Recover a decontamination efficiency from pre/post Poisson counts."""

    case_id: str = "decontamination_primitive"

    def run(self, *, seed: int, planner_id: str) -> CaseResult:
        rng = np.random.default_rng(seed)
        initial_strength_bq = 120_000.0
        true_removal_fraction = 0.65
        sensitivity_cps_per_bq = 0.05
        background_cps = 2.0
        dwell_s = 4.0
        post_strength_bq = initial_strength_bq * (1.0 - true_removal_fraction)
        pre_expected = (initial_strength_bq * sensitivity_cps_per_bq + background_cps) * dwell_s
        post_expected = (post_strength_bq * sensitivity_cps_per_bq + background_cps) * dwell_s
        pre_counts = int(rng.poisson(pre_expected))
        post_counts = int(rng.poisson(post_expected))
        background_counts = background_cps * dwell_s
        pre_net = max(float(pre_counts) - background_counts, 1.0)
        post_net = max(float(post_counts) - background_counts, 0.0)
        estimated_remaining_fraction = post_net / pre_net
        estimated_removal_fraction = float(np.clip(1.0 - estimated_remaining_fraction, 0.0, 1.0))
        variance_remaining = (
            float(post_counts) / pre_net**2 + post_net**2 * float(pre_counts) / pre_net**4
        )
        removal_std = float(np.sqrt(max(variance_remaining, 0.0)))
        absolute_error = abs(estimated_removal_fraction - true_removal_fraction)
        return _result(
            metrics={
                "planner_id": planner_id,
                "true_removal_fraction": true_removal_fraction,
                "estimated_removal_fraction": estimated_removal_fraction,
                "removal_fraction_std": removal_std,
                "decontamination_efficiency_absolute_error": absolute_error,
                "passed": absolute_error <= 0.08,
            },
            measurements=[
                {
                    "measurement_id": "pre-decon",
                    "phase": "before",
                    "counts": pre_counts,
                    "expected_counts": pre_expected,
                    "dwell_s": dwell_s,
                },
                {
                    "measurement_id": "post-decon",
                    "phase": "verification",
                    "counts": post_counts,
                    "expected_counts": post_expected,
                    "dwell_s": dwell_s,
                },
            ],
            estimates=[
                {
                    "estimate_id": "decon-efficiency",
                    "removal_fraction": estimated_removal_fraction,
                    "standard_uncertainty": removal_std,
                    "source_strength_uncertainty_bq": removal_std * initial_strength_bq,
                }
            ],
            actions=[
                {
                    "action_id": "decon-zone-a",
                    "action_type": "decontamination",
                    "requested_mode": "wet_wipe",
                    "status": "succeeded",
                    "true_removed_bq": initial_strength_bq - post_strength_bq,
                }
            ],
            resources=[
                {
                    "action_id": "decon-zone-a",
                    "elapsed_s": 90.0,
                    "waste_generated_kg": 1.8,
                    "shield_mass_used_kg": 0.0,
                }
            ],
            maps={
                "decontamination_profile": {
                    "phase": np.array([0, 1], dtype=np.int64),
                    "source_strength_bq": np.array([initial_strength_bq, post_strength_bq]),
                }
            },
        )


@dataclass(frozen=True)
class ShieldingPrimitiveCase:
    """Estimate effective shield attenuation from a verification measurement."""

    case_id: str = "shielding_primitive"

    def run(self, *, seed: int, planner_id: str) -> CaseResult:
        rng = np.random.default_rng(seed)
        unshielded_rate_cps = 5_000.0
        background_cps = 3.0
        attenuation_coefficient_per_cm = 0.82
        commanded_thickness_cm = 3.0
        effective_thickness_cm = 2.85
        predicted_transmission = float(
            np.exp(-attenuation_coefficient_per_cm * commanded_thickness_cm)
        )
        true_transmission = float(np.exp(-attenuation_coefficient_per_cm * effective_thickness_cm))
        dwell_s = 5.0
        pre_counts = int(rng.poisson((unshielded_rate_cps + background_cps) * dwell_s))
        post_counts = int(
            rng.poisson((unshielded_rate_cps * true_transmission + background_cps) * dwell_s)
        )
        pre_net_rate = max(pre_counts / dwell_s - background_cps, 1.0)
        post_net_rate = max(post_counts / dwell_s - background_cps, 0.0)
        estimated_transmission = post_net_rate / pre_net_rate
        transmission_error = abs(estimated_transmission - true_transmission)
        return _result(
            metrics={
                "planner_id": planner_id,
                "predicted_transmission": predicted_transmission,
                "true_transmission": true_transmission,
                "estimated_transmission": estimated_transmission,
                "shield_transmission_absolute_error": transmission_error,
                "passed": transmission_error <= 0.03,
            },
            measurements=[
                {
                    "measurement_id": "shield-pre",
                    "phase": "before",
                    "counts": pre_counts,
                    "dwell_s": dwell_s,
                },
                {
                    "measurement_id": "shield-post",
                    "phase": "verification",
                    "counts": post_counts,
                    "dwell_s": dwell_s,
                },
            ],
            estimates=[
                {
                    "estimate_id": "effective-shield",
                    "transmission": estimated_transmission,
                    "effective_thickness_cm": -np.log(max(estimated_transmission, 1.0e-12))
                    / attenuation_coefficient_per_cm,
                }
            ],
            actions=[
                {
                    "action_id": "place-shield-a",
                    "action_type": "shield_placement",
                    "status": "succeeded",
                    "commanded_thickness_cm": commanded_thickness_cm,
                    "effective_thickness_cm": effective_thickness_cm,
                }
            ],
            resources=[
                {
                    "action_id": "place-shield-a",
                    "elapsed_s": 45.0,
                    "shield_mass_used_kg": 8.0,
                    "waste_generated_kg": 0.0,
                }
            ],
            maps={
                "shield_response": {
                    "thickness_cm": np.linspace(0.0, 4.0, 81),
                    "transmission": np.exp(
                        -attenuation_coefficient_per_cm * np.linspace(0.0, 4.0, 81)
                    ),
                }
            },
        )


@dataclass(frozen=True)
class MovableContaminatedObjectCase:
    """Move an object-borne source and localize it at the destination."""

    case_id: str = "movable_contaminated_object"

    def run(self, *, seed: int, planner_id: str) -> CaseResult:
        rng = np.random.default_rng(seed)
        detector_xy_m = np.array(
            [
                [-2.0, -2.0],
                [-2.0, 0.0],
                [-2.0, 2.0],
                [0.0, -2.0],
                [0.0, 2.0],
                [2.0, -2.0],
                [2.0, 0.0],
                [2.0, 2.0],
            ]
        )
        initial_xy_m = np.array([-1.0, -0.5])
        destination_xy_m = np.array([1.0, 0.5])
        response_gain_cps_m2 = 420.0
        background_cps = 1.5
        dwell_s = 8.0
        response = _inverse_square_response(detector_xy_m, destination_xy_m)
        expected_rate = background_cps + response_gain_cps_m2 * response
        counts = rng.poisson(expected_rate * dwell_s)
        observed_net_rate = counts / dwell_s - background_cps

        axis = np.linspace(-1.5, 1.5, 31)
        grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
        candidates = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        squared_distance = np.sum(
            (detector_xy_m[:, None, :] - candidates[None, :, :]) ** 2,
            axis=2,
        )
        design = 1.0 / np.maximum(squared_distance, 0.25**2)
        fitted_gain = np.sum(design * observed_net_rate[:, None], axis=0) / np.maximum(
            np.sum(design**2, axis=0),
            1.0e-12,
        )
        fitted_gain = np.maximum(fitted_gain, 0.0)
        residual = observed_net_rate[:, None] - design * fitted_gain[None, :]
        objective = np.sum(residual**2, axis=0)
        best_index = int(np.argmin(objective))
        estimated_xy_m = candidates[best_index]
        localization_error_m = float(np.linalg.norm(estimated_xy_m - destination_xy_m))
        measurements = [
            {
                "measurement_id": f"object-post-{index}",
                "detector_x_m": float(position[0]),
                "detector_y_m": float(position[1]),
                "counts": int(value),
                "dwell_s": dwell_s,
            }
            for index, (position, value) in enumerate(zip(detector_xy_m, counts, strict=True))
        ]
        return _result(
            metrics={
                "planner_id": planner_id,
                "localization_error_m": localization_error_m,
                "estimated_x_m": float(estimated_xy_m[0]),
                "estimated_y_m": float(estimated_xy_m[1]),
                "passed": localization_error_m <= 0.2,
            },
            measurements=measurements,
            estimates=[
                {
                    "estimate_id": "moved-object-source",
                    "x_m": float(estimated_xy_m[0]),
                    "y_m": float(estimated_xy_m[1]),
                    "response_gain_cps_m2": float(fitted_gain[best_index]),
                    "source_strength_uncertainty": float(
                        np.sqrt(objective[best_index] / len(detector_xy_m))
                    ),
                }
            ],
            actions=[
                {
                    "action_id": "move-object-a",
                    "action_type": "object_move",
                    "status": "succeeded",
                    "initial_xy_m": initial_xy_m,
                    "destination_xy_m": destination_xy_m,
                }
            ],
            resources=[
                {
                    "action_id": "move-object-a",
                    "elapsed_s": 70.0,
                    "payload_mass_kg": 4.2,
                    "waste_generated_kg": 0.0,
                }
            ],
            maps={
                "localization_objective": {
                    "x_m": candidates[:, 0],
                    "y_m": candidates[:, 1],
                    "sum_squared_residual": objective,
                }
            },
        )


@dataclass(frozen=True)
class HiddenSourceResidualCase:
    """Detect and localize an omitted source from structured residuals."""

    case_id: str = "hidden_source_residual"

    def run(self, *, seed: int, planner_id: str) -> CaseResult:
        rng = np.random.default_rng(seed)
        angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        detector_xy_m = np.column_stack((2.8 * np.cos(angles), 2.8 * np.sin(angles)))
        known_source_xy_m = np.array([-0.8, -0.5])
        hidden_source_xy_m = np.array([1.0, 0.6])
        known_gain_cps_m2 = 300.0
        hidden_gain_cps_m2 = 200.0
        background_cps = 2.0
        dwell_s = 20.0
        known_rate = known_gain_cps_m2 * _inverse_square_response(detector_xy_m, known_source_xy_m)
        hidden_rate = hidden_gain_cps_m2 * _inverse_square_response(
            detector_xy_m, hidden_source_xy_m
        )
        counts = rng.poisson((known_rate + hidden_rate + background_cps) * dwell_s)
        residual_rate = counts / dwell_s - background_cps - known_rate

        axis = np.linspace(-1.5, 1.5, 31)
        grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
        candidates = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        squared_distance = np.sum(
            (detector_xy_m[:, None, :] - candidates[None, :, :]) ** 2,
            axis=2,
        )
        design = 1.0 / np.maximum(squared_distance, 0.25**2)
        fitted_gain = np.sum(design * residual_rate[:, None], axis=0) / np.maximum(
            np.sum(design**2, axis=0),
            1.0e-12,
        )
        fitted_gain = np.maximum(fitted_gain, 0.0)
        candidate_residual = residual_rate[:, None] - design * fitted_gain[None, :]
        candidate_sse = np.sum(candidate_residual**2, axis=0)
        best_index = int(np.argmin(candidate_sse))
        estimated_xy_m = candidates[best_index]
        baseline_sse = float(np.sum(residual_rate**2))
        best_sse = float(candidate_sse[best_index])
        bic_improvement = float(
            len(detector_xy_m) * np.log(max(baseline_sse, 1.0e-12) / max(best_sse, 1.0e-12))
            - np.log(len(detector_xy_m))
        )
        confidence = float(1.0 / (1.0 + np.exp(-0.25 * (bic_improvement - 6.0))))
        localization_error_m = float(np.linalg.norm(estimated_xy_m - hidden_source_xy_m))
        measurements = [
            {
                "measurement_id": f"hidden-residual-{index}",
                "detector_x_m": float(position[0]),
                "detector_y_m": float(position[1]),
                "counts": int(value),
                "nominal_expected_counts": float((known_rate[index] + background_cps) * dwell_s),
                "verification_residual_cps": float(residual_rate[index]),
            }
            for index, (position, value) in enumerate(zip(detector_xy_m, counts, strict=True))
        ]
        return _result(
            metrics={
                "planner_id": planner_id,
                "selected_hypothesis": "hidden_source",
                "bic_improvement": bic_improvement,
                "diagnostic_confidence": confidence,
                "hidden_source_localization_error_m": localization_error_m,
                "passed": bic_improvement >= 10.0 and localization_error_m <= 0.25,
            },
            measurements=measurements,
            estimates=[
                {
                    "estimate_id": "hidden-source",
                    "hypothesis": "hidden_source",
                    "x_m": float(estimated_xy_m[0]),
                    "y_m": float(estimated_xy_m[1]),
                    "response_gain_cps_m2": float(fitted_gain[best_index]),
                    "confidence": confidence,
                    "source_strength_uncertainty": float(np.sqrt(best_sse / len(detector_xy_m))),
                }
            ],
            actions=[],
            resources=[],
            maps={
                "hidden_source_hypothesis": {
                    "x_m": candidates[:, 0],
                    "y_m": candidates[:, 1],
                    "sum_squared_residual": candidate_sse,
                    "fitted_response_gain_cps_m2": fitted_gain,
                }
            },
        )


@dataclass(frozen=True)
class ClosedLoopComparisonCase:
    """Compare fixed open-loop execution with verify-and-replan control."""

    case_id: str = "closed_loop_vs_open_loop"

    def run(self, *, seed: int, planner_id: str) -> CaseResult:
        rng = np.random.default_rng(seed)
        initial_dose_rate_usv_h = 100.0
        first_removal_fraction = float(np.clip(rng.normal(0.38, 0.03), 0.3, 0.46))
        recovery_removal_fraction = 0.48
        shield_transmission = 0.58
        verification_threshold_usv_h = 45.0

        post_first_dose = initial_dose_rate_usv_h * (1.0 - first_removal_fraction)
        open_loop_final = post_first_dose * shield_transmission
        closed_loop_actions = 2
        post_recovery_dose = post_first_dose
        if post_first_dose > verification_threshold_usv_h:
            post_recovery_dose *= 1.0 - recovery_removal_fraction
            closed_loop_actions += 1
        closed_loop_final = post_recovery_dose * shield_transmission
        benefit = open_loop_final - closed_loop_final
        dwell_s = 10.0
        count_scale = 3.0
        verification_counts = int(rng.poisson(post_first_dose * count_scale * dwell_s))
        verification_estimate = verification_counts / (count_scale * dwell_s)
        action_rows = [
            {
                "strategy": "open_loop",
                "sequence": 1,
                "action_type": "decontamination",
                "status": "succeeded",
            },
            {
                "strategy": "open_loop",
                "sequence": 2,
                "action_type": "shield_placement",
                "status": "succeeded",
            },
            {
                "strategy": "closed_loop",
                "sequence": 1,
                "action_type": "decontamination",
                "status": "succeeded",
            },
            {
                "strategy": "closed_loop",
                "sequence": 2,
                "action_type": "verification_measurement",
                "status": "residual_detected",
            },
            {
                "strategy": "closed_loop",
                "sequence": 3,
                "action_type": "adaptive_decontamination",
                "status": "succeeded",
            },
            {
                "strategy": "closed_loop",
                "sequence": 4,
                "action_type": "shield_placement",
                "status": "succeeded",
            },
        ]
        return _result(
            metrics={
                "planner_id": planner_id,
                "initial_dose_rate_usv_h": initial_dose_rate_usv_h,
                "open_loop_final_dose_rate_usv_h": open_loop_final,
                "closed_loop_final_dose_rate_usv_h": closed_loop_final,
                "closed_loop_benefit_usv_h": benefit,
                "open_loop_action_count": 2,
                "closed_loop_action_count": closed_loop_actions,
                "passed": closed_loop_final < open_loop_final,
            },
            measurements=[
                {
                    "measurement_id": "closed-loop-verification",
                    "phase": "verification",
                    "counts": verification_counts,
                    "dwell_s": dwell_s,
                    "estimated_dose_rate_usv_h": verification_estimate,
                    "threshold_usv_h": verification_threshold_usv_h,
                }
            ],
            estimates=[
                {
                    "estimate_id": "post-first-action",
                    "dose_rate_usv_h": verification_estimate,
                    "diagnosis": "decontamination_underperformance",
                    "replan_required": True,
                }
            ],
            actions=action_rows,
            resources=[
                {
                    "strategy": "open_loop",
                    "elapsed_s": 135.0,
                    "waste_generated_kg": 1.8,
                    "shield_mass_used_kg": 8.0,
                },
                {
                    "strategy": "closed_loop",
                    "elapsed_s": 235.0,
                    "waste_generated_kg": 3.0,
                    "shield_mass_used_kg": 8.0,
                },
            ],
            maps={
                "strategy_dose_trajectories": {
                    "open_loop_dose_rate_usv_h": np.array(
                        [initial_dose_rate_usv_h, post_first_dose, open_loop_final]
                    ),
                    "closed_loop_dose_rate_usv_h": np.array(
                        [
                            initial_dose_rate_usv_h,
                            post_first_dose,
                            post_recovery_dose,
                            closed_loop_final,
                        ]
                    ),
                }
            },
        )


@dataclass(frozen=True)
class ResourceConstrainedMultiActionCase:
    """Select a countermeasure subset under time, waste, and mass limits."""

    case_id: str = "resource_constrained_multi_action"

    def run(self, *, seed: int, planner_id: str) -> CaseResult:
        del seed
        candidates = (
            {
                "action_id": "decon-zone-a",
                "dose_reduction": 36.0,
                "elapsed_s": 18.0,
                "waste_kg": 2.0,
                "shield_mass_kg": 0.0,
                "risk": 3.0,
            },
            {
                "action_id": "deploy-shield",
                "dose_reduction": 42.0,
                "elapsed_s": 14.0,
                "waste_kg": 0.0,
                "shield_mass_kg": 8.0,
                "risk": 2.0,
            },
            {
                "action_id": "relocate-object",
                "dose_reduction": 31.0,
                "elapsed_s": 12.0,
                "waste_kg": 0.0,
                "shield_mass_kg": 0.0,
                "risk": 4.0,
            },
            {
                "action_id": "decon-zone-b",
                "dose_reduction": 24.0,
                "elapsed_s": 16.0,
                "waste_kg": 1.5,
                "shield_mass_kg": 0.0,
                "risk": 2.0,
            },
        )
        limits = {"elapsed_s": 32.0, "waste_kg": 3.0, "shield_mass_kg": 8.0}
        masks = np.array(list(product((False, True), repeat=len(candidates))), dtype=bool)
        objective = np.full(len(masks), -np.inf)
        feasible = np.zeros(len(masks), dtype=bool)
        totals: list[dict[str, float]] = []
        for index, mask in enumerate(masks):
            selected = [item for item, enabled in zip(candidates, mask, strict=True) if enabled]
            total = {
                key: float(sum(float(item[key]) for item in selected))
                for key in (
                    "dose_reduction",
                    "elapsed_s",
                    "waste_kg",
                    "shield_mass_kg",
                    "risk",
                )
            }
            totals.append(total)
            feasible[index] = (
                total["elapsed_s"] <= limits["elapsed_s"]
                and total["waste_kg"] <= limits["waste_kg"]
                and total["shield_mass_kg"] <= limits["shield_mass_kg"]
            )
            if feasible[index]:
                objective[index] = (
                    total["dose_reduction"]
                    - 0.35 * total["elapsed_s"]
                    - 1.2 * total["waste_kg"]
                    - 0.2 * total["shield_mass_kg"]
                    - 0.5 * total["risk"]
                )
        best_index = int(np.argmax(objective))
        selected_mask = masks[best_index]
        selected_ids = [
            item["action_id"]
            for item, enabled in zip(candidates, selected_mask, strict=True)
            if enabled
        ]
        best_total = totals[best_index]
        passed = bool(
            feasible[best_index]
            and best_total["elapsed_s"] <= limits["elapsed_s"]
            and best_total["waste_kg"] <= limits["waste_kg"]
            and best_total["shield_mass_kg"] <= limits["shield_mass_kg"]
        )
        action_rows = [
            {
                **item,
                "selected": bool(selected_mask[index]),
                "status": "planned" if selected_mask[index] else "not_selected",
            }
            for index, item in enumerate(candidates)
        ]
        return _result(
            metrics={
                "planner_id": planner_id,
                "selected_action_ids": selected_ids,
                "optimal_objective": float(objective[best_index]),
                "total_dose_reduction": best_total["dose_reduction"],
                "elapsed_s": best_total["elapsed_s"],
                "waste_kg": best_total["waste_kg"],
                "shield_mass_kg": best_total["shield_mass_kg"],
                "passed": passed,
            },
            measurements=[],
            estimates=[
                {
                    "estimate_id": "resource-constrained-plan",
                    "selected_action_ids": selected_ids,
                    "objective": float(objective[best_index]),
                }
            ],
            actions=action_rows,
            resources=[
                {
                    "plan_id": "resource-constrained-plan",
                    **best_total,
                    "elapsed_limit_s": limits["elapsed_s"],
                    "waste_limit_kg": limits["waste_kg"],
                    "shield_mass_limit_kg": limits["shield_mass_kg"],
                }
            ],
            maps={
                "feasible_action_subsets": {
                    "selection_mask": masks,
                    "feasible": feasible,
                    "objective": objective,
                }
            },
        )


EXTENDED_CASES = {
    case.case_id: case
    for case in (
        DecontaminationPrimitiveCase,
        ShieldingPrimitiveCase,
        MovableContaminatedObjectCase,
        HiddenSourceResidualCase,
        ClosedLoopComparisonCase,
        ResourceConstrainedMultiActionCase,
    )
}
