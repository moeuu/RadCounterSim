"""Built-in validation experiment cases."""

from __future__ import annotations

import numpy as np

from radcounter.core.experiments.artifacts import CaseResult
from radcounter.core.experiments.extended_cases import EXTENDED_CASES
from radcounter.core.models import DetectorSpec, EmissionLine, IsotopeSpec, MaterialSpec
from radcounter.core.radiation import (
    AnalyticSlab,
    AnalyticTransportBackend,
    MaterialTable,
)
from radcounter.core.radiation.sampled_forward import SampledRadiationForwardModel
from radcounter.core.radiation.sampling import point_sample_batch


class AnalyticRadiationValidationCase:
    """Validate inverse-square and slab attenuation in one reproducible run."""

    case_id = "analytic_radiation_validation"

    def run(self, *, seed: int, planner_id: str) -> CaseResult:
        del seed
        detector = DetectorSpec(
            "validation",
            np.array([0.0, 200.0]),
            np.array([1.0, 100.0, 200.0]),
            np.ones(3),
            np.zeros(1),
        )
        isotope = (IsotopeSpec("synthetic", (EmissionLine(100.0, 1.0),)),)
        source = point_sample_batch(np.zeros((1, 3)), np.array([1.0]), np.array([0]), np.array([0]))
        free_backend = AnalyticTransportBackend()
        free_backend.build_scene([], MaterialTable(()))
        free_model = SampledRadiationForwardModel(free_backend)
        free_prediction = free_model.predict_count_rate(
            np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            source,
            isotope,
            detector,
        ).direct_count_rate_cps_per_bin[:, 0]
        inverse_square_ratio = float(free_prediction[0] / free_prediction[1])
        material = MaterialSpec("synthetic_slab", np.array([10.0, 1000.0]), np.array([2.0, 2.0]))
        slab_backend = AnalyticTransportBackend()
        slab_backend.build_scene(
            [
                AnalyticSlab(
                    "slab",
                    "synthetic_slab",
                    np.zeros(3),
                    np.array([1.0, 0.0, 0.0]),
                    0.5,
                )
            ],
            MaterialTable((material,)),
        )
        transmission = float(
            slab_backend.trace_transmission(
                np.array([[-1.0, 0.0, 0.0]]),
                np.array([[1.0, 0.0, 0.0]]),
                np.array([100.0]),
            )[0, 0]
        )
        metrics = {
            "planner_id": planner_id,
            "inverse_square_relative_error": abs(inverse_square_ratio - 4.0) / 4.0,
            "slab_relative_error": abs(transmission - np.exp(-1.0)) / np.exp(-1.0),
        }
        return CaseResult(
            metrics,
            {
                "measurements": (
                    {
                        "detector_id": detector.detector_id,
                        "distance_m": 1.0,
                        "count_rate_cps": free_prediction[0],
                    },
                    {
                        "detector_id": detector.detector_id,
                        "distance_m": 2.0,
                        "count_rate_cps": free_prediction[1],
                    },
                ),
                "estimates": (),
                "actions": (),
                "resources": (),
            },
            {"free_space": {"distance_m": np.array([1.0, 2.0]), "rate": free_prediction}},
            (
                {"event_type": "radiation_scene_committed", "sim_time_s": 0.0},
                {"event_type": "measurement_completed", "sim_time_s": 1.0},
            ),
        )


BUILTIN_CASES = {
    AnalyticRadiationValidationCase.case_id: AnalyticRadiationValidationCase,
    **EXTENDED_CASES,
}
