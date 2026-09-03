#!/usr/bin/env python3
"""Run a kinematic shield scene-edit sensitivity sweep in Isaac Sim.

This experiment deliberately edits the USD scene state.  Its artifacts cannot
be used as evidence that a robot placed the shield.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--translation-std-m", type=float, default=0.04)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "source/extensions/radcounter.isaac")]
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import omni.usd
        from radcounter.isaac.physics import Pose3D, UsdSceneStateEditor
        from radcounter.isaac.runtime.simulation import (
            IsaacRadiationSimulation,
            RuntimeConfiguration,
        )

        from radcounter.core.experiments import (
            EvidenceClass,
            ExperimentArtifactBundle,
            ExperimentCaseRecord,
        )

        stage_path = root / "assets/environments/radcounter_vertical_slice.usda"
        config_path = root / "configs/scenarios/vertical_slice.runtime.json"
        run_id = arguments.run_id or datetime.now(UTC).strftime("vertical-slice-%Y%m%dT%H%M%SZ")
        bundle = ExperimentArtifactBundle(
            root / arguments.output_root,
            run_id,
            stage_path=stage_path,
            config_path=config_path,
            evidence_class=EvidenceClass.KINEMATIC_SCENE_EDIT,
            metadata={
                "experiment": "shield_scene_edit_pose_error_residual",
                "operation_execution": EvidenceClass.KINEMATIC_SCENE_EDIT.value,
                "supports_robot_execution_claim": False,
                "estimator_modified": False,
                "physics_data_class": "synthetic_validation_only",
            },
        )
        protected = "/World/DetectorStations/Protected"
        hidden = "/World/HiddenContaminatedDrum"
        nominal = np.asarray([1.2, 0.0, 0.9])
        identity_xyzw = np.asarray([0.0, 0.0, 0.0, 1.0])
        context = omni.usd.get_context()
        editor = UsdSceneStateEditor()
        base_config = RuntimeConfiguration.from_json(config_path)
        for seed in range(arguments.seeds):
            if not context.open_stage(str(stage_path)):
                raise RuntimeError(f"failed to reopen {stage_path}")
            for _ in range(4):
                app.update()
            stage = context.get_stage()
            simulation = IsaacRadiationSimulation(
                stage,
                replace(base_config, seed=seed),
            )
            before = simulation.expected_rates(detector_paths=[protected], source_paths=[hidden])[
                protected
            ]
            editor.set_geometry_pose(
                "/World/LeadShield",
                Pose3D(nominal, identity_xyzw),
            )
            simulation.synchronize()
            predicted = simulation.expected_rates(
                detector_paths=[protected], source_paths=[hidden]
            )[protected]
            rng = np.random.default_rng(seed)
            pose_error = rng.normal(0.0, arguments.translation_std_m, size=3)
            pose_error[2] = 0.0
            actual = nominal + pose_error
            edit_record = editor.set_geometry_pose(
                "/World/LeadShield",
                Pose3D(actual, identity_xyzw),
            )
            simulation.synchronize()
            observed_expected = simulation.expected_rates(
                detector_paths=[protected], source_paths=[hidden]
            )[protected]
            observed = simulation.measure(detector_paths=[protected], source_paths=[hidden])[0]
            bundle.record(
                ExperimentCaseRecord(
                    case_id=f"seed-{seed:05d}",
                    seed=seed,
                    evidence_class=EvidenceClass.KINEMATIC_SCENE_EDIT,
                    parameters={
                        "shield_error_x_m": float(pose_error[0]),
                        "shield_error_y_m": float(pose_error[1]),
                        "shield_error_z_m": float(pose_error[2]),
                        "scene_edit_revision": edit_record.revision,
                        "operation_execution": edit_record.evidence_class.value,
                    },
                    metrics={
                        "before_cps": before,
                        "predicted_after_cps": predicted,
                        "truth_after_cps": observed_expected,
                        "measured_after_cps": observed.measured_rate_cps,
                        "prediction_residual_cps": observed.measured_rate_cps - predicted,
                        "truth_residual_cps": observed_expected - predicted,
                        "shield_reduction_fraction": 1.0 - observed_expected / before,
                        "counts": observed.counts,
                    },
                )
            )
        summary = bundle.finalize()
        print(summary)
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
