#!/usr/bin/env python3
"""Run a repeatable shield pose-error sweep in the uv-managed Isaac runtime."""

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
        from pxr import Gf
        from radcounter.isaac.runtime.simulation import (
            IsaacRadiationSimulation,
            RuntimeConfiguration,
        )

        from radcounter.core.experiments import ExperimentArtifactBundle, ExperimentCaseRecord

        stage_path = root / "assets/environments/radcounter_vertical_slice.usda"
        config_path = root / "configs/scenarios/vertical_slice.runtime.json"
        run_id = arguments.run_id or datetime.now(UTC).strftime("vertical-slice-%Y%m%dT%H%M%SZ")
        bundle = ExperimentArtifactBundle(
            root / arguments.output_root,
            run_id,
            stage_path=stage_path,
            config_path=config_path,
            metadata={
                "experiment": "shield_pose_error_residual",
                "estimator_modified": False,
                "physics_data_class": "synthetic_validation_only",
            },
        )
        protected = "/World/DetectorStations/Protected"
        hidden = "/World/HiddenContaminatedDrum"
        nominal = np.asarray([1.2, 0.0, 0.9])
        context = omni.usd.get_context()
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
            shield = stage.GetPrimAtPath("/World/LeadShield")
            shield.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*nominal))
            simulation.synchronize()
            predicted = simulation.expected_rates(
                detector_paths=[protected], source_paths=[hidden]
            )[protected]
            rng = np.random.default_rng(seed)
            pose_error = rng.normal(0.0, arguments.translation_std_m, size=3)
            pose_error[2] = 0.0
            actual = nominal + pose_error
            shield.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*actual))
            simulation.synchronize()
            observed_expected = simulation.expected_rates(
                detector_paths=[protected], source_paths=[hidden]
            )[protected]
            observed = simulation.measure(detector_paths=[protected], source_paths=[hidden])[0]
            bundle.record(
                ExperimentCaseRecord(
                    case_id=f"seed-{seed:05d}",
                    seed=seed,
                    parameters={
                        "shield_error_x_m": float(pose_error[0]),
                        "shield_error_y_m": float(pose_error[1]),
                        "shield_error_z_m": float(pose_error[2]),
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
