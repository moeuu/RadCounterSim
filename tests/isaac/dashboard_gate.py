"""Live Kit gate for the operations dashboard and volume-source sampling."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "source/extensions/radcounter.isaac")]


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    dashboard = None
    try:
        import omni.usd
        from pxr import Sdf, UsdGeom
        from radcounter.isaac.ui.dashboard import RadCounterDashboard

        context = omni.usd.get_context()
        assert context.open_stage(str(ROOT / "assets/environments/radcounter_vertical_slice.usda"))
        for _ in range(6):
            app.update()
        dashboard = RadCounterDashboard("radcounter.isaac.dashboard.gate")
        app.update()
        dashboard._initialize_runtime()
        assert dashboard.simulation is not None
        dashboard._measure()
        assert len(dashboard._latest_records) == 5
        dashboard._render_dose_map()
        dose_proxy = context.get_stage().GetPrimAtPath("/World/RadiationDoseProxy")
        assert dose_proxy.IsValid()
        assert len(UsdGeom.Points(dose_proxy).GetPointsAttr().Get()) == 504
        dashboard._render_selected_ray()
        selected_ray = context.get_stage().GetPrimAtPath(
            "/World/RadInterActVisualization/SelectedRay"
        )
        assert selected_ray.IsValid()
        assert selected_ray.GetAttribute("rad:visualization:materialIds").Get()

        dashboard._show_truth.set_value(True)
        dashboard._apply_visualization_visibility()
        assert (
            not context.get_stage()
            .GetPrimAtPath("/World/RadInterActVisualization/TruthSources")
            .IsValid()
        )
        dashboard._truth_authoring.set_value(True)
        dashboard._show_truth.set_value(True)
        dashboard._apply_visualization_visibility()
        assert (
            context.get_stage()
            .GetPrimAtPath("/World/RadInterActVisualization/TruthSources")
            .IsValid()
        )

        volume = UsdGeom.Sphere.Define(context.get_stage(), "/World/VolumeSourceGate").GetPrim()
        volume.CreateAttribute("rad:role", Sdf.ValueTypeNames.String, custom=True).Set("source")
        volume.CreateAttribute("rad:source:type", Sdf.ValueTypeNames.String, custom=True).Set(
            "volume"
        )
        volume.CreateAttribute("rad:source:isotopeId", Sdf.ValueTypeNames.String, custom=True).Set(
            "Cs-137"
        )
        volume.CreateAttribute("rad:source:activityBq", Sdf.ValueTypeNames.Double, custom=True).Set(
            1.28e6
        )
        volume.CreateAttribute("rad:source:sampleCount", Sdf.ValueTypeNames.Int, custom=True).Set(
            128
        )
        volume.CreateAttribute(
            "rad:source:volumeDistribution", Sdf.ValueTypeNames.String, custom=True
        ).Set("uniform")
        volume.CreateAttribute(
            "rad:source:hiddenFromEstimator", Sdf.ValueTypeNames.Bool, custom=True
        ).Set(False)
        volume.CreateAttribute("rad:source:enabled", Sdf.ValueTypeNames.Bool, custom=True).Set(True)
        dashboard.simulation.refresh_scene_state()
        sampled = next(
            source
            for source in dashboard.simulation.sources
            if source.prim_path == "/World/VolumeSourceGate"
        )
        assert sampled.positions_m.shape == (128, 3)
        assert np.isclose(np.sum(sampled.activity_bq), 1.28e6)

        with tempfile.TemporaryDirectory(prefix="radcounter-voxel-gate-") as directory:
            voxel_path = Path(directory) / "volume_activity.npz"
            np.savez_compressed(
                voxel_path,
                voxel_centers_local_m=np.asarray(
                    ((0.0, 0.0, 0.0), (0.2, 0.1, -0.1)), dtype=np.float64
                ),
                activity_bq_per_voxel=np.asarray((4.0e5, 6.0e5), dtype=np.float64),
            )
            voxel_digest = hashlib.sha256(voxel_path.read_bytes()).hexdigest()
            voxel = UsdGeom.Sphere.Define(
                context.get_stage(), "/World/VoxelVolumeSourceGate"
            ).GetPrim()
            for name, value_type, value in (
                ("rad:role", Sdf.ValueTypeNames.String, "source"),
                ("rad:source:type", Sdf.ValueTypeNames.String, "volume"),
                ("rad:source:isotopeId", Sdf.ValueTypeNames.String, "Cs-137"),
                (
                    "rad:source:volumeDistribution",
                    Sdf.ValueTypeNames.String,
                    "voxel_map",
                ),
                ("rad:source:activityMapUri", Sdf.ValueTypeNames.String, str(voxel_path)),
                (
                    "rad:source:activityMapSha256",
                    Sdf.ValueTypeNames.String,
                    voxel_digest,
                ),
                ("rad:source:enabled", Sdf.ValueTypeNames.Bool, True),
            ):
                voxel.CreateAttribute(name, value_type, custom=True).Set(value)
            dashboard.simulation.refresh_scene_state()
            voxel_sampled = next(
                source
                for source in dashboard.simulation.sources
                if source.prim_path == "/World/VoxelVolumeSourceGate"
            )
            assert voxel_sampled.source_type == "volume"
            assert np.array_equal(voxel_sampled.element_indices, np.arange(2))
            assert np.isclose(voxel_sampled.total_activity_bq, 1.0e6)

            np.savez_compressed(
                voxel_path,
                voxel_centers_local_m=np.asarray(((1.2, 0.0, 0.0),)),
                activity_bq_per_voxel=np.asarray((1.0e6,)),
            )
            voxel.GetAttribute("rad:source:activityMapSha256").Set(
                hashlib.sha256(voxel_path.read_bytes()).hexdigest()
            )
            try:
                dashboard.simulation.refresh_scene_state()
            except ValueError as error:
                assert "outside its visible USD primitive" in str(error)
            else:
                raise AssertionError("out-of-volume voxel did not fail closed")
        dashboard.set_workflow_view(
            {
                "estimate": {
                    "basis_count": 1,
                    "positions_world_m": [[0.0, 0.0, 0.5]],
                    "source_strength_bq": [1.0e6],
                    "activity_standard_deviation_bq": [2.0e5],
                },
                "residual": {
                    "confidence": 0.82,
                    "best_hypothesis": "shield_pose",
                    "detector_paths": ["/World/MeasurementRobot/Detector"],
                    "predicted_rate_cps": [10.0],
                    "observed_rate_cps": [12.0],
                    "normalized_residual": [0.63],
                },
                "selected_action": {"action_id": "repair-shield-1"},
                "prediction": {"predicted_rate_cps": 18.2},
            }
        )
        assert "basis_count" in dashboard._estimate.get_value_as_string()
        assert "shield_pose" in dashboard._residual.get_value_as_string()
        assert "repair-shield-1" in dashboard._plan.get_value_as_string()
        for path in (
            "/World/RadInterActVisualization/BeliefSources",
            "/World/RadInterActVisualization/SourceUncertainty",
            "/World/RadInterActVisualization/PredictedPostAction",
            "/World/RadInterActVisualization/ObservedPostAction",
            "/World/RadInterActVisualization/NormalizedResidual",
        ):
            assert context.get_stage().GetPrimAtPath(path).IsValid(), path
        dashboard._experiment_run_id.set_value("dashboard-gate")
        dashboard._save_evaluation_snapshot()
        snapshot = json.loads(
            (ROOT / "artifacts/ui/evaluations/dashboard-gate.json").read_text(encoding="utf-8")
        )
        assert snapshot["evidence_class"] == "configuration_and_public_observations"
        assert np.all(np.linalg.norm(sampled.positions_m, axis=1) <= 1.0 + 1.0e-12)
        print(
            json.dumps(
                {
                    "measurements": len(dashboard._latest_records),
                    "dose_proxy_points": 504,
                    "volume_samples": len(sampled.positions_m),
                }
            ),
            flush=True,
        )
        return 0
    finally:
        if dashboard is not None:
            dashboard.shutdown()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
