"""Live Kit gate for the operations dashboard and volume-source sampling."""

from __future__ import annotations

import json
import sys
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
        dashboard.set_workflow_view(
            {
                "estimate": {"basis_count": 4, "source_strength_bq": [1.0e6]},
                "residual": {"confidence": 0.82, "best_hypothesis": "shield_pose"},
                "selected_action": {"action_id": "repair-shield-1"},
                "prediction": {"predicted_rate_cps": 18.2},
            }
        )
        assert "basis_count" in dashboard._estimate.get_value_as_string()
        assert "shield_pose" in dashboard._residual.get_value_as_string()
        assert "repair-shield-1" in dashboard._plan.get_value_as_string()
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
