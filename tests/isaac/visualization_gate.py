"""Live USD gate for independent operator visualization layers."""

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
    try:
        from pxr import Usd, UsdGeom
        from radcounter.isaac.visualization import (
            ActionVisualizer,
            ColorRangeMode,
            DoseMapStyle,
            DoseMapVisualizer,
            RayDebugVisualizer,
            ResidualVisualizer,
            SourceEstimateVisualizer,
        )

        stage = Usd.Stage.CreateInMemory()
        positions = np.asarray(((0.0, 0.0, 0.5), (1.0, 0.0, 0.5), (2.0, 0.0, 0.5)))
        dose_path = DoseMapVisualizer(stage).author(
            positions,
            (1.0, 10.0, 100.0),
            style=DoseMapStyle(mode=ColorRangeMode.PERCENTILE),
            data_class="count_rate_proxy_not_dose_calibrated",
        )
        estimate_path, uncertainty_path = SourceEstimateVisualizer(stage).author(
            positions[:2],
            (1.0e6, 3.0e6),
            position_standard_deviation_m=((0.1, 0.2, 0.1), (0.2, 0.2, 0.2)),
        )
        predicted_path, observed_path, residual_path = ResidualVisualizer(stage).author(
            positions,
            (10.0, 20.0, 30.0),
            (12.0, 18.0, 35.0),
            (0.5, -0.4, 1.2),
        )
        ray_path = RayDebugVisualizer(stage).author(
            positions[0], positions[2], {"concrete": 0.4, "lead": 0.05}
        )
        route_path, tool_path, target_path = ActionVisualizer(stage).author(
            base_route_world_m=positions,
            tool_path_world_m=positions[::-1],
            target_world_m=positions[1],
        )
        paths = (
            dose_path,
            estimate_path,
            uncertainty_path,
            predicted_path,
            observed_path,
            residual_path,
            ray_path,
            route_path,
            tool_path,
            target_path,
        )
        assert all(path is not None and stage.GetPrimAtPath(path).IsValid() for path in paths)
        assert len(UsdGeom.Points(stage.GetPrimAtPath(dose_path)).GetPointsAttr().Get()) == 3
        ray = stage.GetPrimAtPath(ray_path)
        assert list(ray.GetAttribute("rad:visualization:materialIds").Get()) == [
            "concrete",
            "lead",
        ]
        assert np.allclose(
            ray.GetAttribute("rad:visualization:materialPathLengthsM").Get(),
            (0.4, 0.05),
        )
        assert (
            UsdGeom.BasisCurves(ray).GetWidthsInterpolation()
            == UsdGeom.Tokens.constant
        )
        for path in (route_path, tool_path, target_path):
            assert stage.GetPrimAtPath(path).GetAttribute("rad:visualization:annotationOnly").Get()
        for path in (route_path, tool_path):
            assert (
                UsdGeom.BasisCurves(stage.GetPrimAtPath(path)).GetWidthsInterpolation()
                == UsdGeom.Tokens.constant
            )
        print(json.dumps({"success": True, "layers": len(paths)}), flush=True)
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
