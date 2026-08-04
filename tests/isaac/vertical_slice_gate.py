"""Isaac/Embree vertical-slice execution gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "source/extensions/radcounter.isaac"))


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import omni.usd
        from pxr import Gf, UsdGeom
        from radcounter.isaac.runtime.simulation import IsaacRadiationSimulation

        context = omni.usd.get_context()
        stage_path = ROOT / "assets/environments/radcounter_vertical_slice.usda"
        if not context.open_stage(str(stage_path)):
            raise AssertionError("vertical-slice USD did not open")
        for _ in range(8):
            app.update()
        stage = context.get_stage()
        simulation = IsaacRadiationSimulation.from_config(
            stage,
            ROOT / "configs/scenarios/vertical_slice.runtime.json",
        )
        protected = "/World/DetectorStations/Protected"
        hidden = "/World/HiddenContaminatedDrum"
        before = simulation.expected_rates(detector_paths=[protected], source_paths=[hidden])[
            protected
        ]
        shield = stage.GetPrimAtPath("/World/LeadShield")
        translate = shield.GetAttribute("xformOp:translate")
        translate.Set(Gf.Vec3d(1.2, 0.0, 0.9))
        UsdGeom.Xformable(shield).SetXformOpOrder(
            [
                UsdGeom.Xformable(shield).GetOrderedXformOps()[0],
                UsdGeom.Xformable(shield).GetOrderedXformOps()[1],
            ]
        )
        changed = simulation.synchronize()
        after = simulation.expected_rates(detector_paths=[protected], source_paths=[hidden])[
            protected
        ]
        records = simulation.measure(detector_paths=[protected])
        assert len(simulation.sources) == 2
        assert hidden not in simulation.belief_source_paths
        assert "/World/LeadShield" in changed
        assert after < before * 0.25, (before, after)
        assert records[0].counts >= 0
        print(json.dumps({"before_cps": before, "after_cps": after, "changed": changed}))
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
