"""Performance and selective runtime-cache gate under actual Isaac/Embree."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "source/extensions/radcounter.isaac")]


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import omni.usd
        from pxr import Gf
        from radcounter.isaac.runtime.simulation import IsaacRadiationSimulation

        context = omni.usd.get_context()
        assert context.open_stage(str(ROOT / "assets/environments/radcounter_vertical_slice.usda"))
        for _ in range(6):
            app.update()
        stage = context.get_stage()
        simulation = IsaacRadiationSimulation.from_config(
            stage,
            ROOT / "configs/scenarios/vertical_slice.runtime.json",
        )
        points = np.asarray(
            [(x, y, 0.8) for y in np.linspace(-3.0, 3.0, 10) for x in np.linspace(-5.0, 5.0, 16)]
        )
        started = time.perf_counter()
        first = simulation.dose_proxy_map(points)
        uncached_seconds = time.perf_counter() - started
        first_stats = simulation.transport.statistics
        started = time.perf_counter()
        second = simulation.dose_proxy_map(points)
        cached_seconds = time.perf_counter() - started
        second_stats = simulation.transport.statistics
        assert np.allclose(first, second)
        assert second_stats["traced_rays"] == first_stats["traced_rays"]
        assert second_stats["cache_hits"] > first_stats["cache_hits"]

        source = stage.GetPrimAtPath("/World/HiddenContaminatedDrum")
        source.GetAttribute("rad:source:activityBq").Set(4.0e7)
        simulation.refresh_scene_state()
        simulation.dose_proxy_map(points)
        activity_stats = simulation.transport.statistics
        assert activity_stats["traced_rays"] == second_stats["traced_rays"]

        shield = stage.GetPrimAtPath("/World/LeadShield")
        shield.GetAttribute("xformOp:translate").Set(Gf.Vec3d(1.2, 0.0, 0.9))
        changed = simulation.synchronize()
        final_stats = simulation.transport.statistics
        assert "/World/LeadShield" in changed
        assert final_stats["selectively_patched_rays"] > 0
        assert final_stats["selectively_patched_rays"] < first_stats["traced_rays"]
        assert uncached_seconds < 5.0
        assert cached_seconds < uncached_seconds
        print(
            json.dumps(
                {
                    "uncached_seconds": uncached_seconds,
                    "cached_seconds": cached_seconds,
                    "statistics": final_stats,
                }
            ),
            flush=True,
        )
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
