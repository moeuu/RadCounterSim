"""Live Kit gate for the optional Fukushima CAD system profile."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "source/extensions/radcounter.isaac"),
    str(ROOT / "build/native/python"),
]


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    exit_code = 0
    try:
        import omni.usd
        from pxr import UsdGeom
        from radcounter.isaac.runtime import IsaacRadiationSimulation
        from radcounter.isaac.system_profile import (
            compose_selected_system,
            prepare_environment_stage,
        )

        from radcounter.core.system_profiles import resolve_system_selection

        selection = resolve_system_selection(profile_id="fukushima-packbot")
        assert selection.environment_ready, selection.environment_source_path
        stage_path, environment_manifest = prepare_environment_stage(selection)
        context = omni.usd.get_context()
        assert context.open_stage(str(stage_path))
        for _ in range(12):
            app.update()
        stage = context.get_stage()
        composed = compose_selected_system(stage, selection, stage_path=stage_path)
        simulation = IsaacRadiationSimulation.from_config(stage, composed.runtime_config_path)
        mesh_count = sum(prim.IsA(UsdGeom.Mesh) for prim in stage.Traverse())

        assert mesh_count >= 995
        assert set(composed.robot_paths) == {"packbot", "elios3"}
        assert set(composed.detector_paths) == {"packbot-gm", "elios-imager"}
        assert len(simulation.detectors) == 2
        assert len(simulation.transport.geometry_paths) >= 995
        os.write(
            1,
            (
                json.dumps(
                    {
                        "profile": selection.profile_id,
                        "stage": str(stage_path),
                        "environment_manifest": str(environment_manifest),
                        "meshes": mesh_count,
                        "robots": sorted(composed.robot_paths),
                        "detectors": sorted(composed.detector_paths),
                        "transport_geometries": len(simulation.transport.geometry_paths),
                    }
                )
                + "\n"
            ).encode(),
        )
    except BaseException:
        os.write(2, traceback.format_exc().encode())
        exit_code = 1
    finally:
        app.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
