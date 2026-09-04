#!/usr/bin/env python3
"""Run the RadInterAct vertical slice inside the uv-managed Isaac runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "source/extensions/radcounter.isaac"))

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless})
    try:
        import omni.usd
        from radcounter.isaac.runtime.simulation import (
            IsaacRadiationSimulation,
            measurement_payload,
        )

        stage_path = root / "assets/environments/radcounter_vertical_slice.usda"
        config_path = root / "configs/scenarios/vertical_slice.runtime.json"
        context = omni.usd.get_context()
        if not context.open_stage(str(stage_path)):
            raise RuntimeError(f"failed to open {stage_path}")
        for _ in range(8):
            app.update()
        simulation = IsaacRadiationSimulation.from_config(context.get_stage(), config_path)
        records = simulation.measure(duration_s=args.duration_s)
        payload = {
            "stage": str(stage_path),
            "truth_source_paths": [source.prim_path for source in simulation.sources],
            "belief_source_paths": list(simulation.belief_source_paths),
            **measurement_payload(records),
        }
        serialized = json.dumps(payload, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
