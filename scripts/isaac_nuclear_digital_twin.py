#!/usr/bin/env python3
"""Open a real facility digital twin with adaptive nuclear rendering."""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from radcounter.core.rendering import RenderPurpose, load_digital_twin_config


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("descriptor", help="Digital-twin rendering YAML")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--capture", action="store_true", help="Use capture renderer policy")
    parser.add_argument("--duration-s", type=float, default=0.0, help="0 runs until GUI closes")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    config = load_digital_twin_config(args.descriptor)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless})
    import omni.timeline
    import omni.usd
    from radcounter.isaac.rendering import NuclearDigitalTwinRuntime

    context = omni.usd.get_context()
    if context.get_stage() is None:
        context.new_stage()
    stage = context.get_stage()
    purpose = RenderPurpose.CAPTURE if args.capture else RenderPurpose.INTERACTIVE
    runtime = NuclearDigitalTwinRuntime(stage, config, purpose=purpose)
    task = asyncio.ensure_future(runtime.load())
    while not task.done() and app.is_running():
        app.update()
    if task.cancelled() or task.exception() is not None:
        runtime.close()
        app.close()
        if task.exception() is not None:
            raise task.exception()
        raise RuntimeError("digital-twin loading was cancelled")
    print(json.dumps(task.result().as_dict(), indent=2))

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    started = time.monotonic()
    previous = started
    try:
        while app.is_running():
            app.update()
            now = time.monotonic()
            runtime.observe_frame_time((now - previous) * 1000.0)
            previous = now
            if args.duration_s > 0.0 and now - started >= args.duration_s:
                break
    finally:
        timeline.stop()
        runtime.close()
        app.close()


if __name__ == "__main__":
    main()
