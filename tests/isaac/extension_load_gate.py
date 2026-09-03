"""Load the packaged extension manifest and every declared Python module."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        manager.add_path(str(ROOT / "source/extensions"))
        enabled = manager.set_extension_enabled_immediate("radcounter.isaac", True)
        for _ in range(4):
            app.update()
        assert enabled is not False
        assert manager.is_extension_enabled("radcounter.isaac")
        extension = manager.get_extension_dict("radcounter.isaac")
        assert extension["package"]["version"] == "0.2.0"
        print(json.dumps({"enabled": True, "version": extension["package"]["version"]}), flush=True)
        manager.set_extension_enabled_immediate("radcounter.isaac", False)
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
