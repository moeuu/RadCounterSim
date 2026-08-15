import tomllib
from pathlib import Path


def test_isaac_extension_declares_direct_host_dependencies() -> None:
    root = Path(__file__).resolve().parents[2]
    extension_root = root / "source/extensions/radcounter.isaac"
    config = tomllib.loads((extension_root / "config/extension.toml").read_text(encoding="utf-8"))
    assert config["python"]["module"][0]["name"] == "radcounter.isaac"
    assert set(config["dependencies"]) == {
        "omni.kit.uiapp",
        "omni.ui.scene",
        "omni.kit.viewport.utility",
        "omni.kit.viewport.window",
        "omni.appwindow",
        "omni.usd",
        "omni.timeline",
        "omni.physx",
        "isaacsim.ros2.bridge",
        "isaacsim.robot_motion.motion_generation",
        "isaacsim.robot.wheeled_robots",
        "isaacsim.asset.importer.urdf",
        "isaacsim.asset.importer.mjcf",
        "isaacsim.core.experimental.prims",
    }
    assert config["dependencies"]["isaacsim.ros2.bridge"]["optional"] is True
    assert config["dependencies"]["isaacsim.robot_motion.motion_generation"]["optional"] is True
    assert config["dependencies"]["isaacsim.robot.wheeled_robots"]["optional"] is True
    assert config["dependencies"]["isaacsim.asset.importer.urdf"]["optional"] is True
    assert config["dependencies"]["isaacsim.asset.importer.mjcf"]["optional"] is True
    assert config["dependencies"]["isaacsim.core.experimental.prims"]["optional"] is True


def test_isaac_extension_contains_runtime_physics_and_embree_adapters() -> None:
    root = Path(__file__).resolve().parents[2]
    package = root / "source/extensions/radcounter.isaac/radcounter/isaac"
    required = (
        "extension.py",
        "runtime/session.py",
        "ui/window.py",
        "physics/actions.py",
        "usd/embree_scene.py",
        "robot/input_router.py",
        "robot/control_window.py",
    )
    assert all((package / relative_path).is_file() for relative_path in required)
