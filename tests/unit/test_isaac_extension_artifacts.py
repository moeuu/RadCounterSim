import tomllib
from pathlib import Path


def test_isaac_extension_declares_direct_host_dependencies() -> None:
    root = Path(__file__).resolve().parents[2]
    extension_root = root / "source/extensions/radcounter.isaac"
    config = tomllib.loads((extension_root / "config/extension.toml").read_text(encoding="utf-8"))
    assert config["python"]["module"][0]["name"] == "radcounter.isaac"
    assert set(config["dependencies"]) == {
        "omni.kit.uiapp",
        "omni.usd",
        "omni.timeline",
        "omni.physx",
    }


def test_isaac_extension_contains_runtime_physics_and_embree_adapters() -> None:
    root = Path(__file__).resolve().parents[2]
    package = root / "source/extensions/radcounter.isaac/radcounter/isaac"
    required = (
        "extension.py",
        "runtime/session.py",
        "ui/window.py",
        "physics/actions.py",
        "usd/embree_scene.py",
    )
    assert all((package / relative_path).is_file() for relative_path in required)
