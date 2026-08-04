from pathlib import Path


def test_generic_robot_adapter_uses_isaac6_articulation_and_importers() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (
        root
        / "source/extensions/radcounter.isaac/radcounter/isaac/robot/generic.py"
    ).read_text(encoding="utf-8")
    assert "isaacsim.core.experimental.prims import Articulation" in source
    assert "isaacsim.asset.importer.urdf" in source
    assert "isaacsim.asset.importer.mjcf" in source
    assert "initialize_cpp_data_view" in source
