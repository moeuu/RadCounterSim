from pathlib import Path
from xml.etree import ElementTree


def test_ros_message_action_service_artifacts_are_complete() -> None:
    root = Path(__file__).resolve().parents[2] / "ros2_ws/src/radcounter_msgs"
    expected = {
        "msg/RadiationMeasurement.msg",
        "msg/SourceEstimate.msg",
        "msg/CountermeasureStatus.msg",
        "action/MeasureRadiation.action",
        "action/ExecuteCountermeasure.action",
        "srv/GetDoseMap.srv",
        "srv/EvaluateCountermeasure.srv",
        "srv/ResetEpisode.srv",
    }
    assert all((root / path).is_file() for path in expected)
    radiation = (root / "msg/RadiationMeasurement.msg").read_text(encoding="utf-8")
    assert "float64[] energy_bin_edges_kev" in radiation
    assert "uint64 scene_revision" in radiation
    cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    assert all(f'"{path}"' in cmake for path in expected)


def test_ros_packages_are_jazzy_ament_packages() -> None:
    root = Path(__file__).resolve().parents[2] / "ros2_ws/src"
    messages = ElementTree.parse(root / "radcounter_msgs/package.xml").getroot()
    bringup = ElementTree.parse(root / "radcounter_bringup/package.xml").getroot()
    assert messages.findtext("name") == "radcounter_msgs"
    assert bringup.findtext("name") == "radcounter_bringup"
    assert any(
        item.text == "rosidl_default_generators" for item in messages.findall("build_depend")
    )
    assert any(item.text == "rclpy" for item in bringup.findall("exec_depend"))


def test_ros_adapter_is_optional_for_core_imports() -> None:
    import radcounter.core

    assert radcounter.core is not None
