import xml.etree.ElementTree as ET
from pathlib import Path


def test_ros_bringup_registers_bridge_executable_and_launch_node() -> None:
    root = Path(__file__).resolve().parents[2]
    package = root / "ros2_ws/src/radcounter_bringup"
    setup_text = (package / "setup.py").read_text(encoding="utf-8")
    launch_text = (package / "launch/radcounter.launch.py").read_text(encoding="utf-8")
    assert "radcounter_bridge = radcounter_bringup.bridge_node:main" in setup_text
    assert 'executable="radcounter_bridge"' in launch_text
    assert (package / "radcounter_bringup/bridge_node.py").is_file()


def test_ros_bringup_declares_launch_runtime_dependencies() -> None:
    root = Path(__file__).resolve().parents[2]
    package_xml = root / "ros2_ws/src/radcounter_bringup/package.xml"
    package = ET.parse(package_xml).getroot()
    dependencies = {item.text for item in package.findall("exec_depend")}
    assert {"rclpy", "radcounter_msgs", "launch", "launch_ros"} <= dependencies
