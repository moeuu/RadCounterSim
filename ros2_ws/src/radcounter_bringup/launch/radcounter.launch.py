"""Launch the namespaced RadCounterSim ROS bridge."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    default_config = str(
        Path(get_package_share_directory("radcounter_bringup")) / "config/bridge.yaml"
    )
    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    config = LaunchConfiguration("config")
    return LaunchDescription(
        [
            DeclareLaunchArgument("namespace", default_value="radcounter"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("config", default_value=default_config),
            Node(
                package="radcounter_bringup",
                executable="radcounter_bridge",
                name="bridge",
                namespace=namespace,
                output="screen",
                parameters=[config, {"use_sim_time": use_sim_time}],
            ),
        ]
    )
