from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    config = Path(get_package_share_directory("radcounter_robot_gateway")) / "config/gateway.yaml"
    return LaunchDescription(
        [
            Node(
                package="radcounter_robot_gateway",
                executable="robot_gateway",
                name="radcounter_robot_gateway",
                output="screen",
                parameters=[str(config)],
            )
        ]
    )
