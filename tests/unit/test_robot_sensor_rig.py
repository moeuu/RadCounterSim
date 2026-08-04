import numpy as np
import pytest
from pydantic import ValidationError

from radcounter.core.robots import (
    BaseControllerConfig,
    CameraSensorConfig,
    LidarDimension,
    LidarSensorConfig,
    RobotFleetConfig,
    RobotImportConfig,
    RobotSensorBackendRegistry,
    SensorMountConfig,
    SpatialVelocityCommand,
    spatial_to_joint_velocity,
)


def test_robot_accepts_multiple_configurable_sensor_mounts() -> None:
    robot = RobotImportConfig(
        id="drone",
        uri="drone.urdf",
        sensors=(
            SensorMountConfig(
                id="lidar",
                type="lidar",
                parent_link="base_link",
                lidar=LidarSensorConfig(dimension=LidarDimension.VOLUMETRIC),
            ),
            SensorMountConfig(
                id="rgbd",
                type="camera",
                parent_link="camera_link",
                camera=CameraSensorConfig(
                    annotators=("rgb", "distance_to_image_plane")
                ),
            ),
        ),
    )
    assert len(robot.sensor_mounts) == 2
    assert robot.sensor_mounts[1].camera is not None
    assert robot.sensor_mounts[1].camera.width_px == 1280


def test_sensor_type_requires_matching_configuration() -> None:
    with pytest.raises(ValidationError, match="requires a lidar block"):
        SensorMountConfig(id="missing", type="lidar")


def test_fleet_rejects_duplicate_robot_and_sensor_ids() -> None:
    sensor = SensorMountConfig(
        id="camera",
        type="camera",
        camera=CameraSensorConfig(),
    )
    with pytest.raises(ValidationError, match="duplicate sensor IDs"):
        RobotImportConfig(id="robot", uri="robot.usd", sensors=(sensor, sensor))
    robot = RobotImportConfig(id="robot", uri="robot.usd")
    with pytest.raises(ValidationError, match="Robot IDs must be unique"):
        RobotFleetConfig(robots=(robot, robot))


def test_aerial_spatial_velocity_is_clamped() -> None:
    config = BaseControllerConfig(
        type="aerial",
        maximum_linear_speed_m_s=2.0,
        maximum_vertical_speed_m_s=1.0,
        maximum_roll_pitch_rate_rad_s=0.5,
        maximum_angular_speed_rad_s=1.5,
    )
    result = spatial_to_joint_velocity(
        SpatialVelocityCommand(
            linear_x_m_s=4.0,
            linear_y_m_s=-3.0,
            linear_z_m_s=2.0,
            angular_x_rad_s=1.0,
            angular_y_rad_s=-1.0,
            angular_z_rad_s=3.0,
        ),
        config,
    )
    np.testing.assert_allclose(result, [2.0, -2.0, 1.0, 0.5, -0.5, 1.5])


def test_custom_six_axis_mixer_and_sensor_backend() -> None:
    config = BaseControllerConfig(
        type="custom_flight",
        joint_names=("rotor_a", "rotor_b"),
        spatial_to_joint_velocity=(
            (1.0, 0.0, 1.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0, 0.0, 0.0, -1.0),
        ),
    )
    values = spatial_to_joint_velocity(
        SpatialVelocityCommand(
            linear_x_m_s=1.0,
            linear_z_m_s=2.0,
            angular_z_rad_s=0.5,
        ),
        config,
    )
    np.testing.assert_allclose(values, [3.5, 2.5])

    sentinel = object()
    registry = RobotSensorBackendRegistry()
    registry.register("vendor", lambda **_: sentinel)
    assert registry.create("vendor", path="/World/Sensor") is sentinel


def test_isaac_adapter_uses_non_deprecated_rtx_sensor_api() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    source = (
        root
        / "source/extensions/radcounter.isaac/radcounter/isaac/robot/sensor_rig.py"
    ).read_text(encoding="utf-8")
    assert "isaacsim.sensors.experimental.rtx import Lidar, LidarSensor" in source
    assert "isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera" in source
    assert "isaacsim.sensors.camera" not in source
