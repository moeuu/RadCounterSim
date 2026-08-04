import numpy as np

from radcounter.core.robots import (
    BaseControllerConfig,
    GripperConfig,
    JointGroupConfig,
    RobotFleetConfig,
    RobotImportConfig,
    TwistCommand,
    interpolate_gripper,
    resolve_joint_groups,
    twist_to_joint_velocity,
)


def test_robot_fleet_accepts_urdf_and_joint_patterns() -> None:
    fleet = RobotFleetConfig(
        robots=(
            RobotImportConfig(
                id="robot",
                uri="robot.urdf",
                joint_groups=(
                    JointGroupConfig(name="arm", joint_patterns=("arm_*",)),
                ),
            ),
        )
    )
    groups, unresolved = resolve_joint_groups(
        ("left_wheel", "arm_1", "arm_2"),
        fleet.robots[0].joint_groups,
    )
    assert groups["arm"] == ("arm_1", "arm_2")
    assert unresolved == ()


def test_differential_base_maps_twist_to_wheels() -> None:
    config = BaseControllerConfig(
        type="differential",
        joint_names=("left", "right"),
        wheel_radius_m=0.1,
        track_width_m=0.5,
    )
    result = twist_to_joint_velocity(
        TwistCommand(linear_x_m_s=1.0, angular_z_rad_s=2.0),
        config,
    )
    np.testing.assert_allclose(result, [5.0, 15.0])


def test_custom_matrix_and_gripper_are_simulator_independent() -> None:
    base = BaseControllerConfig(
        type="holonomic",
        joint_names=("a", "b"),
        twist_to_joint_velocity=((1.0, 1.0, 0.0), (1.0, -1.0, 0.0)),
    )
    np.testing.assert_allclose(
        twist_to_joint_velocity(
            TwistCommand(linear_x_m_s=2.0, linear_y_m_s=0.5),
            base,
        ),
        [2.5, 1.5],
    )
    gripper = GripperConfig(
        joint_names=("left", "right"),
        open_positions=(0.04, 0.04),
        closed_positions=(0.0, 0.0),
    )
    np.testing.assert_allclose(interpolate_gripper(gripper, 0.5), [0.02, 0.02])
