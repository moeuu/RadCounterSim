# Arbitrary robot import and control

The portable robot layer is configuration-driven. It does not require a
robot-specific Python class for normal articulated systems.

Configurable LiDAR, camera, and radiation-detector mounts are documented in
`docs/robot-sensor-rigs.md`.

## Supported assets

- USD/USDA/USDC: referenced directly.
- URDF: converted with the Isaac Sim 6 URDF importer.
- Xacro: expanded with the ROS xacro executable and then imported as URDF.
- MJCF/XML: converted with the Isaac Sim MJCF importer.
- package:// resources: resolved from configured ROS package roots.
- HTTP(S) robot assets: downloaded into the content-addressed robot cache.

The importer caches converted USD by source bytes and complete import
configuration. Imported roots are tagged as robot-controlled streaming focus
points.

## Configuration

See configs/robots/fleet.example.yaml. A robot description declares:

- source format, package roots, initial pose, and import/collision options;
- named or wildcard joint groups with position, velocity, or effort control;
- differential, Ackermann, floating, or matrix-defined mobile-base mapping;
- one or more grippers with open and closed joint positions;
- optional Lula description for Cartesian inverse kinematics.

For mecanum, omni-wheel, tracked, or unusual vehicles, set
twist_to_joint_velocity to an N x 3 matrix. It maps
[linear_x, linear_y, angular_z] to the configured N wheel/joint velocities.

## Isaac usage

~~~python
from radcounter.core.robots import load_robot_fleet, TwistCommand
from radcounter.isaac.robot import RobotFleetManager

fleet = load_robot_fleet("configs/robots/fleet.example.yaml")
manager = RobotFleetManager(stage, fleet)
manager.import_all()

# Bind only after physics/articulations are initialized.
capabilities = manager.bind_all()
manager.controller("measurement_robot").command_twist(
    TwistCommand(linear_x_m_s=0.5, angular_z_rad_s=0.2)
)
manager.controller("mitigation_robot").command_gripper(
    "tool_gripper", fraction_closed=1.0
)
~~~

bind_all() audits actual DOF names and reports unresolved configured groups.
This catches a wrong URDF/MJCF joint name before a mitigation action starts.

The generic controller intentionally exposes named joints, base twists, and
grippers. Robot-specific locomotion, whole-body control, or proprietary
hardware can implement RobotControllerPlugin and register a custom factory
without changing planners or workflow code. ROS 2 controllers may use the same
descriptor as their joint/topic mapping source.
