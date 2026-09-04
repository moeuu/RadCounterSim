# Configurable robot fleets and sensor rigs

RadInterAct separates the vehicle asset from its sensor rig. One fleet can
contain wheeled, tracked, holonomic, articulated, quadruped, floating, and
aerial robots. Every robot can carry any number of LiDARs, cameras, radiation
detectors, or project-specific sensors.

See `configs/robots/multimodal_fleet.example.yaml` for a three-robot example:

- a survey UGV with 3-D LiDAR, RGB-D camera, GM tube, and directional NaI;
- a survey drone with solid-state LiDAR, nadir RGB/depth camera, and CZT;
- a mitigation mobile manipulator with wrist RGB-D and an ion chamber.

## Adjusting a rig

Each item under `sensors` has a `parent_link` and a sensor-to-link `pose`.
Changing these values moves the sensor without editing URDF, USD, Xacro, or
Python code.

~~~yaml
sensors:
  - id: roof_lidar
    type: lidar
    parent_link: base_link
    update_rate_hz: 10
    pose:
      translation_m: [0.0, 0.0, 0.72]
      rotation_rpy_deg: [0.0, 0.0, 0.0]
    lidar:
      dimension: 3d
      model: Example_Rotary
      annotators: [generic-model-output]
  - id: front_rgbd
    type: camera
    parent_link: base_link
    pose:
      translation_m: [0.32, 0.0, 0.58]
      rotation_rpy_deg: [0.0, -8.0, 0.0]
    camera:
      width_px: 1280
      height_px: 720
      annotators: [rgb, distance_to_image_plane]
  - id: directional_detector
    type: radiation
    parent_link: sensor_pan_link
    radiation:
      detector_model_id: collimated_nai
      integration_time_s: 0.5
~~~

`parent_link` may also be an absolute USD prim path. This is useful when a
sensor is attached to a fixture or a dynamically selected payload mount.

## Isaac Sim usage

Mount sensors after importing the robots and composing their referenced USD:

~~~python
from radcounter.core.robots import load_robot_fleet
from radcounter.isaac.robot import RobotFleetManager, IsaacRobotSensorRigManager

fleet = load_robot_fleet("configs/robots/multimodal_fleet.example.yaml")
robots = RobotFleetManager(stage, fleet)
robots.import_all()

rig = IsaacRobotSensorRigManager(stage)
rig.mount_fleet(fleet)

point_cloud, lidar_info = rig.read(
    "survey_ugv", "roof_lidar", "generic-model-output"
)
rgb, rgb_info = rig.read("survey_ugv", "front_rgbd", "rgb")
depth, depth_info = rig.read(
    "survey_ugv", "front_rgbd", "distance_to_image_plane"
)
detector_xyz, detector_wxyz = rig.world_pose("survey_ugv", "directional_nai")
~~~

LiDAR and camera creation uses the Isaac Sim 6
`isaacsim.sensors.experimental.rtx` authoring/runtime split. Camera annotators
can include `rgb`, `distance_to_camera`, `distance_to_image_plane`, semantic
segmentation, motion vectors, and other Replicator annotators. LiDAR supports
rotating and solid-state RTX configurations and optional point-cloud drawing.

The radiation mount stores the detector catalog ID and exposes its live world
pose. The radiation workflow uses that pose when constructing the detector
array, so moving a robot also moves every attached detector. Set
`detector_config_uri` to use a detector described by the custom YAML/CSV API.

## Drone control

`SpatialVelocityCommand` adds vertical, roll, pitch, and yaw rates without
changing the existing planar `TwistCommand` API:

~~~python
from radcounter.core.robots import SpatialVelocityCommand

robots.controller("survey_drone").command_spatial_velocity(
    SpatialVelocityCommand(
        linear_x_m_s=1.0,
        linear_z_m_s=0.4,
        angular_z_rad_s=0.3,
    )
)
~~~

The `aerial` and `floating` controllers directly command the articulation root
and are intended for deterministic experiments. A physical rotor, aerodynamic,
PX4, ArduPilot, or vendor controller should implement `RobotControllerPlugin`.
Use `spatial_to_joint_velocity` for a custom six-axis mixer or register a
controller plugin when actuator dynamics are required.

## Custom and physical sensors

A sensor backend receives the stage, authored prim path, complete mount config,
backend options, and a live world-pose callback. This keeps vendor SDK and ROS
code outside the simulator-independent model.

~~~python
from radcounter.core.robots import RobotSensorBackendRegistry

registry = RobotSensorBackendRegistry()
registry.register("my_camera", create_my_camera_runtime)
rig = IsaacRobotSensorRigManager(stage, backend_registry=registry)
~~~

For package-based plugins, publish a Python entry point in the
`radcounter.robot_sensors` group. Configure it with `type: custom` and a
`custom.backend` name. The same boundary can wrap USB cameras, serial LiDAR,
ROS 2 subscriptions, network devices, or a proprietary detector SDK.

The `ros` section records namespace, frame, QoS, TF, and per-channel topic
names on the sensor prim. A ROS 2 publishing backend can consume the same
configuration, avoiding a second hand-maintained sensor layout.
