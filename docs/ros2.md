# ROS 2 Jazzy interface

The workspace defines `radcounter_msgs` and a `radcounter_bringup` bridge node. The
node exposes namespaced endpoints:

- Actions: `measure_radiation`, `execute_countermeasure`
- Services: `get_dose_map`, `evaluate_countermeasure`, `reset_episode`
- Topics: `measurements`, `source_estimates`, `countermeasure_status`

Build and launch on a ROS 2 Jazzy host:

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
ros2 launch radcounter_bringup radcounter.launch.py namespace:=radcounter
```

The default node uses a deterministic analytic host so interface round trips can be
tested without Isaac Sim. `HostBridge` is the replacement boundary for an
Isaac-hosted implementation. On the Isaac side, `Ros2RobotController` requires a
live `rclpy` node whose executor is already spinning; it does not create a hidden
executor thread.

Python project commands outside the sourced ROS environment remain managed through
`uv`. ROS-generated packages are supplied by the Jazzy/colcon workspace rather than
installed from PyPI.
