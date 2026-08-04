# ROS 2 motion gateway

The Jazzy gateway exposes `/radcounter/execute_robot_task` using `radcounter_msgs/action/ExecuteRobotTask`.

It orchestrates these standard actions:

- Nav2: `/navigate_to_pose` (`nav2_msgs/action/NavigateToPose`)
- MoveIt 2: `/move_action` (`moveit_msgs/action/MoveGroup`)
- Gripper: `/gripper_controller/gripper_cmd` (`control_msgs/action/GripperCommand`)
- Isaac physics queue: `/radcounter/isaac/command` and `/radcounter/isaac/result`

Install the user-local message dependencies and build with:

```bash
bash scripts/install_ros_motion_deps.sh
scripts/build_ros2.sh
source scripts/host_env.sh
source ros2_ws/install/setup.bash
ros2 launch radcounter_robot_gateway robot_gateway.launch.py
```

ROS callbacks never mutate USD or PhysX. `IsaacRos2CommandHost.update()` drains commands on the Isaac physics thread and correlates every result by `task_id`.

## Explicit joint trajectories

The gateway supports both MoveIt's `MoveGroup` action and ros2_control's
`control_msgs/action/FollowJointTrajectory`. Set these `ExecuteRobotTask` goal fields
to select the latter:

- `use_joint_trajectory=true`
- `joint_names`
- `pickup_joint_positions`
- `placement_joint_positions`
- `trajectory_duration_s`

The gateway rejects inconsistent vector sizes or nonpositive durations. Navigation,
gripper, Isaac grasp/release, settle, disposal validation, and radiation-scene commit
remain identical for MoveIt and explicit trajectory execution.
