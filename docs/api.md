# Core API

The stable package boundaries are `radcounter.core.models`, `radiation`,
`sensors`, `actions`, `estimation`, `planning`, `workflow`, and `experiments`.
All physical values use unit-bearing field names. Estimation, residual,
non-oracle planning, and workflow modules have no `TruthState` dependency.

ROS 2 messages are generated from `ros2_ws/src/radcounter_msgs`. Isaac Sim and
ROS adapters are optional imports and must fail explicitly when their runtime
is absent.
