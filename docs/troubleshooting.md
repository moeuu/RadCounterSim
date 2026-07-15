# Troubleshooting

## Python dependency mismatch

Run `uv sync --all-groups --frozen`. Do not repair the environment with system
`pip`.

## ROS 2 imports are unavailable

The core does not require ROS 2. For adapters, source a ROS 2 Jazzy workspace
where `radcounter_msgs` has been built with `colcon build`.

## Isaac or Embree is unavailable

Use the analytic backend for core validation. Do not claim native/physics
acceptance until the external-runtime gates in `docs/validation.md` pass.
