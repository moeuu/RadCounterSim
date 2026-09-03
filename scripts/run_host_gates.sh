#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

scripts/build_native.sh
export LD_LIBRARY_PATH="/home/moeu/.local/embree/4.3.0/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT:$ROOT/build/native/python${PYTHONPATH:+:$PYTHONPATH}"
uv run python -m pytest tests/unit tests/integration tests/regression
uv run ruff check .

(
  export OMNI_KIT_ACCEPT_EULA=YES RADCOUNTER_HOST_ENV_NO_ROS=1
  source scripts/host_env.sh
  for gate in \
    extension_load_gate.py \
    vertical_slice_gate.py \
    runtime_geometry_gate.py \
    rotating_shield_gate.py \
    performance_gate.py \
    visualization_gate.py \
    physics_actions_gate.py \
    articulation_ik_gate.py \
    real_robot_gate.py \
    articulated_object_gate.py \
    dashboard_gate.py \
    workflow_gate.py
  do
    uv run --project "$RADCOUNTER_ISAAC_ROOT" --locked python "tests/isaac/$gate"
  done
  uv run --project "$RADCOUNTER_ISAAC_ROOT" --locked \
    python scripts/run_gui_validation.py \
    --headless --no-keep-open --phase-hold-s 0 \
    --artifact artifacts/gui-validation/host-gate.json
)

(
  scripts/build_ros2.sh
  source scripts/host_env.sh
  cd ros2_ws
  export COLCON_TRACE="${COLCON_TRACE:-}"
  set +u
  source install/setup.bash
  set -u
  cd "$ROOT"
  uv run --project /home/moeu/.local/ros2/python-runtime --locked \
    python tests/ros/ros_motion_gateway_gate.py
)

(
  export OMNI_KIT_ACCEPT_EULA=YES ROS_DOMAIN_ID=87
  source scripts/host_env.sh
  export COLCON_TRACE="${COLCON_TRACE:-}"
  set +u
  source ros2_ws/install/setup.bash
  set -u
  uv run --project "$RADCOUNTER_ISAAC_ROOT" --locked \
    python tests/isaac/ros_command_host_gate.py
)
