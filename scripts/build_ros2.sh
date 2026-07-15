#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repository_root/scripts/host_env.sh"

cd "$repository_root/ros2_ws"
colcon build \
  --symlink-install \
  --event-handlers console_direct+ \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$RADCOUNTER_ROS2_PYTHON_ROOT/.venv/bin/python"

printf 'ROS workspace: %s/install\n' "$repository_root/ros2_ws"
