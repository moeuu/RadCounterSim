#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repository_root/scripts/host_env.sh"

ros_workspace="$repository_root/ros2_ws"
cmake_cache="$ros_workspace/build/radcounter_msgs/CMakeCache.txt"
if [[ -f "$cmake_cache" ]]; then
  cached_source="$(sed -n 's|^CMAKE_HOME_DIRECTORY:INTERNAL=||p' "$cmake_cache")"
  expected_source="$ros_workspace/src/radcounter_msgs"
  if [[ -n "$cached_source" && "$cached_source" != "$expected_source" ]]; then
    stale_build="$(mktemp -d "${TMPDIR:-/tmp}/radcounter-ros2-stale.XXXXXX")"
    printf 'Moving stale ROS build products to %s\n' "$stale_build"
    for generated_dir in build install log; do
      if [[ -e "$ros_workspace/$generated_dir" ]]; then
        mv "$ros_workspace/$generated_dir" "$stale_build/$generated_dir"
      fi
    done
  fi
fi

cd "$ros_workspace"
colcon build \
  --symlink-install \
  --cmake-clean-cache \
  --event-handlers console_direct+ \
  --cmake-args \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$RADCOUNTER_ROS2_PYTHON_ROOT/.venv/bin/python"

printf 'ROS workspace: %s/install\n' "$repository_root/ros2_ws"
