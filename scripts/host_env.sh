#!/usr/bin/env bash
# RadCounterSim host paths. Source this file from Bash.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  printf '%s\n' "source scripts/host_env.sh instead of executing it" >&2
  exit 2
fi

export RADCOUNTER_ISAAC_ROOT="${RADCOUNTER_ISAAC_ROOT:-$HOME/.local/isaacsim/6.0.1-uv}"
export ISAAC_SIM_PATH="$RADCOUNTER_ISAAC_ROOT"
export RADCOUNTER_EMBREE_ROOT="${RADCOUNTER_EMBREE_ROOT:-$HOME/.local/embree/4.3.0/usr}"
export EMBREE_ROOT="$RADCOUNTER_EMBREE_ROOT"
export RADCOUNTER_ROS2_ROOT="${RADCOUNTER_ROS2_ROOT:-$HOME/.local/ros2/jazzy}"
export RADCOUNTER_ROS2_PYTHON_ROOT="${RADCOUNTER_ROS2_PYTHON_ROOT:-$HOME/.local/ros2/python-runtime}"

export CMAKE_PREFIX_PATH="$EMBREE_ROOT${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$EMBREE_ROOT/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="$EMBREE_ROOT/lib/x86_64-linux-gnu/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export PATH="$HOME/.local/bin:$PATH"

if [[ "${RADCOUNTER_HOST_ENV_NO_ROS:-0}" != "1" ]]; then
  # ROS setup scripts are not nounset-safe. Preserve the caller's shell mode.
  _radcounter_restore_nounset=0
  case "$-" in
    *u*) _radcounter_restore_nounset=1; set +u ;;
  esac
  source "$HOME/.local/ros2/jazzy/setup.bash"
  if [[ "$_radcounter_restore_nounset" == "1" ]]; then set -u; fi
  unset _radcounter_restore_nounset
  export LD_LIBRARY_PATH="$HOME/.local/ros2/deps/noble/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export PATH="$HOME/.local/ros2/bin:$HOME/.local/bin:$PATH"
  export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
fi
