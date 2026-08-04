#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RADCOUNTER_HOST_ENV_NO_ROS=1
source "$repository_root/scripts/host_env.sh"

isaac_ros_lib="$RADCOUNTER_ISAAC_ROOT/.venv/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/jazzy/lib"
export ROS_DISTRO="${ROS_DISTRO:-jazzy}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$isaac_ros_lib"

eula_marker="$RADCOUNTER_ISAAC_ROOT/.venv/lib/python3.12/site-packages/isaacsim/kit/EULA_ACCEPTED"
if [[ "${OMNI_KIT_ACCEPT_EULA:-}" != "YES" && ! -f "$eula_marker" ]]; then
  cat >&2 <<'TEXT'
Isaac Sim requires acceptance of the NVIDIA Omniverse EULA.
Review: https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html
After accepting, run with OMNI_KIT_ACCEPT_EULA=YES.
TEXT
  exit 2
fi

experience="${RADCOUNTER_ISAAC_EXPERIENCE:-isaacsim.exp.full}"
exec uv run --project "$RADCOUNTER_ISAAC_ROOT" --locked \
  isaacsim "$experience" \
  --ext-folder "$repository_root/source/extensions" \
  "$@"
