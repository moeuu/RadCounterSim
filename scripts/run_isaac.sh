#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RADCOUNTER_HOST_ENV_NO_ROS=1
source "$repository_root/scripts/host_env.sh"

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
