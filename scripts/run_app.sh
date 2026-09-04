#!/usr/bin/env bash
set -euo pipefail

repository_root="${RADCOUNTER_APP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export RADCOUNTER_HOST_ENV_NO_ROS=1
source "$repository_root/scripts/host_env.sh"

if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' "RadInterAct requires uv: https://docs.astral.sh/uv/" >&2
  exit 2
fi
if [[ ! -f "$RADCOUNTER_ISAAC_ROOT/uv.lock" ]]; then
  printf '%s\n' "Isaac Sim 6.0.1 was not found at: $RADCOUNTER_ISAAC_ROOT" >&2
  printf '%s\n' "Install your own Isaac Sim copy or set RADCOUNTER_ISAAC_ROOT." >&2
  exit 2
fi

eula_marker="$RADCOUNTER_ISAAC_ROOT/.venv/lib/python3.12/site-packages/isaacsim/kit/EULA_ACCEPTED"
if [[ "${OMNI_KIT_ACCEPT_EULA:-}" != "YES" && ! -f "$eula_marker" ]]; then
  printf '%s\n' "Review and accept NVIDIA's Omniverse EULA before first launch." >&2
  printf '%s\n' "Then launch once with OMNI_KIT_ACCEPT_EULA=YES." >&2
  exit 2
fi

runtime_root="${RADCOUNTER_LLM_RUNTIME_DIR:-$repository_root/runtime/llm}"
model_path="${RADCOUNTER_LLM_MODEL:-$runtime_root/models/Qwen3-4B-Q4_K_M.gguf}"
llm_pid_file="$runtime_root/logs/llama-server-launcher-$$.pid"
export RADCOUNTER_LLM_PID_FILE="$llm_pid_file"
if [[ -z "${RADCOUNTER_LLM_ENDPOINT:-}" ]]; then
  if [[ ! -f "$model_path" ]]; then
    printf '%s\n' "The local command model is not installed: $model_path" >&2
    printf '%s\n' "Run: uv run python scripts/fetch_llm_model.py" >&2
    exit 2
  fi
  if [[ ! -x "$runtime_root/bin/linux-x86_64-cpu/llama-server" && \
        ! -x "$runtime_root/bin/linux-x86_64-vulkan/llama-server" && \
        ! -x "$runtime_root/bin/linux-x86_64-cuda/llama-server" && \
        -z "${RADCOUNTER_LLAMA_SERVER:-}" ]]; then
    printf '%s\n' "The bundled llama.cpp runtime is not installed." >&2
    printf '%s\n' "Run: ./scripts/build_llama_runtime.sh" >&2
    exit 2
  fi
fi

application_pid=""
cleanup_application() {
  trap - INT TERM HUP
  if [[ -n "$application_pid" ]] && kill -0 "$application_pid" 2>/dev/null; then
    kill -TERM "$application_pid" 2>/dev/null || true
  fi
  if [[ -f "$llm_pid_file" ]]; then
    read -r llm_pid < "$llm_pid_file" || true
    if [[ "$llm_pid" =~ ^[0-9]+$ && -r "/proc/$llm_pid/cmdline" ]]; then
      llm_command="$(tr '\0' ' ' < "/proc/$llm_pid/cmdline")"
      if [[ "$llm_command" == *llama-server* ]]; then
        kill -TERM "$llm_pid" 2>/dev/null || true
      fi
    fi
    rm -f "$llm_pid_file"
  fi
}
trap cleanup_application EXIT INT TERM HUP

uv run --project "$RADCOUNTER_ISAAC_ROOT" --locked \
  python "$repository_root/scripts/run_gui.py" "$@" &
application_pid="$!"
wait "$application_pid"
application_status="$?"
application_pid=""
exit "$application_status"
