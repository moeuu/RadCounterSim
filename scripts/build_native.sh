#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_directory="${RADCOUNTER_NATIVE_BUILD_DIR:-$repository_root/build/native}"
export RADCOUNTER_HOST_ENV_NO_ROS=1
source "$repository_root/scripts/host_env.sh"

pybind11_directory="$(
  uv run --group native python -m pybind11 --cmakedir
)"

uv run --group native cmake \
  -S "$repository_root/native" \
  -B "$build_directory" \
  -G Ninja \
  -Dpybind11_DIR="$pybind11_directory" \
  -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
  -DCMAKE_BUILD_TYPE=Release
uv run --group native cmake --build "$build_directory"

printf 'Native module: %s/python\n' "$build_directory"
