#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

seed="${1:-42}"
run_root="${RADCOUNTER_RUN_ROOT:-runs}"

uv sync --locked
uv run radcounter-experiments \
  --case closed_loop_vs_open_loop \
  --seed "$seed" \
  --run-root "$run_root"
