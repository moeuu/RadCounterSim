# RadCounterSim

RadCounterSim is a closed-loop radiation measurement and countermeasure
simulation platform. It separates simulator-only truth from the state available
to estimators and planners, then executes the cycle

`MEASURE -> ESTIMATE -> PLAN -> PREDICT -> EXECUTE -> VERIFY -> DIAGNOSE -> UPDATE`.

The repository has three layers:

- `radcounter.core`: Isaac Sim-independent models, radiation calculations,
  configuration, logging, estimation, planning, and workflow code.
- `radcounter.radiation.native`: C++17/Embree/pybind11 transport backend.
- `radcounter.isaac`: Isaac Sim 6.0.1 integration, USD, UI, robots, and optional
  ROS 2 bridge.

## Python environment

Python dependencies are managed only with [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
uv run radcounter-validate configs/scenarios/analytic_free_space.yaml
uv run radcounter-headless configs/scenarios/analytic_free_space.yaml
uv run radcounter-experiments --case analytic_radiation_validation --seed 42
uv run pytest
```

Do not install project dependencies with `pip` into the system interpreter.

The experiment command writes the required manifest, resolved configuration,
JSONL events, Parquet tables, metrics, NPZ maps, snapshots directory, and HTML
report under `outputs/<scenario>/<timestamp>_<run_id>/`.

## External runtimes

The target runtime is Ubuntu 24.04, Isaac Sim 6.0.1, Embree 4, and ROS 2 Jazzy.
Those runtimes are optional for the pure Python core. This development host uses:

- Isaac Sim 6.0.1.0: `~/.local/isaacsim/6.0.1-uv` (dedicated `uv.lock`)
- Embree 4.3.0: `~/.local/embree/4.3.0/usr`
- ROS 2 Jazzy 2026-06-18: `~/.local/ros2/jazzy`

```bash
source scripts/host_env.sh
./scripts/build_native.sh
./scripts/build_ros2.sh
uv run python scripts/audit_host_gates.py --require-all
```

Isaac Sim requires the user to review and accept NVIDIA's Omniverse EULA. The
launch script never accepts it implicitly. After acceptance, launch with
`OMNI_KIT_ACCEPT_EULA=YES ./scripts/run_isaac.sh`.

## Safety boundary

Estimator and planner APIs accept `BeliefState` and public observations only.
`TruthState` is owned by the simulator/action execution boundary and must never
be passed to those APIs. Countermeasure effectiveness is learned from a new
measurement, not by reading the truth delta.

## Status

Milestone implementation status is tracked in `docs/CHANGELOG.md`. A file or
interface scaffold is not evidence that its milestone acceptance criteria pass.
