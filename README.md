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
uv run radcounter-import-environment configs/environments/vertical_slice_import.yaml
uv run radcounter-experiments --case analytic_radiation_validation --seed 42
uv run pytest
```

Do not install project dependencies with `pip` into the system interpreter.

## External 3D environments

The Isaac window and import CLI accept USD, glTF/GLB, OBJ, STL, PLY, DAE,
3MF, SDF/Gazebo world, URDF/Xacro, STEP/IGES/BREP CAD, and PCD/XYZ point-cloud
environments. Geometry is normalized to metre, right-handed, Z-up coordinates
and shared by rendering, PhysX collision, and Embree attenuation. Proprietary
formats can use a no-shell external converter or an importer plugin. See
`docs/environment-import.md`; CAD tessellation uses the `cad` uv group.

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

## Large environments and arbitrary robots

Large sites can be preprocessed into spatial tiles and deterministic LOD payloads with `uv run radcounter-build-large-environment`. Isaac loads only a bounded working set around robot and detector focus points; radiation source-detector segments pin every intersected tile at LOD0. See `docs/large-environments.md`.

Robot fleets are described in YAML and may reference USD, URDF, Xacro, or MJCF assets. The Isaac adapter imports them into a content-addressed USD cache and exposes named position, velocity, effort, base-twist, gripper, and optional Lula IK commands through one controller. See `docs/generic-robots.md` and `configs/robots/fleet.example.yaml`.

## Robot input

The same robot can be controlled from WASD/QE keyboard input, a gamepad, the in-simulator control window, localhost command messages, or an autonomous callback. A TTL-based priority mux applies emergency and manual commands before autonomous commands and stops the base when an input expires. See `docs/robot-control.md` and use `uv run radcounter-robot-command` for external commands.

## Water decontamination

Pressure-water washing models nozzle flow, pressure, cone geometry, standoff, incidence, washability, clean-water inventory, wastewater capacity, activity recovery, runoff redeposition, and environmental discharge. The same Isaac scene supports `--method water` and displays both activity and surface wetness. See `docs/water-decontamination.md`.

## Detector arrays

Synchronized arrays can mix omnidirectional survey meters, spectrometers, dose meters, neutron counters, directional collimators, coded-aperture imagers, Compton cameras, and custom detector plugins. CSV/YAML response descriptors and external reading buffers make laboratory and hardware detectors connect through the same schema. See `docs/detector-integration.md`.
