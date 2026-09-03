# RadInterAct

RadInterAct is a scene-consistent platform for evaluating radiological
measurement and robot-executed countermeasures. Rendering, physical interaction,
radiation paths, source state, and detector observations share one mutable USD
scene while simulator-only state remains inaccessible to estimators and planners.

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
uv run python -m pytest
```

Do not install project dependencies with `pip` into the system interpreter.

## External 3D environments

The Isaac window and import CLI accept USD, glTF/GLB, OBJ, STL, PLY, DAE,
3MF, SDF/Gazebo world, URDF/Xacro, STEP/IGES/BREP CAD, and PCD/XYZ point-cloud
environments. Geometry is normalized to metre, right-handed, Z-up coordinates
and shared by rendering, PhysX collision, and Embree attenuation. Proprietary
formats can use a no-shell external converter or an importer plugin. See
`docs/environment-import.md`; CAD tessellation uses the `cad` uv group.

Complete runtime compositions are selected independently from a system catalog:

```bash
uv run radcounter-system list
uv run radcounter-system activate --profile fukushima-packbot
radcounter-app
uv run radcounter-system activate --profile vertical-slice
```

The Operations window exposes the same catalog as ordinary preset,
environment, robot, and detector selectors, so no command or file path is
required for normal switching. The LLM instruction area remains a separate
robot-task control. The Fukushima Daiichi profile fetches the pinned CC BY 4.0
SolidWorks source and converts it directly to USD on Linux with Isaac Sim's
bundled HOOPS converter; see `docs/system-profiles.md`.

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

## Articulated GUI workflow

The GUI workflow loads NVIDIA's Clearpath Ridgeback + Franka Panda and Nova
Carter assets. It drives real articulation and wheel joints, solves the
seven-axis arm with Lula IK, closes the physical gripper before attaching a
payload constraint, and performs contact-driven decontamination. Shield
placement/correction, contaminated-drum relocation/disposal, and obstacle
relocation all use the same base-arm-gripper sequence. Direct USD pose edits are
available for separately labeled `kinematic_scene_edit` experiments, but are
never accepted as `physical_robot_execution` evidence.

```bash
export OMNI_KIT_ACCEPT_EULA=YES RADCOUNTER_HOST_ENV_NO_ROS=1
source scripts/host_env.sh
uv run --project "$RADCOUNTER_ISAAC_ROOT" --locked python scripts/run_gui.py
```

Use `--headless --no-keep-open --phase-hold-s 0` for a noninteractive gate. The
complete public audit is written to `artifacts/gui-validation/latest.json`.
The visible GUI is capped at 60 FPS by default to avoid consuming a full GPU
while idle. Override it with `--max-fps 30`, or use `--max-fps 0` to remove the
cap.

The viewport keeps one active robot explicit in a top status bar and provides
one-click follow/onboard views, a 12 FPS building overview, through-wall robot
beacons, routes, targets, measurement locations, and contact-derived
decontamination progress. Only the main viewport is rendered; selecting an
onboard view switches that viewport instead of rendering every robot camera.
See `docs/robot-monitoring.md`.

## Local natural-language application

The interactive application accepts English instructions, maps
them to a strict allowlist, previews physical operations, and executes them
through the existing workflow boundary. Release builds own a bundled
`llama.cpp` sidecar and an official Qwen3-4B GGUF model; users do not install
Ollama, PyTorch, or a Python inference SDK.

Complex instructions can sequence up to 24 logical steps, tour every feasible
measurement station, run bounded multi-pass irregular-surface decontamination
against public removal/remaining/coverage conditions, and place then reposition
physical shield panels at host-derived source-line fractions. Every physical
attempt is revalidated against the live scene and remains confirmation-gated.

```bash
./scripts/build_llama_runtime.sh
uv run python scripts/fetch_llm_model.py
OMNI_KIT_ACCEPT_EULA=YES ./scripts/run_app.sh
```

Isaac Sim remains a user-installed prerequisite and is never redistributed by
the OSS package. After initial setup, `radcounter-app` or the optional desktop
entry launches the simulator and private local model as one application. See
`docs/natural-language-control.md` for runtime layout, safety policy, hardware
fallbacks, and packaging details.

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

The rotating-shield detector has two explicit modes. The physical mode in
`configs/detectors/rotating_shield_counter.physical.synthetic.yaml` rotates
material-tagged USD shielding, synchronizes Embree, and evaluates the common
detector-response path at every actual posture. The response-mask configuration
is retained only as a separately identified approximation. The physical-mode
component gate is `tests/isaac/rotating_shield_gate.py` and reports
`kinematic_scene_edit` evidence, never robot-execution evidence.
