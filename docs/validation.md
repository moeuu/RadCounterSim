# Validation

## Pure Python core

```bash
uv sync --all-groups
uv lock --check
uv run ruff check .
uv run python -m pytest -q
uv run radcounter-validate configs/scenarios/analytic_free_space.yaml
uv run radcounter-headless configs/scenarios/analytic_free_space.yaml
```

## Isaac Sim 6.0.1 gate

The host-gate script builds native transport, runs the portable and native
tests, executes the Isaac/PhysX component gates (including dynamic geometry and
the physical rotating shield), runs the full articulated scenario, and then
checks the ROS 2 adapters:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
scripts/run_host_gates.sh
```

For the rotating-shield component alone:

```bash
export OMNI_KIT_ACCEPT_EULA=YES RADCOUNTER_HOST_ENV_NO_ROS=1
source scripts/host_env.sh
export PYTHONPATH="$PWD/build/native/python:$PWD/source/extensions/radcounter.isaac:$PWD${PYTHONPATH:+:$PYTHONPATH}"
uv run --project "$RADCOUNTER_ISAAC_ROOT" --locked python \
  tests/isaac/rotating_shield_gate.py
```

That gate is a direct USD component-posture experiment and records
`kinematic_scene_edit`. The articulated scenario records
`physical_robot_execution`; the two evidence classes are not interchangeable.

## Native gate

```bash
export RADCOUNTER_HOST_ENV_NO_ROS=1
source scripts/host_env.sh
PYTHONPATH="$PWD/build/native/python:$PWD" \
  uv run python -m pytest -q tests/integration/test_embree_runtime.py
```

Software and synthetic-operation gates do not establish detector, material, or
treatment accuracy. The research-data and controlled-measurement requirements
are listed in `paper-simulator-scope.md`.
