# Executable vertical slice

`assets/environments/radcounter_vertical_slice.usda` is the minimum end-to-end research scene. It contains a room, a triangle activity-map surface, one estimator-hidden movable point source, a measurement robot, a countermeasure robot, a lead shield, a movable steel obstacle, a decontamination tool, four validation stations, and a disposal zone.

The portable stage keeps lightweight robot placeholders so it can be inspected
without Isaac assets. `scripts/run_gui.py` replaces those placeholders before
physics starts with the official Ridgeback + Franka Panda and Nova Carter USDs,
then authors task tools and grasp frames under the real articulations.

Generate deterministic assets with:

```bash
uv run python scripts/create_vertical_slice_assets.py
```

Run a headless measurement with the uv-managed Isaac Sim environment:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
source scripts/host_env.sh
uv run --project "$RADCOUNTER_ISAAC_ROOT" --locked python scripts/run_vertical_slice.py --output artifacts/vertical_slice.json
```

The attenuation table in `configs/scenarios/vertical_slice.runtime.json` is marked `synthetic_validation_only`. It exercises software behavior and must not be used for radiation-safety decisions.
