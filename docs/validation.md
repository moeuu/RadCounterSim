# Validation

## Pure Python core

```bash
uv sync --all-groups
uv run pytest
uv run radcounter-validate configs/scenarios/analytic_free_space.yaml
uv run radcounter-headless configs/scenarios/analytic_free_space.yaml
```

## Isaac Sim 6.0.1 gate

Isaac Sim was not available on the scaffold host. Before claiming Milestone 0
acceptance, generate the UI extension and C++ extension from the locally
installed 6.0.1 official templates, merge the RadCounterSim modules, record the
actual extension dependency names in an ADR, and run both GUI registration and
headless startup tests in that runtime.

## Native gate

Embree 4 and CMake were not available on the scaffold host. The native backend
must not be reported complete until slab, cube, nested solid, thin sheet,
dynamic instance, race, leak, and batch benchmark evidence is captured.
