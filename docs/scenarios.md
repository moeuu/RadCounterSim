# Scenario authoring

Scenario YAML is validated before Isaac Sim starts. Every physical field uses a
unit-bearing key. Relative file paths resolve from the scenario file directory.

The schema supports an optional simulator-neutral `environment` object in
addition to isotopes, point sources, detectors, measurement poses, runtime
seed, output directory, and radiation/scatter backends. The environment object
is the same contract accepted by `radcounter-import-environment`; see
`docs/environment-import.md`.

Use:

```bash
uv run radcounter-validate configs/scenarios/analytic_free_space.yaml
```
