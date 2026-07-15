# Scenario authoring

Scenario YAML is validated before Isaac Sim starts. Every physical field uses a
unit-bearing key. Relative file paths resolve from the scenario file directory.

The initial schema supports isotopes, point sources, one detector, measurement
poses, runtime seed, output directory, and explicit radiation/scatter backends.
Later milestones extend the same versioned schema for surfaces, robots,
actions, resources, and closed-loop termination.

Use:

```bash
uv run radcounter-validate configs/scenarios/analytic_free_space.yaml
```
