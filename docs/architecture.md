# Architecture

## Dependency direction

`radcounter.core` has no Isaac Sim, USD, ROS 2, or Embree import. Native and
Isaac adapters depend on core contracts; core never depends on either adapter.

Truth and belief are separate object graphs. The action executor receives
`TruthState`, while estimation and planning receive `BeliefState`, measurements,
public geometry, public robot state, and action completion notifications. Only
the experiment logger may persist `ActionResult.truth_details`.

Radiation is event-driven. Geometry, material, source pose, source activity,
and detector revisions are tracked independently. Source activity is omitted
from transfer-matrix cache keys so decontamination can update `H @ x` without
ray tracing. Geometry-changing actions invalidate geometry-dependent transport.

## External adapters

Isaac Sim, Embree, and ROS 2 adapters are optional. Missing runtimes must cause
an explicit availability error, never a silent physics fallback. The analytic
backend is an explicit configured backend intended for CI and validation.
