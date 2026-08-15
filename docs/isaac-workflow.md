# Isaac workflow integration

`radcounter.isaac.workflow.IsaacWorkflowServices` is the live-stage adapter for
`radcounter.core.workflow.ClosedLoopCoordinator`. It executes the following path:

1. Integrate detectors in the live USD/Embree scene.
2. Strip `expected_rate_cps` and every other Truth-only field.
3. Invoke an injected estimator with `tuple[PublicMeasurement, ...]`.
4. Generate candidates from the live USD scene and the returned `BeliefState`.
5. Preview the selected action from belief strengths and public geometry only.
6. Execute manufacturer-asset workflows through `RidgebackFrankaController`
   and `NovaCarterController`; the generic rigid-body controller remains
   available only for portable/custom-robot integrations.
7. Synchronize the actual USD pose/activity state into Embree.
8. Re-measure, calculate a public residual, and invoke an injected belief updater.

The simulator deliberately does not instantiate or choose a source-estimation
algorithm. Applications inject the estimator, residual diagnosis, and belief updater.
This keeps simulator Truth inaccessible to estimator and planner APIs.

## Scene-derived candidate generation

`IsaacActionCandidateGenerator` scans live `rad:*` USD metadata and creates:

- mobile measurement station visits;
- contact decontamination actions;
- obstacle-aware shield placements;
- movable-object relocation and disposal actions.

Each candidate includes checks for mobile path clearance, the articulated base
pose, horizontal and vertical manipulator reach, pickup and placement grasp-frame
poses, placement collision, support stability, disposal-class compatibility,
robot availability, and mission resources. Object-root poses and grasp-frame
offsets are tracked separately, including their rotation when the mobile base
changes yaw. Dose terms are computed from `BeliefState.source_strength_bq`, public
source sample positions, and Embree transmission. Runtime source activity is never
read for planner metrics.

## Physics-step lifecycle

`IsaacPhysicsStepLoop` subscribes through
`omni.physx.get_physx_interface().subscribe_physics_step_events`. It is the supported
way to run `IsaacRos2CommandHost.update(dt)`, contact-treatment ticks, and periodic
USD-to-Embree synchronization on the Isaac physics thread. Releasing the subscription
or leaving its context manager stops all callbacks.

## Workflow UI artifact

Pass `artifact_path=artifacts/ui/latest_workflow.json` to
`IsaacWorkflowServices`. The adapter atomically writes public estimate, prediction,
selected action, verification, residual, and revision data. The Operations dashboard
can load this file or receive the same mapping through `set_workflow_view()`.

## Gate

Run the real-host gate with:

```bash
export OMNI_KIT_ACCEPT_EULA=YES RADCOUNTER_HOST_ENV_NO_ROS=1
source scripts/host_env.sh
uv run --project "$RADCOUNTER_ISAAC_ROOT" --locked python tests/isaac/workflow_gate.py
```

The gate requires a physical shield displacement, an actual PhysX callback, initial
and verification detector integrations, a planner decision, and no Truth field in the
injected estimator input.

Run the complete articulated GUI path headlessly with:

```bash
uv run --project "$RADCOUNTER_ISAAC_ROOT" --locked python scripts/run_gui_validation.py \
  --headless --no-keep-open --phase-hold-s 0
```

That gate requires all eight operations and all final invariants to pass without
operation-time pose teleports.
