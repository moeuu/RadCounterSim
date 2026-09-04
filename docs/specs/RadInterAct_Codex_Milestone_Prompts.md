# RadInterAct — Staged Codex Implementation Prompts

## How to use this document

Do not ask Codex to implement every feature at once. Submit the prompts below one
milestone at a time, in order. At each milestone, first inspect the existing code and
the API of the targeted Isaac Sim version, record design changes in
`docs/decisions/ADR-XXXX.md`, and proceed only after the tests pass.

Instructions common to every milestone:

```text
- Treat RadInterAct_Codex_Implementation_Spec.md as the highest-level specification.
- Do not modify unrelated existing code.
- Add type annotations and docstrings to public APIs.
- Include units in variable names.
- Preserve the dependency boundary between TruthState and BeliefState.
- Before implementation, list the files to be changed and provide a test plan.
- After implementation, report the commands run, passing tests, and unresolved items.
- Do not declare completion with mocks alone. Build the unit-testable, dependency-free
  core first where appropriate.
```

---

## Prompt 0 — Repository and extension scaffold

```text
Implement Milestone 0 of RadInterAct_Codex_Implementation_Spec.md.

Requirements:
1. Use the UI extension and C++ extension templates from an Isaac Sim 6.0.1 source workspace.
2. Create the radcounter.core, radcounter.radiation.native, and radcounter.isaac layers.
3. Register the UI extension in the Examples Browser and make Load, Reset, and Clear work.
4. Provide execution paths for a headless startup test and pure-Python unit tests.
5. Create a YAML scenario schema, Pydantic configuration model, and validate_scenario.py.
6. Create skeletons for the run manifest and JSONL event logger.
7. Document CI commands and local setup in the README.
8. Verify Isaac Sim extension dependency names against local 6.0.1 official templates or
   examples; do not guess them.

Acceptance criteria:
- The extension startup test passes.
- A non-empty pytest suite passes.
- The sample scenario validates.
- The RadInterAct example appears in the GUI.
```

## Prompt 1 — Core models and analytic radiation backend

```text
Implement Milestone 1.

Implementation scope:
- SourceType, EmissionLine, IsotopeSpec, PointSourceState, SurfaceSourceState
- MaterialSpec, DetectorSpec, RadiationMeasurement, RevisionState
- MaterialTable energy interpolation
- Point-source inverse-square forward model
- NoScatterModel
- AnalyticTransportBackend with no attenuation and a single slab
- Detector efficiency, background, dead time, and Poisson sampling
- Deterministic RNG hierarchy

Tests:
- 1/r^2
- exp(-mu*l)
- Energy interpolation
- Poisson reproducibility
- Zero or disabled source
- Unit validation

Acceptance criteria:
- All mathematical unit tests pass.
- The units for source, length, activity, count, and dose are documented.
```

## Prompt 2 — USD radiation metadata and mesh extraction

```text
Implement Milestone 2.

Implementation scope:
- Helpers for rad:* custom attributes
- UsdRadiationRegistry
- SceneDescriptor
- UsdGeom.Mesh extraction
- Triangulation, metersPerUnit, world transforms, instances, and negative/non-uniform scale
- Per-triangle material IDs
- Transport-mesh selection
- USD change notices and revision updates
- URI and checksum for per-face activity sidecars

Create a demo USD containing:
- A room
- A concrete wall
- A thin shield panel
- A contaminated floor
- A movable contaminated box
- A detector mount

Tests:
- Demo-stage scan
- Descriptor counts
- Triangles, areas, and transforms
- Revision classification
- Change notification without a full rescan
```

## Prompt 3 — Embree C++ backend

```text
Implement Milestone 3.

Implementation scope:
- EmbreeTransportScene C++ class
- pybind11 bindings
- Triangle-mesh registration
- Static geometry and dynamic instances
- Transform update, removal, and commit
- Finite segment rays
- Collection of all intersections through repeated closest-hit queries
- Solid entry/exit path length
- Thin-sheet effective thickness
- Per-material path lengths
- Per-energy transmission
- GIL release, parallel batch queries, and thread safety
- Explicit error handling

Test geometry:
- Slab
- Cube
- Nested cubes
- Two materials
- Thin sheet at normal and oblique incidence
- Moving instance
- Invalid odd-hit mesh

Benchmark:
- 1k, 10k, and 100k rays
- Save results as JSON.

Acceptance criteria:
- Analytic tests pass.
- No memory leak is detected.
- Commit/trace race tests pass.
```

## Prompt 4 — Surface sources, sensors, dose maps, and cache

```text
Implement the radiation portions of Milestones 4 and 5.

Implementation scope:
- SourceSampleBatch
- Point and surface quadrature: centroid, stratified, and adaptive
- Transform updates for attached sources
- RadiationForwardModel
- OmnidirectionalCounter, RotatingShieldCounter, and DoseRateMeter
- Measurement state machine
- Moving-integration trajectory sampling
- Transfer matrix H
- TransferMatrixCache and revision rules
- DoseMapEvaluator
- Chunked processing

Important:
- Decontamination activity changes must not invalidate the ray-trace cache.
- Shield geometry changes must invalidate geometry-dependent cache entries.
- Keep truth scatter/bias separate from the planner model.

Tests:
- Surface-rectangle convergence
- Sensor on a moving prim
- Cache hits and misses
- Activity-only fast update
- Rotating-shield physical-geometry mode
```

## Prompt 5 — UI and visualization

```text
Implement the UI and visualization portions of Milestone 5.

Frames:
- Scenario
- Radiation Scene
- Measurement
- Estimation placeholder
- Countermeasure placeholder
- Closed Loop placeholder
- Visualization
- Experiment

Visualization:
- 2D dose heatmap
- Truth-source debug overlay
- Belief-source overlay placeholder
- Selected ray path and material lengths
- Revision and timing display

Requirements:
- Run heavy computation in asynchronous tasks.
- Provide cancellation tokens.
- Discard results from stale revisions.
- Never perform blocking traces in UI callbacks.
```

## Prompt 6 — Deterministic countermeasure actions

```text
Implement Milestone 6.

Implementation scope:
- CountermeasureAction, ActionResult, and ActionType
- Action lifecycle and state machine
- ResourceState and consumption
- DeterministicRobotController
- DecontaminationExecutor
- ShieldPlacementExecutor
- MoveObjectExecutor
- RemoveObjectExecutor
- DisposalZone
- Truth-side action uncertainty
- Separation of public_details and truth_details

Decontamination:
- Triangle activity map
- Footprint path
- Exposure model
- Spatial efficiency random field
- discard and transfer_to_waste modes

Shielding:
- Shield asset spawn and movement
- Actual pose error
- Embree update

Movement and removal:
- Attached-source following
- Disposal validation

Integration tests must verify measurement and map changes before and after every action.
```

## Prompt 7 — Physics-based robot execution

```text
Implement Milestone 7.

Implementation scope:
- IsaacPhysicsRobotController
- Mobile measurement-robot navigation
- Countermeasure mobile manipulator, or composed base and arm
- End-effector planning
- Grasp/release constraints
- Shield pick-and-place
- Obstacle picking or pushing
- Contaminated-object movement
- Decontamination-tool footprint integration from rays or contact
- Settle detection
- Timeout, abort, and recovery

Requirements:
- Use the same API as deterministic actions.
- Reflect the actual USD pose in the radiation registry.
- Record physics-execution failures in the action result.
- Update radiation state, not only the manipulation animation.

Minimum demo:
1. Grasp a shield panel and place it at the target pose.
2. Move a contaminated box.
3. Move a non-contaminated obstacle out of the way.
4. Treat a specified patch with the decontamination tool.
```

## Prompt 8 — Source estimation

```text
Implement Milestone 8.

Implementation scope:
- CandidateBasis: 3D grid and surface-triangle graph
- Measurement stacking
- Poisson negative log likelihood
- Nonnegative MLE
- L1 proximal solver
- Surface total-variation regularization
- Coarse grid to connected components to continuous MLE refinement
- Fisher uncertainty
- Optional bootstrap
- SourceEstimate serialization and visualization

Prohibited:
- Reading TruthState
- Supplying the ground-truth source count to the solver

Tests:
- Noiseless and noisy single source
- Two sources
- Hidden surface patch
- Source-height variation
- Finite-difference gradient
- Uncertainty shape
```

## Prompt 9 — Predicted/observed residuals and re-estimation

```text
Implement Milestone 9.

Implementation scope:
- Nominal action preview on a BeliefState clone
- Predicted verification measurement
- Observed verification measurement
- Raw and normalized residuals
- DeconResidualHypothesis
- ShieldPoseErrorHypothesis
- HiddenSourceHypothesis
- GlobalGainBackgroundHypothesis
- SourceLocalizationErrorHypothesis
- Likelihood/BIC selection
- Belief updates
- Action-effect parameter updates
- Residual visualization and logs

Fault-injection tests:
- 30% decontamination residual
- 5 cm shield translation error
- Hidden source
- Detector gain bias
- Mixed failure case
```

## Prompt 10 — Planner and closed-loop coordinator

```text
Implement Milestone 10.

Implementation scope:
- Action candidate generators
- Measurement-pose candidates
- Decontamination-region candidates
- Shield-placement candidates
- Object movement/removal candidates
- Repair candidates
- Navigation, manipulation, and resource feasibility
- Action objective
- Approximate expected information gain
- OpenLoop, Greedy, Nearest, Random, Oracle, and ClosedLoopResidual planners
- ClosedLoopCoordinator state machine
- Pause, resume, and stop
- Termination conditions
- Snapshot persistence

End-to-end demo:
MEASURE -> ESTIMATE -> PLAN -> PREDICT -> EXECUTE -> VERIFY -> DIAGNOSE -> UPDATE -> REPLAN

Requirements:
- Only Oracle may read TruthState.
- Run baselines and the proposed planner with the same seed batch runner.
```

## Prompt 11 — ROS 2, MoveIt 2, and Nav2 adapters

```text
Implement the ROS 2 portion of Milestone 11.

Implementation scope:
- radcounter_msgs
- RadiationMeasurement, SourceEstimate, and CountermeasureStatus
- MeasureRadiation.action
- ExecuteCountermeasure.action
- GetDoseMap and EvaluateCountermeasure services
- Ros2RobotController
- Standard tf, joint_states, cmd_vel, and trajectory integration
- MoveIt 2 manipulation adapter
- Nav2 navigation adapter
- Namespace and multi-robot support

Requirements:
- ROS 2 must remain an optional dependency; the core and GUI work without it.
- Target Jazzy primarily.
- Use simulated time.
- Configure timeouts and QoS in configuration files.
```

## Prompt 12 — Experiment automation, performance, documentation, and release

```text
Implement the final milestone.

Implementation scope:
- Headless batch runner
- Seed sweep
- Baseline sweep
- Parquet, JSON, and NPZ output
- HTML report
- Git, configuration, asset, and hardware manifest
- Benchmark suite
- Regression baselines
- API documentation
- Scenario authoring guide
- Troubleshooting guide
- One-command demo scripts

Generate these experiments:
1. Analytic radiation validation
2. Decontamination primitive
3. Shielding primitive
4. Movable contaminated object
5. Hidden-source residual
6. Closed-loop versus open-loop
7. Resource-constrained multi-action workflow

Final checks:
- Convert every Definition of Done item into a checklist.
- Identify every unimplemented item.
- Create a changelog suitable for a versioned release tag.
```
