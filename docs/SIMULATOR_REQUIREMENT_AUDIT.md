# RadCounterSim simulator requirement audit

Audit date: 2026-08-06

## Scope and verdict

The simulator's core architecture, radiation transport, countermeasure actions,
closed-loop workflow, ROS 2 adapters, and articulated Isaac execution are
implemented. The implementation specification is nevertheless **not literally
100% complete**.

The remaining required gaps are continuous point-source position/activity MLE
refinement and the complete eight-section operational UI/visualization surface.
The current dashboard loads the articulated scene, synchronizes the radiation
runtime, authors explicitly authorized sources, measures detectors, renders a
dose proxy, and displays externally supplied estimate/residual/plan artifacts.
It does not yet run estimation, action preview/execution, closed-loop control, or
experiment configuration directly from the GUI, and it lacks the specified
source-uncertainty, residual-map, selected-ray, and action overlays. The exact
performance-target matrix in section 29.3 also has not been recorded; the live
performance gate validates cache behavior on a smaller workload with a looser
limit.

`PFPlusMLEEstimator` remains an optional-phase item in the specification and is
not implemented. Dependency locking uses `uv.lock` rather than the spec's
suggested `requirements-lock.txt`.

## Requirement matrix

| Requirement | Status | Implementation and executable evidence |
| --- | --- | --- |
| Independent core/native/Isaac layering | Complete | `radcounter/core`, `native`, and `source/extensions/radcounter.isaac`; extension-load gate |
| Truth/Belief separation | Complete | `TruthState`, `BeliefState`, estimator-visible USD registry, `PublicMeasurement`; non-oracle planner truth-isolation tests |
| Point/surface/volume sources | Complete | Canonical `rad:*` metadata, USD registry, runtime sampling, dashboard volume-source gate |
| Surface-activity NPZ sidecar | Complete | Triangle mapping, cumulative exposure, last-treated step, SHA-256 validation, atomic replacement |
| Dynamic Embree transport | Complete | Finite segments, material path lengths, solid and thin-sheet modes, grazing clamp, real Embree instances, transform/remove/commit, 8-wide packet tracing and tail handling |
| Revision and transfer-map caching | Complete | Geometry/material/source-pose/source-activity/detector revisions; activity-only zero-ray update; finite-AABB selective ray patching |
| Detector models | Complete | Omnidirectional state machine, Poisson counts, background, dead time, dose conversion, rotating physical-geometry and response-mask modes |
| Physical decontamination | Complete | Tool-footprint ray contacts, distance/normal/speed checks, exposure kinetics, correlated truth efficiency, waste transfer/discard, recontamination, activity-map persistence |
| Shield placement and correction | Complete | Full lifecycle, FixedJoint grasp/release, actual settled USD pose, pose error, Embree synchronization, verification measurement |
| Object move and removal | Complete | Local-frame attached sources, physical pick/place, disposal-class/zone validation, source/manipulation disable only after arrival while retaining the PhysX prim |
| Deterministic and PhysX controllers | Complete | Portable deterministic controller, Isaac rigid/articulation controller, grasp frames, settle checks, reachability/IK adapter |
| Scene-derived action candidates | Complete | Measurement/decon/shield/move/remove generation with visibility-graph obstacle-avoiding routes, workspace, grasp, collision, support, disposal, parking-slot, and resource feasibility checks; dose terms use Belief only |
| Closed-loop execution | Complete | Measure -> injected estimate -> plan -> physical action -> synchronize -> remeasure -> residual -> belief update in `IsaacWorkflowServices` |
| Physics-step integration | Complete | Lifecycle-safe PhysX event subscription and periodic radiation synchronization |
| ROS 2 integration | Complete | Messages/actions/services, Nav2, MoveIt 2, FollowJointTrajectory, gripper, cancellation/error propagation, Isaac DDS command host |
| UI and artifacts | Partial | Scene/runtime/source/measurement/dose-proxy controls and estimate/residual/plan artifact views exist; the specified Estimation, Countermeasure, Closed Loop, Visualization, and Experiment controls are not complete |
| Visualization layers | Partial | Batched point-based dose proxy exists; source uncertainty, predicted/observed post-action, residual-map, selected-ray material path, and action overlays are not implemented as the specified layers |
| Experiments and reproducibility | Complete | Seeded batch runner, JSON/JSONL/Parquet/report artifacts, regression bounds, vertical-slice multi-seed runner |
| Vertical-slice scenario | Complete | Room, surface source, hidden contaminated object, measurement/countermeasure robots, shield, decon patch, movable obstacle, verification stations |
| Grid Poisson sparse and surface-TV estimators | Complete | Nonnegative Poisson MLE/L1, smooth graph-TV, connected components, Fisher covariance, and bootstrap are unit-tested and consumed through an injected estimator callback |
| Fisher uncertainty and residual hypotheses | Existing | Active-set Fisher covariance, bootstrap, decon/shield/hidden-source/gain/source-location residual handling and belief updates |
| Planner baselines | Existing | Open-loop, greedy dose reduction, nearest source, random, oracle, and closed-loop residual planners |
| Continuous point-source position/activity MLE | Missing | Coarse sparse estimation and connected-component summaries exist, but no continuous-coordinate refinement follows them |
| Concrete PF+MLE estimator | Optional / missing | A generic `SourceEstimator` protocol exists; no concrete particle-filter estimator is implemented |
| Section 29.3 performance matrix | Not demonstrated | Cache and selective invalidation are live-tested, but the four exact target workloads/timings are not captured by the current benchmark suite |

## Validation evidence

- `uv lock --check`: passed.
- `uv run ruff check .`: passed with zero violations (163 repository-wide
  violations fixed in this audit pass).
- Portable pytest suite: 142 passed, 3 skipped. The skipped tests require the
  optional native Embree module in the active `uv` environment; the native host
  build/runtime gate has separate passing evidence.
- Native Embree build and integration tests: passed, including a 17-ray packet/tail case.
- ROS 2 `colcon` build: passed.
- ROS DDS motion gateway gate: passed for Nav2, MoveIt 2, FollowJointTrajectory, gripper, cancellation/result handling.
- Isaac extension-load, vertical-slice, performance/cache, PhysX action/contact-decon, articulation/IK-contract, dashboard, and closed-loop workflow gates: passed.
- Cross-process ROS 2 to Isaac command-host gate: passed.
- Visible Isaac Sim GUI end-to-end articulated run: passed all 8 operations and
  all 13 final invariants with a 12-DOF Ridgeback + Franka and 7-DOF Nova
  Carter; 61 native traces, 157 traced rays, 102 cache hits, and 89 selectively
  patched rays.

The articulation-contract gate still validates an injected deterministic IK
solver. In addition, `real_robot_gate.py`, `articulated_object_gate.py`, and the
full GUI workflow execute the manufacturer Franka asset with Isaac's supported
Lula kinematics configuration. This remains simulation validation and does not
claim physical-hardware validation.

`AnalyticHostBridge` remains intentionally available as a portable baseline. The actual simulator path uses `IsaacRos2CommandHost`; retaining the analytic bridge is not a fallback in the validated Isaac/ROS gate.
