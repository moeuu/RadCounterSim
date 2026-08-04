# RadCounterSim simulator requirement audit

Audit date: 2026-07-15

## Scope and verdict

All simulator-side requirements in the implementation specification are implemented. Source-estimation algorithm development was explicitly excluded from this completion pass. Existing grid/TV estimation, Fisher uncertainty, residual diagnosis, and planner implementations remain available and are connected through injected public interfaces.

The original specification is therefore not literally 100% complete if source-estimation research is included: continuous point-source position/activity MLE refinement and a concrete PF+MLE estimator are not claimed as completed here.

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
| UI and artifacts | Complete | Source authoring with explicit hidden-Truth consent, measurement/dose map, estimate/residual/plan panels, atomic workflow artifact loading/writing |
| Experiments and reproducibility | Complete | Seeded batch runner, JSON/JSONL/Parquet/report artifacts, regression bounds, vertical-slice multi-seed runner |
| Vertical-slice scenario | Complete | Room, surface source, hidden contaminated object, measurement/countermeasure robots, shield, decon patch, movable obstacle, verification stations |
| Grid Poisson sparse and surface-TV estimators | Existing | Implemented and unit-tested before this simulator completion pass; consumed through an injected estimator callback |
| Fisher uncertainty and residual hypotheses | Existing | Active-set Fisher covariance, bootstrap, decon/shield/hidden-source/gain/source-location residual handling and belief updates |
| Planner baselines | Existing | Open-loop, greedy dose reduction, nearest source, random, oracle, and closed-loop residual planners |
| Continuous point-source position/activity MLE | Out of scope | Coarse sparse estimation and connected-component support exist; continuous refinement is not claimed |
| Concrete PF+MLE estimator | Out of scope | A generic `SourceEstimator` protocol exists; no concrete particle-filter estimator is claimed |

## Validation evidence

- `uv run ruff check .`: passed.
- Portable pytest suite: 85 passed, 1 skipped.
- Native Embree build and integration tests: passed, including a 17-ray packet/tail case.
- ROS 2 `colcon` build: passed.
- ROS DDS motion gateway gate: passed for Nav2, MoveIt 2, FollowJointTrajectory, gripper, cancellation/result handling.
- Isaac extension-load, vertical-slice, performance/cache, PhysX action/contact-decon, articulation/IK-contract, dashboard, and closed-loop workflow gates: passed.
- Cross-process ROS 2 to Isaac command-host gate: passed.
- Visible Isaac Sim GUI end-to-end run: passed all 8 operations and all 10 final invariants; 170 native traces, 206 traced rays, 83 cache hits, and 149 selectively patched rays.

The articulation gate validates a live Isaac articulation and the controller's IK contract with an injected deterministic IK solver. `IsaacLulaIkSolver` is implemented as the production adapter, but this audit does not claim physical-hardware validation or a robot-specific Lula calibration file.

`AnalyticHostBridge` remains intentionally available as a portable baseline. The actual simulator path uses `IsaacRos2CommandHost`; retaining the analytic bridge is not a fallback in the validated Isaac/ROS gate.
