# PhysX countermeasure execution

`IsaacPhysicsRobotController` uses `RigidPrim` velocities for base motion, articulation targets supplied by an IK solver for tool motion, and a USD `FixedJoint` for grasp transport. There is intentionally no transform-teleport fallback in physics mode. Removal succeeds only after a removable object is inside a compatible disposal zone.

`ContactDrivenDecontaminator` accepts treatment only from a live PhysX closest-hit query. It checks collision path, tool-to-surface distance, normal alignment, tool speed, and triangle membership. Accepted exposure updates triangle activity through an uncertain Truth-side efficiency field. Removed activity is either discarded by explicit configuration or transferred to a waste source.

The activity NPZ is atomically replaced and both USD SHA-256 attributes are updated. This causes source-activity revision invalidation without forcing a geometry retrace.

## Articulation and IK

`IsaacPhysicsRobotController` accepts an Isaac `Articulation` and an
`InverseKinematicsSolver`. `IsaacLulaIkSolver` is the concrete adapter for Isaac Motion
Generation's Lula and `ArticulationKinematicsSolver`. `move_end_effector()` commands
joint targets through PhysX and terminates only when the USD end-effector frame reaches
the requested pose. `check_reachability()` exposes the same IK solve to scene candidate
generation without moving the robot.

Pick-and-place accepts optional pickup and placement tool poses. When supplied, both
are reached through articulation/IK before the fixed-joint grasp and release. Without
an articulation, the vertical-slice rigid proxy still performs physically stepped base
motion and fixed-joint transport; it never writes the countermeasure object's final
transform directly.
