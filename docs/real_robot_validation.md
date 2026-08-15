# Real robot countermeasure validation

The real-robot path replaces the vertical-slice box placeholders with the
official Isaac Sim 6 assets below.

- Countermeasure: Clearpath Ridgeback + Franka Panda
- Measurement: NVIDIA Nova Carter

The full GUI validation performs these physical operations in one scene:

1. Nova Carter drives through every authored measurement station using its
   differential wheel joints. Each leg is replanned from the robot's live pose;
   a route generated at an earlier station is never reused with a stale start,
   and collision bounds are rebuilt from the live scene before each leg. Route
   planning excludes only the robot being commanded; the other articulated
   robot remains a collision obstacle and must be physically bypassed.
2. Ridgeback drives with three planar articulation joints and folds Franka into
   a repeatable travel posture between manipulation tasks.
3. Franka moves a decontamination pad through a Lula-IK raster. Activity changes
   only after live PhysX contact queries accept pad distance, normal, speed, and
   dwell.
4. Franka opens its fingers, approaches an authored grasp frame, closes its
   fingers, attaches the verified payload to `panda_hand`, lifts, carries,
   places, releases, and checks physical settling. The closed aperture matches
   the authored 30 mm service-handle thickness, preventing an artificial
   interpenetration impulse before the grasp constraint is created. The handle
   projects 240 mm from the plate to preserve wrist clearance during the
   vertical approach while remaining part of the same rigid shield assembly.
5. The same sequence places and corrects a 2.2 kg lead cassette, relocates and
   disposes the contaminated drum, and relocates the large obstacle. Disposal
   is accepted only when the released object's bounds are contained by the
   disposal zone and supported by its floor. Generic-object pickup keeps the
   Ridgeback base 900 mm from the grasp frame: outside the combined chassis and
   drum envelope but inside the verified 950 mm Franka workspace. The target
   object remains a route obstacle during pickup approach and is excluded only
   after it has been grasped and becomes the robot's carried payload. The drum
   crossbar is 500 mm from its center and is joined to the drum by an authored
   support stem, giving the descending wrist clearance from the 320 mm radius.
   Loaded pickup uses an 80 mm vertical clearance lift followed by a diagonal
   140 mm retraction toward the actual pickup-base pose and a 220 mm lift,
   keeping the payload inside Franka's spherical workspace for either approach
   yaw instead of lifting vertically at maximum extension. After relocation,
   disposal re-approaches the drum from the opposite side at the same 900 mm
   stand-off. Base yaw does not rotate the payload in world space, so the
   planner preserves the authored grasp-to-root offset throughout transport;
   this keeps the carried drum on the straight disposal-zone entry line.
6. Embree is synchronized after every operation, then Nova Carter re-measures
   the scene and the public belief is updated from the residual.

Run the complete visible GUI workflow with Isaac Sim's uv environment:

```bash
export RADCOUNTER_HOST_ENV_NO_ROS=1 OMNI_KIT_ACCEPT_EULA=YES
source scripts/host_env.sh
uv run --project "$RADCOUNTER_ISAAC_ROOT" --locked python scripts/run_gui.py
```

The GUI audit is written to `artifacts/gui-validation/latest.json`. For a
noninteractive run, append `--headless --no-keep-open --phase-hold-s 0`.
The persistent visible GUI loop is capped at 60 FPS by default; pass
`--max-fps 30` to lower it or `--max-fps 0` to disable the cap.

The smaller decontamination/shield asset audit remains available with:

```bash
export RADCOUNTER_HOST_ENV_NO_ROS=1 OMNI_KIT_ACCEPT_EULA=YES
source scripts/host_env.sh
uv run --project "$RADCOUNTER_ISAAC_ROOT" --locked \
  python scripts/run_real_robot_validation.py
```

The audit is written to `artifacts/real-robot/latest.json`; phase captures are
written to `artifacts/real-robot/frames/`.

That audit uses the same irregular vertical wall as the full workflow. The
Ridgeback folds the arm, follows the authored corridor into the separate
decontamination room, executes the six-lane contact raster, and returns through
the corridor before starting the shield task.

`tests/isaac/articulated_object_gate.py` separately checks the drum and obstacle
service-handle approach poses after an arm-stow and articulated base move.
