# Real robot countermeasure validation

The real-robot path replaces the vertical-slice box placeholders with the
official Isaac Sim 6 assets below.

- Countermeasure: Clearpath Ridgeback + Franka Panda
- Measurement: NVIDIA Nova Carter

The validation performs these physical operations in one scene:

1. Nova Carter drives using its left and right wheel joints.
2. Franka moves a decontamination pad through Lula IK joint targets.
3. Activity changes only after live PhysX contact queries accept the pad pose.
4. Franka opens its fingers, grasps a 2.2 kg lead cassette, lifts it, carries it,
   places it between source and detector, and opens its fingers.
5. Embree transport is synchronized and the source-specific response is checked.

Run the visible validation with Isaac Sim's uv environment:

```bash
export RADCOUNTER_HOST_ENV_NO_ROS=1 OMNI_KIT_ACCEPT_EULA=YES
source scripts/host_env.sh
uv run --project "$RADCOUNTER_ISAAC_ROOT" --locked \
  python scripts/run_real_robot_validation.py
```

The audit is written to `artifacts/real-robot/latest.json`; phase captures are
written to `artifacts/real-robot/frames/`.
