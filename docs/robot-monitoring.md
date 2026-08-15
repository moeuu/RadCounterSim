# Robot monitoring and camera behavior

The Isaac Operations workspace is designed so an operator can understand every
robot without knowing USD paths or camera commands. The selected robot-set from
the central catalog is the single source of robot names, prim paths, and list
membership.

## Operator surfaces

- The viewport top bar always names one active robot and shows its world
  position, current operation, phase, contact-derived coverage, and progress.
- Each robot row has **見る** for a smooth diagonal rear follow view and **搭載**
  for the selected robot's onboard view. Camera input or a camera-path change
  cancels automatic tracking; pressing **見る** restores it.
- A permanent top-right building overview projects structural CAD bounds,
  contamination markers, all robot positions and headings, the planned route,
  and the current target into a common top-down coordinate frame.
- During travel the selected robot remains in follow view. The first controller
  transition sequence is `approaching` -> `contact_confirmed` ->
  `decontaminating`; the last transition makes one smooth transition to a work
  view framing the robot and treatment target. Later progress events do not
  retrigger the camera.
- Colored 3D bounding beacons, robot names, and camera distance remain visible
  through building geometry. This preserves the opaque CAD model while keeping
  hidden robots locatable.
- The 3D route and target show intent. For measurement, the target is the
  detector measurement position. For decontamination, the physical tool raster
  is the planned path, while treated coverage remains the live per-face fade of
  the exact irregular activity-bearing surface.

## Performance policy

The main 3D viewport retains the application-wide 60 FPS default cap. Monitoring
state and the overview update at 12 Hz, within the 10–15 FPS budget. The overview
is a lightweight `omni.ui.scene` projection and is not another rendered camera.
Onboard mode changes the camera of the single main viewport; no cameras are
rendered for unselected robots.

`RobotMonitorOverlay.audit()` records the update rate, active robot, camera mode,
catalog robot count, map geometry counts, contamination markers, route points,
overlay errors, and the single-rendered-viewport invariant in GUI validation
artifacts.

The articulated CAD decontamination gate must explicitly select the matching
central-catalog robot set; it never injects a hidden task robot into a PackBot or
other operator roster:

```bash
python scripts/run_gui_validation.py \
  --profile fukushima-packbot \
  --robot-set articulated-decommissioning \
  --detector-set vertical-slice-gamma \
  --decontamination-smoke-test --no-keep-open
```
