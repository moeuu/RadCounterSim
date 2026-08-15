# Surface-source and decontamination authoring rules

This file is the repository's canonical specification for surface contamination
and its removal. It records the implementation used for the August 6, 2026
reference videos so later work does not regress to a tidy framed rectangle or an
unverified animation.

## Canonical references

The authoritative visual and behavioral references are:

- `scripts/isaac_shield_placement_video.py`, especially
  `author_irregular_drum_surface_source`, and
  `/home/moeu/Pictures/Research/RadCounterSim/manipulator_surface_source_shield_video_20260806/manipulator_surface_source_shield_placement_20s.mp4`.
  The recorded source has 252 active faces selected from 968 candidates.
- `scripts/isaac_surface_decon_validation.py`, especially `activity_field`,
  `create_high_wall_surface_source`, `_high_reach_scan_target`, and
  `render_high_reach_decontamination_video`, and
  `/home/moeu/Pictures/Research/RadCounterSim/high_reach_wall_decontamination_video_20260806/high_reach_wall_decontamination_20s.mp4`.
  The recorded 20-second run reduced activity from 61,772,808.55 Bq to
  6,670,376.89 Bq (89.20% removed).

The absolute video paths are evidence locations on the development machine, not
runtime dependencies. The scripts and the rules below are the portable source of
truth.

## Surface-source construction

1. Sample a dense two-dimensional lattice on the existing host surface. The
   reference implementations use 44 x 22 candidates on the drum and 48 x 28 on
   a flat wall or floor.
2. Form a deterministic scalar deposition field from several offset,
   differently scaled Gaussian lobes plus correlated sinusoidal roughness. A
   single ellipse, rectangle, or manually arranged handful of polygons is not an
   acceptable substitute.
3. Threshold the field, subtract at least one clean hole, and add detached
   droplets or satellite islands. The result must have an asymmetric ragged
   boundary, concavities, internal gaps, and disconnected deposits.
4. Author geometry only for active cells/faces. Inactive candidates must expose
   the underlying wall, floor, vessel, or component; they must not become a
   visible rectangular backing plate.
5. Store activity per active face/cell, proportional to the local field value.
   Color is a muted scientific overlay blended with the host material and varies
   with local activity. The contamination is not a uniformly colored red sign.
6. Keep fixtures separate. A frame is allowed only when it represents real
   equipment required by the scenario, never to outline or tidy the source.
7. The geometry used by contact/raycast decontamination must be the same
   activity-bearing irregular geometry that is rendered. A hidden regular
   collision or activity proxy is forbidden.
8. A generator must be deterministic for a fixed configuration and expose enough
   metadata to audit candidate count, active count, total activity, and the
   irregular-mask method.

## Decontamination operation

1. Use a continuous boustrophedon/serpentine raster over the contaminated extent,
   as in the six-lane high-wall reference scan. Alternate lane direction and
   choose spacing from the physical tool footprint so the sweep covers the
   irregular field, its holes, and its satellite deposits without converting the
   source to a rectangle. Inset each centerline endpoint by the pad's projected
   half-extent; the pad boundary, rather than an unnecessarily extreme center
   pose, covers the edge of the source.
2. Move the real articulated robot and tool through approach, scan, and retreat
   phases. A visual-only tool animation is insufficient for operational
   validation. Scripted reference renders may use analytical motion, but product
   workflows must retain articulation/IK and physics evidence.
3. Change activity only from verified treatment contact. Enforce surface identity,
   tool-to-surface distance, treatment-axis/normal alignment, maximum surface
   speed, and dwell/exposure. Misses and invalid contacts must not remove
   activity. Sample the full physical pad footprint densely enough for the
   activity-cell pitch; a few center/corner rays are not an area-contact model.
4. Use the reference cumulative-exposure model: each contacted cell/triangle
   decays exponentially according to footprint exposure, local efficiency, and
   the configured removal-rate constant. Give each contacted face one
   speed-adjusted frame of exposure regardless of how many footprint rays hit
   it; ray count is spatial sampling density, not elapsed time. Use the shared
   `effective_contact_exposure_s` model for analytical grids and PhysX contact.
   Removed activity must be transferred to the configured waste sink when that
   mode is enabled.
5. Update the overlay from remaining per-face activity during treatment. Fade it
   toward the host material and hide a face only below the documented residual
   threshold; the reference renderer uses 10% of initial local activity.
6. Success requires physical tool motion, accepted contacts, reduced total
   activity, and meaningful coverage of the active irregular source. Merely
   touching one triangle is not a successful request to decontaminate the whole
   source.
7. Record at minimum: initial/final/removed activity, removed fraction, accepted
   and rejected contacts, treated face indices or coverage, waypoint errors, tool
   path length, robot/navigation audit, and collateral-object displacement.
8. Preserve the full approach -> raster treatment -> retreat sequence and verify
   that navigation and manipulation do not move unrelated drums, shields, or
   obstacles.
9. Natural-language multi-pass requests must remain bounded. A single logical
   decontamination step may request at most five complete passes and may stop
   early only from host-computed public metrics: cumulative remaining or removed
   activity relative to the first pass baseline, or treated
   active-face coverage. Every pass must independently re-resolve feasibility
   and execute the complete approach -> irregular-surface raster -> retreat
   sequence. The LLM must never declare that a threshold was reached, and an
   unmet threshold at the attempt limit must be reported to the operator rather
   than silently presented as success.

## Complex validation environment

The facility in this section belongs to the standalone `vertical-slice`
environment. It must not be silently authored over an imported CAD, point
cloud, or other catalog environment. External environments keep their own
visible/collision geometry and use named catalog spawn anchors for robots,
work surfaces, and inspection cameras. A decontamination overlay placed in an
external environment must still lie on the visible host CAD surface and retain
the exact rendered/contact/activity geometry required above.

1. Product GUI validation uses the deterministic layout returned by
   `decommissioning_facility_layout`: the original cell, remote decontamination
   room, reactor service room, and shield-staging room are connected by three
   declared corridor segments. Do not collapse this validation scene back to a
   single room or a straight two-room demonstration.
2. Author at least ten fixed, individually identified decommissioning obstacles
   or equipment assemblies, including process, maintenance, waste, and service
   equipment. Their room, type, and stable equipment ID must remain available as
   public facility metadata so GUI artifacts can audit the environment without
   depending on display names.
3. Keep the primary decontamination and staging service centerlines clear. The
   pure layout regression uses a 0.65 m minimum planar clearance from declared
   walls and equipment, exceeding the current 0.55 m mobile-planner clearance.
   Added visual complexity must not make a required robot route physically
   impossible or rely on collision-disabled obstacles.
4. The remote room's east wall and the irregular activity-bearing mesh retain
   the same visible collision/contact geometry described above. Additional rooms
   and fixtures must not introduce a regular activity proxy or replace the
   canonical irregular source.
5. A staged secondary physical shield may be present in the shield-staging room,
   but it must have a distinct inventory identity and must not alter or obstruct
   the existing primary shield's pickup, placement, or correction routes.
6. The primary shield's single service handle is authored on its west face. The
   current controller preserves payload world orientation, so shield placement
   must use the matching yaw-zero, handle-side approach. The remote-room service
   alcove keeps the 25% placement base and the return route to the 65% placement
   clear. Do not restore a yaw-pi fallback unless payload rotation or a verified
   second grasp frame and chassis-to-payload collision check are implemented.

## Required maintenance workflow

Any pull request or local change that intentionally alters source generation,
contact constraints, scan planning, activity decay, visual fading, or success
criteria must:

1. update this document with the new invariant and rationale;
2. update or add regression tests for the invariant;
3. run the relevant unit tests and at least one Isaac validation for behavior
   that depends on articulation, PhysX contact, or raycasts; and
4. retain an auditable artifact containing the metrics listed above.

For a bounded multi-pass request, the artifact must additionally record the
logical workflow step, attempt number, configured maximum, completion criterion,
threshold, observed public value, and whether the stop condition was met.

Do not weaken these rules solely to make a failing run pass. Fix geometry,
reachability, scan planning, or physics setup and document any intentional change.
