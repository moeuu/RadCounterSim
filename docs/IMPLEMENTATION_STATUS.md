# Non-estimation implementation status

Source-estimation algorithm development is excluded from this implementation pass. Existing estimator interfaces remain available to consume measurements.

| Requirement | Implementation evidence |
|---|---|
| Truth and Belief separation | Hidden USD sources are included in Truth measurement and removed from `belief_source_paths`. |
| USD radiation metadata | Canonical `rad:*` author/registry plus SHA-verified NPZ activity maps. |
| Dynamic Embree | Transform update, removal, finite segments, solid path lengths, thin-sheet angle correction. |
| Revision/cache behavior | Five revision classes and selective finite-ray AABB invalidation; activity-only projection does not retrace. |
| Detectors | Energy-line efficiency, background, dead time, Poisson integration, station and dose-proxy queries. |
| Decontamination | PhysX contact query, footprint, distance/normal/speed checks, triangle exposure, uncertain efficiency, waste transfer. |
| Shield/move/remove | PhysX velocity controller, IK-only arm interface, FixedJoint grasp, settle and disposal validation. |
| Executable scene | Self-contained A4-room-equivalent vertical slice USD and generated sidecar. |
| ROS 2/Nav2/MoveIt | Jazzy action gateway, gripper client, cancel/feedback/result handling, Isaac physics-thread command host. |
| UI | Operations dashboard for stage, runtime, measurements, dose proxy, synchronization, and artifact export. |
| Experiments | Seed sweep with nominal/Truth pose separation and checksummed atomic artifacts. |
| Gates | Native Embree, live Isaac/PhysX, ROS DDS roundtrip, regression and performance gates. |
