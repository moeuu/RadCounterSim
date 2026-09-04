# RadInterAct Implementation Specification

## 0. Purpose

This document is the Codex-facing design specification for implementing
**RadInterAct**, a closed-loop radiation-source countermeasure simulator that couples
robot actions with changes in the radiation field on NVIDIA Isaac Sim.

The implementation scope includes all of the following capabilities:

- Mobile-robot navigation through the environment
- Robot-arm grasping, movement, placement, and removal of shields, obstacles, and
  contaminated objects
- Representation of surface sources, point sources, and sources attached to contaminated objects
- Local reduction of source intensity through decontamination
- Changes in source-to-detector transmission through shielding
- Changes in source position or presence when contaminated objects move or are removed
- Omnidirectional detectors, detectors with rotating shields, and energy-binned counting
- Fast recalculation of dose-rate maps and detector counts
- Estimation of radiation-source position, intensity, and uncertainty
- A closed loop covering pre-action prediction, execution, post-action measurement,
  predicted-versus-observed residuals, re-estimation, and replanning
- Generation and identification of residual contamination, shield displacement, hidden
  sources, and estimation-scale errors
- Action evaluation that includes measurement time, work time, intervention count,
  shielding material, and robot operating time
- A GUI, headless experiments, logging, reproducibility, unit and integration tests, and
  performance benchmarks
- Extension interfaces for ROS 2, MoveIt 2, and Nav2 integration

## 1. Critical design decisions

### 1.1 Do not fork OceanSim directly

Use OceanSim as a design reference, but implement RadInterAct as an independent Isaac
Sim extension for the following reasons:

- Keep underwater camera and sonar code separate from radiation transport and
  countermeasure code.
- Make future Isaac Sim upgrades easier.
- Allow the radiation kernel to be unit-tested without Isaac Sim.
- Present the research contribution as an independent simulation platform.

### 1.2 Separate truth and belief completely

Maintain two distinct internal states:

- **TruthState**: Actual source distribution, decontamination efficiency, shield pose,
  hidden sources, detector errors, and other ground-truth quantities.
- **BeliefState**: Estimated source distribution, estimated uncertainty, and assumed
  countermeasure effects used during planning.

Estimators and planners must never read `TruthState`. They may use only public
environment geometry, robot state, measurements, and countermeasure-completion notices.

### 1.3 Decouple radiation computation from physics steps

Do not recompute the entire radiation field on every physics tick. Compute it only for:

- Measurement requests
- Dose-map update requests
- Candidate countermeasure evaluation
- Completed pose changes for shields, contaminated objects, or environment geometry
- Activity updates caused by decontamination
- Each step of the closed loop

### 1.4 Provide two execution modes

- **Deterministic action mode**: Determine navigation and grasp success at a high level
  and apply only the final pose and effect. Use this mode for closed-loop algorithm
  evaluation and CI.
- **Physics action mode**: Move robots with PhysX, controllers, grippers, MoveIt 2, and
  related components. Use this mode for evaluation before hardware integration.

Both modes must use the same `CountermeasureAction` and `ActionResult` types.

## 2. Target versions and development environment

### 2.1 Recommended pinned environment

- Ubuntu 24.04
- NVIDIA Isaac Sim 6.0.1
- ROS 2 Jazzy
- Python from the Isaac Sim bundled environment
- C++17 or newer
- Intel Embree 4.x
- NumPy, SciPy, Pydantic, PyYAML, pandas, and pyarrow
- pytest

Pin versions in `requirements-lock.txt` and `environment_manifest.json`. If Isaac Sim
5.0 support is necessary, create a separate branch instead of adding large numbers of
version conditionals to the same codebase.

### 2.2 Development approach

Generate extension templates in an Isaac Sim source workspace and divide the system into
three layers:

1. `radcounter.core`: Isaac Sim-independent Python models, estimation, planning, and logging.
2. `radcounter.radiation.native`: C++/pybind11 backend using Embree.
3. `radcounter.isaac`: Integration with USD, UI, robots, physics, and ROS 2.

## 3. Repository structure

```text
RadInterAct/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements-lock.txt
├── environment_manifest.json
├── docs/
│   ├── architecture.md
│   ├── radiation_model.md
│   ├── usd_metadata.md
│   ├── scenarios.md
│   ├── validation.md
│   └── CHANGELOG.md
├── configs/
│   ├── materials/
│   │   ├── lead.yaml
│   │   ├── steel.yaml
│   │   ├── concrete.yaml
│   │   └── air.yaml
│   ├── isotopes/
│   │   └── example_isotope.yaml
│   ├── detectors/
│   │   ├── omni_counter.yaml
│   │   └── rotating_shield_counter.yaml
│   ├── robots/
│   │   ├── measurement_robot.yaml
│   │   └── countermeasure_robot.yaml
│   └── scenarios/
│       ├── analytic_free_space.yaml
│       ├── shield_demo.yaml
│       ├── decon_demo.yaml
│       ├── movable_source_demo.yaml
│       └── closed_loop_demo.yaml
├── assets/
│   ├── environments/
│   ├── shields/
│   ├── tools/
│   ├── contaminated_objects/
│   └── robots/
├── source/extensions/
│   ├── radcounter.radiation.native/
│   │   ├── config/extension.toml
│   │   ├── premake5.lua
│   │   ├── include/radcounter_radiation/
│   │   ├── src/
│   │   ├── bindings/
│   │   ├── radcounter/radiation/native/__init__.py
│   │   └── tests/
│   └── radcounter.isaac/
│       ├── config/extension.toml
│       ├── data/
│       ├── docs/
│       ├── radcounter/isaac/
│       │   ├── extension.py
│       │   ├── scenario.py
│       │   ├── ui.py
│       │   ├── app_controller.py
│       │   ├── usd/
│       │   ├── robots/
│       │   ├── visualization/
│       │   ├── ros2/
│       │   └── tests/
│       └── premake5.lua
├── radcounter/
│   └── core/
│       ├── models/
│       ├── radiation/
│       ├── sensors/
│       ├── actions/
│       ├── estimation/
│       ├── planning/
│       ├── workflow/
│       ├── experiments/
│       └── logging/
├── ros2_ws/src/
│   ├── radcounter_msgs/
│   └── radcounter_bringup/
├── scripts/
│   ├── run_gui.py
│   ├── run_headless.py
│   ├── validate_scenario.py
│   ├── build_transfer_matrix.py
│   ├── benchmark_radiation.py
│   └── export_run_report.py
├── experiments/
│   ├── baselines/
│   ├── sweeps/
│   └── notebooks/
└── tests/
    ├── unit/
    ├── integration/
    ├── regression/
    └── data/
```

## 4. Extension responsibilities

### 4.1 `radcounter.radiation.native`

Responsibilities:

- Create and destroy Embree devices, scenes, geometry, and instances
- Receive triangle meshes extracted from USD
- Trace finite segments between source and detector points
- Batch-compute per-material path lengths or per-energy transmission
- Update dynamic-object transforms
- Perform parallel computation with the GIL released
- Return an explicit error when Embree is unavailable

This extension must not call UI or USD APIs directly.

### 4.2 `radcounter.isaac`

Responsibilities:

- UI extension and Examples Browser registration
- scenario load/reset/clear
- Read USD meshes, transforms, and custom attributes
- Create robots, sensors, shields, tools, and contaminated objects
- Physics callbacks and event subscriptions
- visualization
- optional ROS 2 bridge

### 4.3 `radcounter.core`

Responsibilities:

- Types for `TruthState`, `BeliefState`, actions, and measurements
- Radiation-source, material, and detector models
- High-level radiation-computation API
- Activity-map and cache management
- Source estimation
- residual diagnosis
- planner
- closed-loop state machine
- Experiment runner and logger

## 5. Core types and state model

Define every public data model with Pydantic v2 or a frozen dataclass. Include units in
field names.

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping
import numpy as np

class SourceType(str, Enum):
    POINT = "point"
    SURFACE = "surface"
    VOLUME = "volume"

@dataclass(frozen=True)
class EmissionLine:
    energy_keV: float
    photons_per_decay: float

@dataclass(frozen=True)
class IsotopeSpec:
    isotope_id: str
    emission_lines: tuple[EmissionLine, ...]

@dataclass
class PointSourceState:
    source_id: str
    position_world_m: np.ndarray
    activity_bq: float
    isotope_id: str
    enabled: bool = True
    attached_prim_path: str | None = None

@dataclass
class SurfaceSourceState:
    source_id: str
    prim_path: str
    triangle_indices: np.ndarray          # [K]
    activity_bq_per_triangle: np.ndarray # [K]
    isotope_id: str
    enabled: bool = True

@dataclass(frozen=True)
class MaterialSpec:
    material_id: str
    energies_keV: np.ndarray
    linear_attenuation_m_inv: np.ndarray
    geometry_mode: str                    # "solid" | "thin_sheet"
    explicit_thickness_m: float | None

@dataclass(frozen=True)
class DetectorSpec:
    detector_id: str
    energy_bin_edges_keV: np.ndarray
    efficiency_energy_keV: np.ndarray
    intrinsic_efficiency: np.ndarray
    background_cps_per_bin: np.ndarray
    dead_time_s: float
    dose_conversion: np.ndarray | None

@dataclass(frozen=True)
class RadiationMeasurement:
    measurement_id: str
    detector_id: str
    timestamp_sim_s: float
    duration_s: float
    position_world_m: np.ndarray
    orientation_world_wxyz: np.ndarray
    counts_per_bin: np.ndarray
    expected_background_counts: np.ndarray
    dose_rate_sv_h: float | None
    covariance: np.ndarray
    scene_revision: int
```

### 5.1 Revision management

```python
@dataclass
class RevisionState:
    geometry_revision: int = 0
    material_revision: int = 0
    source_pose_revision: int = 0
    source_activity_revision: int = 0
    detector_revision: int = 0
```

Update rules:

- Shield, obstacle, or contaminated-object pose change: `geometry_revision += 1`
- Material or thickness change: `material_revision += 1`
- Movement of an object carrying a source: `source_pose_revision += 1`, plus
  `geometry_revision += 1` when needed
- Decontamination: only `source_activity_revision += 1`
- Detector-calibration change: `detector_revision += 1`

Decontamination must not retrace the transfer matrix. Recompute using only the existing
matrix multiplied by the new activity vector.

## 6. USD metadata implementation

The first version must use namespaced custom attributes instead of a custom USD schema
plugin. Store large per-face activity arrays in `.npz` sidecars and keep their URI and
checksum in USD.

### 6.1 Common attributes

```text
rad:role                        token
rad:enabled                     bool
rad:objectId                    string
```

Allowed `rad:role` values:

```text
source
attenuator
shield
contaminated_surface
contaminated_object
detector
decon_tool
disposal_zone
robot
obstacle
```

### 6.2 Source attributes

```text
rad:source:type                 token    point|surface|volume
rad:source:isotopeId            string
rad:source:activityBq           double
rad:source:surfaceActivityBqM2  double
rad:source:activityMapUri       asset
rad:source:activityMapSha256    string
rad:source:hiddenFromEstimator  bool
rad:source:movableWithPrim      bool
```

### 6.3 Attenuator attributes

```text
rad:material:id                 string
rad:material:mode               token    solid|thin_sheet
rad:material:thicknessM         double
rad:material:attenuationUri     asset
rad:shield:movable              bool
rad:shield:resourceUnits        double
```

### 6.4 Decontamination-target attributes

```text
rad:decon:enabled               bool
rad:decon:activityMapUri        asset
rad:decon:efficiencyMean        double
rad:decon:efficiencyStd         double
rad:decon:minToolDwellS         double
rad:decon:surfaceId             string
```

### 6.5 Manipulation-target attributes

```text
rad:manipulation:movable        bool
rad:manipulation:removable      bool
rad:manipulation:graspFrame     token
rad:manipulation:massKg         double
rad:manipulation:disposalClass  string
```

### 6.6 `UsdRadiationRegistry`

Implementation file:

```text
radcounter/isaac/usd/radiation_registry.py
```

public API:

```python
class UsdRadiationRegistry:
    def scan_stage(self, stage) -> "SceneDescriptor": ...
    def register_prim(self, prim_path: str) -> None: ...
    def unregister_prim(self, prim_path: str) -> None: ...
    def on_objects_changed(self, notice) -> None: ...
    def get_source_descriptors(self) -> list[SourceDescriptor]: ...
    def get_attenuator_descriptors(self) -> list[AttenuatorDescriptor]: ...
    def get_detector_descriptors(self) -> list[DetectorDescriptor]: ...
    def get_decon_surfaces(self) -> list[DeconSurfaceDescriptor]: ...
```

Subscribe to USD change notices, classify transform and radiation-attribute changes, and
update revisions. Do not rescan the entire stage for every change.

## 7. USD mesh extraction

Implementation file:

```text
radcounter/isaac/usd/mesh_extractor.py
```

### 7.1 Output type

```python
@dataclass(frozen=True)
class MeshGeometry:
    mesh_id: str
    prim_path: str
    vertices_local_m: np.ndarray      # float32 [N,3]
    triangles: np.ndarray             # uint32 [M,3]
    world_transform: np.ndarray       # float64 [4,4]
    material_id_per_triangle: np.ndarray # int32 [M]
    is_closed_volume: bool
    geometry_mode: str
    explicit_thickness_m: float | None
    dynamic: bool
```

### 7.2 Implementation requirements

- Triangulate `UsdGeom.Mesh` points, `faceVertexCounts`, and `faceVertexIndices`.
- For quads and n-gons, use a USD triangulation utility or stable ear clipping rather
  than fan triangulation.
- Read `metersPerUnit` and convert all values to meters.
- Keep world transforms as double precision and convert to `float32` for Embree input.
- Handle negative scale, non-uniform scale, instances, and prototypes.
- Resolve material bindings and `rad:material:id` for each triangle.
- When collision and visual meshes differ, allow an explicit transport mesh for radiation.
- Prefer `rad:transportMesh=true`; otherwise use the render mesh.
- Preserve an extraction index map so source-surface triangle indices match transport-mesh
  triangle indices.

## 8. Embree backend

### 8.1 Python public interface

```python
from typing import Protocol

class RayTransportBackend(Protocol):
    def build_scene(
        self,
        meshes: list[MeshGeometry],
        material_table: "MaterialTable",
    ) -> None: ...

    def update_instance_transform(
        self,
        mesh_id: str,
        world_transform: np.ndarray,
    ) -> None: ...

    def remove_geometry(self, mesh_id: str) -> None: ...

    def commit_updates(self) -> int: ...

    def trace_transmission(
        self,
        origins_m: np.ndarray,       # [R,3]
        targets_m: np.ndarray,       # [R,3]
        energies_keV: np.ndarray,    # [E]
    ) -> np.ndarray:                 # [R,E]
        ...

    def trace_path_lengths(
        self,
        origins_m: np.ndarray,
        targets_m: np.ndarray,
    ) -> "PathLengthBatch": ...
```

### 8.2 C++ class

```cpp
class EmbreeTransportScene {
public:
    EmbreeTransportScene();
    ~EmbreeTransportScene();

    GeometryHandle addTriangleMesh(
        std::string meshId,
        py::array_t<float> vertices,
        py::array_t<uint32_t> triangles,
        py::array_t<int32_t> materialIds,
        GeometryMode mode,
        std::optional<float> thicknessM,
        bool dynamic);

    void updateTransform(const std::string& meshId,
                         py::array_t<double> transform44);
    void removeGeometry(const std::string& meshId);
    uint64_t commit();

    py::array_t<float> traceTransmission(
        py::array_t<float> origins,
        py::array_t<float> targets,
        py::array_t<float> energies,
        py::array_t<float> attenuationTable);

    PathLengthBatch tracePathLengths(...);
};
```

### 8.3 Scene organization

- Combine static environment meshes into one geometry or a small set grouped by material.
- Register movable shields, contaminated objects, and obstacles as instances, then commit
  after transform updates.
- Parallelize ray queries in C++.
- Release the GIL during computation in pybind11 bindings.
- Use a read-write lock so scene commits and traces never run concurrently.

### 8.4 segment ray tracing

Each ray is a finite segment from a source sample to a detector.

```text
origin = source + eps * dir
tnear = eps
tfar  = distance - eps
```

Repeatedly retrieve the closest hit and set `tnear = hit_t + eps` to collect every
intersection. Enforce a maximum hit count and return an error flag when exceeded.

### 8.5 Path length through solid geometry

- Keep hits ordered by distance.
- Determine entry and exit for each geometry ID.
- When normal orientation is reliable, use `dot(ray_dir, geometric_normal)` to classify
  entry and exit.
- For meshes with unreliable orientation, pair hits per geometry and sum their lengths.
- Warn that an odd number of hits is invalid geometry, then apply conservative or zero
  attenuation according to configuration.
- For nested materials, add the materials of every active geometry.

### 8.6 thin sheet

Represent plates, films, and simplified shield panels as `thin_sheet`; add the following
for each intersection:

```text
effective_thickness = thickness / max(abs(dot(ray_dir, normal)), cos_limit)
```

Apply a cap to prevent an infinite value at grazing angles.

### 8.7 transmission

```text
T(E) = exp(-sum_m mu_m(E) * length_m)
```

Clamp the exponent at a lower bound to avoid numerical underflow. Use
`trace_path_lengths` for debugging and validation and `trace_transmission` as the normal
fast path.

### 8.8 fallback backend

Always implement `AnalyticTransportBackend` with:

- No attenuator
- An analytic single-slab solution
- Core tests that can run in CI when Embree is not installed

## 9. Radiation-source model

### 9.1 point source

For each isotope emission line, compute:

```text
photon_rate_s = activity_bq * photons_per_decay
fluence_rate = photon_rate_s / (4*pi*r^2)
```

### 9.2 surface source

Store activity per triangle.

```text
activity_triangle_bq = surface_activity_bq_m2 * triangle_area_m2
```

Quadrature modes:

- `centroid`: one point at the triangle centroid
- `stratified_n`: `n` points within the triangle
- `adaptive`: adjust `n` using detector distance and triangle size

Each sample has local coordinates and a weight.

```python
@dataclass(frozen=True)
class SourceSampleBatch:
    positions_world_m: np.ndarray  # [S,3]
    activity_bq: np.ndarray        # [S]
    isotope_index: np.ndarray      # [S]
    source_id_index: np.ndarray    # [S]
    triangle_index: np.ndarray     # [S], -1 for a point source
```

For samples attached to movable objects, cache local positions and update only world
positions when the object transform changes.

### 9.3 volume source

This is not required for the first paper, but provide the interface. Convert voxel centers
and voxel activity into a sample batch.

### 9.4 activity repository

```python
class SourceRepository:
    def get_truth_samples(self) -> SourceSampleBatch: ...
    def get_belief_basis_samples(self) -> SourceSampleBatch: ...
    def scale_triangle_activity(self, source_id, triangle_ids, factors): ...
    def move_attached_sources(self, prim_path, transform): ...
    def deactivate_source(self, source_id): ...
```

Use separate repository instances for truth and belief.

## 10. Radiation forward model

```python
class RadiationForwardModel:
    def predict_count_rate(
        self,
        detector_poses: np.ndarray,
        source_samples: SourceSampleBatch,
        detector: DetectorSpec,
        scene_snapshot: SceneSnapshot,
    ) -> CountRatePrediction: ...

    def predict_dose_rate(...): ...
    def build_transfer_matrix(...): ...
```

Contribution from energy line `e`, source sample `s`, and detector pose `d`:

```text
lambda_sde = A_s * Y_e * G(r_sd) * T_sd(E_e) * epsilon_d(E_e, theta_sd)
```

- `G(r)=1/(4*pi*max(r,r_min)^2)`
- `T` is Embree attenuation.
- `epsilon` is detector-response interpolation.
- Aggregate into energy bins.
- Add background.
- Apply the dead-time model.

### 10.1 direct/scatter plugin

```python
class ScatterModel(Protocol):
    def add_scatter(self, direct_prediction, context) -> np.ndarray: ...
```

Implementations:

- `NoScatterModel`
- `EmpiricalBuildupModel`
- `TruthOnlyBiasModel`: Apply spatial bias, energy redistribution, and background drift
  only on the ground-truth side to create mismatch with the estimation model.

Never silently ignore unimplemented scatter. Always store the model name in configuration
and logs.

## 11. Detector implementation

### 11.1 Sensor hierarchy

```python
class RadiationSensor:
    def start_measurement(self, duration_s: float) -> str: ...
    def update(self, sim_time_s: float) -> None: ...
    def cancel(self) -> None: ...
    def get_latest(self) -> RadiationMeasurement | None: ...

class OmnidirectionalCounter(RadiationSensor): ...
class RotatingShieldCounter(RadiationSensor): ...
class DoseRateMeter(RadiationSensor): ...
```

### 11.2 Measurement state machine

```text
IDLE -> INTEGRATING -> FINALIZING -> READY
                  \-> CANCELLED
```

If the detector moves during integration, sample its pose `trajectory_subsamples` times
and compute the average rate. The default configuration uses stationary measurement.

### 11.3 Poisson sampling

```text
expected_counts_bin = rate_cps_bin * duration_s
observed_counts_bin ~ Poisson(expected_counts_bin)
```

Derive a child seed for each detector from the run seed. Reproducibility tests must match
exactly.

### 11.4 Rotating shield

Implement two modes:

1. `physical_geometry`: Rotate an actual shield mesh around the detector and compute
   attenuation with Embree.
2. `response_mask`: Fast mode that applies a precomputed angular response.

Record angle, rotation speed, integration time at each angle, and encoder noise.

## 12. Transfer matrix and cache

### 12.1 Matrix definition

```text
y = H x + b
```

- `x`: Activity of the candidate source basis
- `H`: Unit-activity count response over detector pose, energy bin, and candidate basis
- `b`: background

### 12.2 cache key

```python
@dataclass(frozen=True)
class TransferMatrixKey:
    detector_pose_hash: str
    basis_hash: str
    geometry_revision: int
    material_revision: int
    detector_revision: int
    energy_grid_hash: str
```

Do not include `source_activity_revision` in the key. Decontamination changes only `x`.

### 12.3 partial invalidation

- Shield-pose change: for the MVP, invalidate everything through the geometry revision.
- Optional later optimization: recompute only ray rows that can intersect the shield's
  bounding box.
- Movable-source pose change: update only source-basis columns.

### 12.4 Chunked computation

For large maps, chunk detector evaluation points and source samples so peak temporary
memory remains below the configured limit.

## 13. Dose maps

```python
class DoseMapEvaluator:
    def create_planar_grid(bounds, z_m, resolution_m) -> EvaluationGrid: ...
    def create_3d_grid(bounds, resolution_m) -> EvaluationGrid: ...
    def mask_occupied_cells(grid, collision_scene) -> EvaluationGrid: ...
    def evaluate(grid, state, chunk_size) -> DoseMap: ...
```

Output:

```python
@dataclass(frozen=True)
class DoseMap:
    points_world_m: np.ndarray
    dose_rate_sv_h: np.ndarray
    standard_deviation: np.ndarray | None
    revision: RevisionState
```

## 14. Visualization

Implementation files:

```text
radcounter/isaac/visualization/
├── dose_map_visualizer.py
├── source_estimate_visualizer.py
├── residual_visualizer.py
├── ray_debug_visualizer.py
└── action_visualizer.py
```

Requirements:

- Use `UsdGeom.Points` or an instancer for 2D heatmaps; do not create large numbers of
  per-cell cubes.
- Support fixed, percentile, and logarithmic color ranges.
- Show the truth-source overlay only with debug authorization and an explicit toggle.
- Put belief sources, uncertainty, predicted post-action values, observed post-action
  values, and normalized residuals on separate layers.
- Display material paths and path lengths for selected rays.

## 15. Robot abstraction

```python
class RobotController(Protocol):
    async def navigate_to(self, pose, timeout_s) -> "ExecutionStatus": ...
    async def move_end_effector(self, pose, timeout_s) -> "ExecutionStatus": ...
    async def execute_joint_trajectory(self, trajectory) -> "ExecutionStatus": ...
    async def grasp(self, target_prim_path) -> "ExecutionStatus": ...
    async def release(self) -> "ExecutionStatus": ...
    async def stop(self) -> None: ...

class DeterministicRobotController(RobotController): ...
class IsaacPhysicsRobotController(RobotController): ...
class Ros2RobotController(RobotController): ...
```

### 15.1 Measurement robot

- Differential-drive or omnidirectional mobile base
- Detector mast
- Derive detector pose from the robot-base transform and sensor extrinsics
- Optional Nav2 integration

### 15.2 Countermeasure robot

- A mobile base, manipulator, and gripper are recommended.
- A fixed manipulator is acceptable initially, but design the controller interface for a
  mobile manipulator.
- It can manipulate shields, obstacles, and contaminated objects.
- It carries the decontamination tool through a tool changer or fixed attachment.

### 15.3 Grasp implementation

Two modes:

- Deterministic: after verifying reachability of the target grasp frame and a collision-free
  path, parent or constrain the target prim to the gripper prim.
- Physics: use a surface gripper or fixed-joint constraint and verify contact and relative pose.

After grasping, reflect the target pose in the radiation registry.

## 16. action model

```python
class ActionType(str, Enum):
    MEASURE = "measure"
    DECONTAMINATE = "decontaminate"
    PLACE_SHIELD = "place_shield"
    MOVE_SHIELD = "move_shield"
    MOVE_OBJECT = "move_object"
    REMOVE_OBJECT = "remove_object"
    REPAIR_ACTION = "repair_action"

@dataclass(frozen=True)
class CountermeasureAction:
    action_id: str
    action_type: ActionType
    robot_id: str
    target_prim_path: str | None
    target_region: dict | None
    target_pose_world: np.ndarray | None
    parameters: dict
    predicted_duration_s: float
    resource_cost: dict[str, float]

@dataclass(frozen=True)
class ActionResult:
    action_id: str
    status: str
    started_sim_s: float
    completed_sim_s: float
    public_details: dict
    truth_details: dict | None
    before_revision: RevisionState
    after_revision: RevisionState
```

Only the experiment logger may access `truth_details`; never pass it to estimators or planners.

## 17. Decontamination implementation

### 17.1 activity map

Store surface-source activity per triangle in `.npz`.

```text
triangle_indices
activity_bq
last_treated_step
cumulative_treatment_exposure
```

### 17.2 tool footprint

`DeconToolModel`:

```python
@dataclass(frozen=True)
class DeconToolSpec:
    footprint_sample_points_local_m: np.ndarray
    treatment_axis_local: np.ndarray
    max_contact_distance_m: float
    max_normal_angle_deg: float
    max_surface_speed_m_s: float
    rate_constant_s_inv: float
```

On each physics tick, cast footprint sample rays along the tool axis.

Valid-contact conditions:

- Distance to the target surface is at or below the threshold.
- Angle between the tool axis and surface normal is at or below the threshold.
- End-effector tangential speed over the surface is at or below the limit.
- The target prim has `rad:decon:enabled=true`.

Accumulate exposure per triangle.

```text
E_i += contact_weight * dt
nominal_removal_fraction_i = 1 - exp(-k * E_i)
```

### 17.3 ground truth failure

On the truth side, use:

```text
actual_fraction_i = clamp(nominal_fraction_i * local_efficiency_i, 0, 1)
local_efficiency_i ~ spatially correlated random field
```

This allows generation of untreated spots, tool-position offsets, and efficiency variation.

### 17.4 removed activity

Support two configured modes:

- `discard`: Remove treated activity from the scene.
- `transfer_to_waste`: Move treated activity to a waste-container source.

The second mode is recommended for research; do not imply that radioactivity disappears
merely because a surface was decontaminated.

### 17.5 API

```python
class DecontaminationExecutor:
    async def execute(self, action, robot, truth_state) -> ActionResult: ...
    def preview_nominal_effect(self, action, belief_state) -> SourceStateDelta: ...
```

## 18. Shielding implementation

### 18.1 shield asset

Each shield asset contains:

- visual mesh
- collider
- mass/inertia
- grasp frame
- support/contact frame
- radiation transport mesh
- material ID
- Solid or thin-sheet mode
- nominal thickness
- resource units

### 18.2 shield action

```text
PLAN -> NAVIGATE_TO_SHIELD -> GRASP -> NAVIGATE_TO_TARGET
-> PLACE -> RELEASE -> WAIT_SETTLE -> COMMIT_RADIATION_SCENE -> COMPLETE
```

In `WAIT_SETTLE`, wait until linear and angular velocity fall below their thresholds. On
timeout, return `FAILED` or `PARTIAL`.

### 18.3 Shield displacement

In truth mode, add translational and rotational errors to the target pose.

```text
actual_pose = target_pose * pose_error_transform
```

Read the actual pose from USD and apply it to the radiation scene. Do not use the planner's
predicted pose directly.

### 18.4 Immediate effect measurement

After shield placement:

1. Update the geometry revision.
2. Update and commit the Embree instance transform.
3. Compute predicted dose at selected verification poses.
4. Move the measurement robot to a verification pose.
5. Perform the measurement.
6. Generate the residual.

## 19. Moving and removing contaminated objects and obstacles

### 19.1 contaminated object

Store source samples in the object's local frame. Update their world poses when the object
pose changes.

### 19.2 obstacle

A non-contaminated obstacle has no source, but can affect robot reachability, path planning,
and radiation attenuation. If `rad:material:id` is set, register it as attenuation geometry.

### 19.3 move action

```text
NAVIGATE -> GRASP/PUSH -> MOVE -> RELEASE -> SETTLE -> UPDATE SOURCE/GEOMETRY
```

Support push and pick action subtypes.

### 19.4 remove action

Do not simply disable a source through the API. After verifying that the target entered a
disposal zone, do one of the following:

- Leave it in the scene with the disposal container's shielding.
- Move it outside the evaluation domain and deactivate the source.

Log where activity is stored before and after removal.

## 20. Measurement and source estimation

### 20.1 candidate basis

The initial implementation supports two basis types:

- 3D grid basis for unknown point or small voxel sources
- Surface-triangle basis for contamination on floors, walls, and object surfaces

### 20.2 Poisson sparse estimator

For observed counts `y`, response matrix `H`, and background `b`, solve:

```text
minimize_x>=0  sum_i [(Hx+b)_i - y_i log((Hx+b)_i)]
              + lambda_l1 ||x||_1
              + lambda_tv TV(x)
```

Implementation classes:

```python
class SourceEstimator(Protocol):
    def fit(self, measurements, basis, forward_model) -> "SourceEstimate": ...
    def update(self, previous, new_measurements, action_context) -> "SourceEstimate": ...

class GridPoissonSparseEstimator(SourceEstimator): ...
class SurfacePoissonTVEstimator(SourceEstimator): ...
class PFPlusMLEEstimator(SourceEstimator): ...   # optional phase
```

### 20.3 solver

The initial version uses SciPy and implements:

- Nonnegative L-BFGS-B for unregularized MLE
- Proximal gradient or FISTA for L1
- A simplified TV proximal using a graph incidence matrix, or split Bregman
- Finite-difference verification of gradient tests

### 20.4 local refinement

Refine leading sparse-grid candidates with continuous-coordinate MLE.

```text
coarse sparse grid -> connected components -> source seeds
-> continuous position/activity MLE
```

### 20.5 uncertainty

At minimum, implement a Fisher-information approximation on the active set.

```text
F = H_A^T diag(1/max(lambda, eps)) H_A + regularization
Cov = pseudo_inverse(F)
```

Also provide a bootstrap option.

### 20.6 Output

```python
@dataclass(frozen=True)
class SourceEstimate:
    estimate_id: str
    basis_activity_bq: np.ndarray
    covariance_diag: np.ndarray
    point_hypotheses: tuple
    predicted_measurements: np.ndarray
    objective_value: float
    converged: bool
    diagnostics: dict
```

## 21. prediction–measurement residual

### 21.1 predicted post-action

The planner applies the action's nominal effect to a clone of `BeliefState`.

```python
predicted_belief_after = action_model.preview(action, belief_before)
predicted_measurement = forward_model.predict(
    verification_poses,
    predicted_belief_after,
)
```

### 21.2 observed post-action

The action executor applies a stochastic actual effect to `TruthState`, updates the actual
USD pose and source activity, and performs a verification measurement.

### 21.3 normalized residual

```text
r = y_observed - y_predicted
z = r / sqrt(max(y_predicted + variance_model, 1))
```

Preserve energy bin, pose, and time.

### 21.4 residual hypotheses

```python
class ResidualHypothesis(Protocol):
    hypothesis_id: str
    def fit(self, context) -> HypothesisFit: ...

class DeconResidualHypothesis: ...
class ShieldPoseErrorHypothesis: ...
class HiddenSourceHypothesis: ...
class GlobalGainBackgroundHypothesis: ...
class SourceLocalizationErrorHypothesis: ...
```

#### DeconResidualHypothesis

Regress the residual activity coefficient inside the treated region.

#### ShieldPoseErrorHypothesis

Generate a finite set of candidates around the nominal pose, recompute predicted
measurements for each, and select the maximum-likelihood pose correction.

#### HiddenSourceHypothesis

Subtract known source contributions from the residual and perform sparse inversion over
unused candidate bases.

#### GlobalGainBackgroundHypothesis

```text
y_observed ≈ gain * y_predicted + background_offset
```

Fit this model.

### 21.5 hypothesis selection

For each hypothesis, compute BIC from negative log likelihood and parameter count. Return
the best hypothesis and confidence, then update `BeliefState` and action-effect parameters.

## 22. planner

### 22.1 action candidate generator

```python
class ActionCandidateGenerator:
    def generate_measurement_actions(...): ...
    def generate_decon_actions(...): ...
    def generate_shield_actions(...): ...
    def generate_move_remove_actions(...): ...
    def generate_repair_actions(...): ...
```

### 22.2 feasibility

For each candidate, evaluate:

- Availability of a mobile path
- Manipulator reachability
- collision
- grasp frame
- shield placement stability
- Disposal-zone capacity
- Resource availability
- Robot availability

Use deterministic geometric feasibility for the MVP and a controller dry run in physics mode.

### 22.3 objective

```text
score(a) =
  w_dose * expected_task_path_dose_after(a)
+ w_peak * expected_peak_dose_after(a)
+ w_unc  * residual_source_uncertainty_after(a)
+ w_time * action_time(a)
+ w_res  * resource_cost(a)
+ w_risk * robot_execution_risk(a)
- w_info * expected_information_gain(a)
```

Lower scores are better.

### 22.4 resource state

```python
@dataclass
class ResourceState:
    remaining_measurement_time_s: float
    remaining_work_time_s: float
    remaining_robot_runtime_s: dict[str, float]
    remaining_shield_units: dict[str, int]
    remaining_decon_media: float
    remaining_countermeasure_count: int
```

### 22.5 baseline planners

Always implement these planners for paper comparisons:

- `OpenLoopPlanner`
- `GreedyDoseReductionPlanner`
- `NearestSourcePlanner`
- `RandomPlanner`
- `OraclePlanner` — uses `TruthState`, but only for experimental evaluation
- `ClosedLoopResidualPlanner` — proposed method

## 23. closed-loop orchestrator

```python
class WorkflowState(str, Enum):
    INITIALIZE = "initialize"
    MEASURE = "measure"
    ESTIMATE = "estimate"
    PLAN = "plan"
    PREDICT = "predict"
    EXECUTE = "execute"
    VERIFY = "verify"
    DIAGNOSE = "diagnose"
    UPDATE = "update"
    COMPLETE = "complete"
    FAILED = "failed"
```

```python
class ClosedLoopCoordinator:
    async def run_episode(self, config) -> EpisodeResult: ...
    async def step(self) -> WorkflowState: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def stop(self) -> None: ...
```

Transitions:

```text
INITIALIZE
 -> MEASURE
 -> ESTIMATE
 -> PLAN
 -> PREDICT
 -> EXECUTE
 -> VERIFY
 -> DIAGNOSE
 -> UPDATE
 -> PLAN or COMPLETE
```

Termination conditions:

- Task-path dose is below the threshold
- Peak dose is below the threshold
- Resources are exhausted
- Maximum step count is reached
- No valid action remains
- safety violation

Save an immutable snapshot at every transition.

## 24. UI

Implement `radcounter.isaac` from the current UI template.

### 24.1 Frames

1. **Scenario**
   - config path
   - Load/Reset/Clear
   - seed
   - deterministic/physics mode
2. **Radiation Scene**
   - stage scan
   - rebuild Embree
   - revision display
   - material/source counts
3. **Measurement**
   - detector selection
   - duration
   - start/cancel
   - latest count/dose
4. **Estimation**
   - basis selection
   - regularization parameters
   - run/update
5. **Countermeasure**
   - action type
   - target prim/region
   - preview effect
   - execute
6. **Closed Loop**
   - one step
   - auto run
   - pause/stop
   - current workflow state
7. **Visualization**
   - truth toggle
   - estimate toggle
   - dose map
   - residual map
   - ray debug
8. **Experiment**
   - baseline
   - run ID
   - save report

Do not run heavy computation synchronously inside UI callbacks. Create asynchronous tasks
and manage progress and cancellation tokens.

## 25. ROS 2 interface

Core functionality must work without ROS 2. Treat ROS 2 as an adapter.

### 25.1 messages

`RadiationMeasurement.msg`

```text
std_msgs/Header header
string measurement_id
string detector_id
geometry_msgs/Pose detector_pose
float64 duration_s
float64[] energy_bin_edges_kev
uint32[] counts_per_bin
float64 dose_rate_sv_h
float64[] covariance_flat
uint64 scene_revision
```

`SourceEstimate.msg`

```text
std_msgs/Header header
string estimate_id
geometry_msgs/Point[] positions
float64[] activity_bq
float64[] position_covariance_flat
float64[] activity_std_bq
```

`CountermeasureStatus.msg`

```text
std_msgs/Header header
string action_id
string action_type
string state
float32 progress
string message
```

### 25.2 actions/services

```text
MeasureRadiation.action
ExecuteCountermeasure.action
GetDoseMap.srv
EvaluateCountermeasure.srv
ResetEpisode.srv
```

### 25.3 robot topics

Use standard `/tf`, `/joint_states`, `/cmd_vel`, `FollowJointTrajectory`, and MoveIt 2.

## 26. scenario configuration

```yaml
schema_version: 1
scenario_id: closed_loop_demo_001
seed: 42

simulation:
  action_mode: deterministic
  physics_dt_s: 0.008333333
  rendering_dt_s: 0.033333333
  max_episode_steps: 8

world:
  usd_path: assets/environments/demo_room.usd
  radiation_transport_mesh_policy: tagged_or_render

radiation:
  backend: embree
  scatter_model_truth: truth_only_bias
  scatter_model_planner: none
  min_distance_m: 0.05
  ray_epsilon_m: 1.0e-4
  max_hits_per_ray: 128
  source_quadrature: adaptive
  chunk_rays: 100000

materials:
  directory: configs/materials

sources:
  - id: floor_contamination
    type: surface
    prim_path: /World/Room/Floor
    isotope_id: example_isotope
    activity_map_uri: assets/environments/maps/floor_activity.npz
    hidden_from_estimator: false
  - id: hidden_object_source
    type: point
    prim_path: /World/Props/Box03
    activity_bq: 1.0e6
    isotope_id: example_isotope
    hidden_from_estimator: true

robots:
  measurement:
    config: configs/robots/measurement_robot.yaml
    initial_pose: [0.5, 0.5, 0.0, 0.0]
  countermeasure:
    config: configs/robots/countermeasure_robot.yaml
    initial_pose: [1.0, 0.5, 0.0, 0.0]

detectors:
  - config: configs/detectors/rotating_shield_counter.yaml
    robot_id: measurement
    mount_prim: /World/MeasurementRobot/SensorMount

truth_action_uncertainty:
  decon_efficiency_mean: 0.75
  decon_efficiency_std: 0.15
  shield_translation_std_m: 0.03
  shield_rotation_std_deg: 2.0
  grasp_failure_probability: 0.02

estimator:
  type: surface_poisson_tv
  lambda_l1: 1.0e-3
  lambda_tv: 1.0e-2
  max_iterations: 500
  uncertainty: fisher

planner:
  type: closed_loop_residual
  weights:
    task_path_dose: 1.0
    peak_dose: 0.3
    uncertainty: 0.2
    time: 0.05
    resource: 0.1
    execution_risk: 0.1
    information_gain: 0.2

resources:
  measurement_time_s: 600
  work_time_s: 1800
  robot_runtime_s:
    measurement: 1800
    countermeasure: 1800
  shields:
    lead_panel_small: 2
  decon_media_units: 100
  max_countermeasure_count: 6

verification:
  poses:
    - [1.0, 1.0, 0.8]
    - [2.0, 1.0, 0.8]
  measurement_duration_s: 10.0

outputs:
  run_root: outputs
  save_truth: true
  save_transfer_matrices: false
  save_dose_maps: true
  save_video: false
```

Before starting Isaac Sim, use `validate_scenario.py` to check consistency among the
schema, asset paths, units, prim paths, and material table.

## 27. Logging and experiment reproducibility

run directory:

```text
outputs/<scenario>/<timestamp>_<run_id>/
├── manifest.json
├── resolved_config.yaml
├── events.jsonl
├── measurements.parquet
├── estimates.parquet
├── actions.parquet
├── resources.parquet
├── metrics.json
├── maps/
├── snapshots/
└── report.html
```

`manifest.json`:

- git commit SHA
- dirty status
- Isaac Sim version
- Embree version
- OS, CPU, and GPU
- Python package versions
- random seeds
- config SHA256
- asset SHA256

Important events:

- scene loaded
- radiation scene committed
- measurement started/completed
- estimate completed
- action previewed/started/completed/failed
- verification completed
- residual hypothesis selected
- belief updated
- resource consumed
- episode completed

## 28. test suite

### 28.1 unit tests — radiation

1. A point source in free space follows `1/r^2`.
2. A single slab follows `exp(-mu*l)`.
3. Exponents from a two-material slab add correctly.
4. Oblique-incidence thickness is correct for a thin sheet.
5. Entry/exit path length through a closed cube is correct.
6. Nested solids are correct.
7. A rectangular surface source converges to a high-density numerical-integration reference.
8. Zero activity, disabled sources, and zero efficiency are handled.
9. A Poisson seed reproduces results.
10. Cache invalidation follows the revision rules.

### 28.2 unit tests — actions

1. A nominal 50% decontamination action halves target-triangle activity.
2. Non-target triangles do not change.
3. Repeated decontamination accumulates.
4. Removed activity moves to the waste source.
5. Shield placement increments the geometry revision.
6. Shield movement changes transmission.
7. Source-sample poses follow movement of a contaminated object.
8. A source is not deactivated before reaching the disposal zone.

### 28.3 unit tests — estimation

1. Recover a noiseless single source.
2. Mean error for a Poisson-noisy single source is within tolerance.
3. Two-source case.
4. Sparse surface patch.
5. Test zero and nonzero regularization.
6. Finite-difference gradient check.
7. Fisher covariance shape and positive semidefiniteness.

### 28.4 unit tests — residual

1. Select the decontamination-residual hypothesis correctly.
2. Approximately recover shield translation error.
3. Detect a hidden source as a new candidate.
4. Avoid misclassifying global gain error as source error.

### 28.5 integration tests — headless Isaac

`test_closed_loop_smoke.py`:

1. stage load
2. robot spawn
3. source/material scan
4. Embree build
5. measurement robot move
6. initial measurement
7. estimate
8. shield action
9. verification measurement
10. residual update
11. second action plan
12. output files validation

`test_all_countermeasures.py`:

- decon
- shield
- move contaminated object
- remove object
- move non-contaminated obstacle

### 28.6 regression tests

For small fixed-seed scenarios, save key metrics and map hashes. Review the reason for any
difference after a version update.

## 29. validation criteria

### 29.1 physics/math

- free-space analytic relative error < 1e-6
- slab attenuation relative error < 1e-4
- Report surface-quadrature convergence.
- Material-path debugging matches expected lengths.

### 29.2 functional

- Every action completes in deterministic mode.
- At least one shield pick-and-place and obstacle movement completes in physics mode.
- The decontamination footprint updates the activity map along the tool path.
- Detector measurements after an action reflect scene state.
- A test double proves that the estimator does not read `TruthState`.

### 29.3 performance target

Identify reference hardware in the manifest and use these initial targets:

- Transmission query for 100,000 segment rays: at most 0.2 s
- 2D 128 x 128 map with 2,000 source samples and no cache: at most 2 s
- Map after an activity-only update: at most 0.1 s
- Shield-pose update plus verification at 32 poses: at most 0.5 s

Prioritize correctness even when a target is missed, and save benchmark results. When
publishing performance values, report the reference hardware and scene complexity.

## 30. Implementation order

### Milestone 0: scaffold

- extension templates
- core package
- CI
- config validation
- logging skeleton

Completion criterion: the UI appears in Isaac Sim and the headless startup test passes.

### Milestone 1: analytic radiation core

- data models
- point source
- material interpolation
- detector model
- analytic backend
- unit tests

Completion criterion: free-space and slab tests pass.

### Milestone 2: USD registry + mesh extraction

- custom attributes
- stage scan
- mesh triangulation
- revision management

Completion criterion: the expected source and attenuator descriptors are obtained from
the demo USD.

### Milestone 3: Embree backend

- C++ extension
- pybind
- scene build/update
- multi-hit path length
- benchmarks

Completion criterion: analytic slab/cube tests and the batch performance test pass.

### Milestone 4: source sampling + radiation sensor

- surface source
- detector integration
- Poisson measurement
- rotating shield sensor

Completion criterion: a robot-mounted sensor returns a measurement.

### Milestone 5: map/cache/visualization

- transfer matrix
- cache
- dose map
- UI visualization

Completion criterion: a decontamination activity update refreshes the map without
rebuilding rays.

### Milestone 6: deterministic countermeasure actions

- decon
- shield
- move/remove
- resources
- action state machine

Completion criterion: integration tests for every action pass.

### Milestone 7: robot physics execution

- navigation adapter
- manipulator adapter
- grasp/release
- shield placement
- obstacle/source object movement
- decon tool contact

Completion criterion: the physics-mode smoke test passes.

### Milestone 8: estimation

- grid/surface basis
- Poisson MLE/L1/TV
- uncertainty
- estimate visualization

Completion criterion: synthetic recovery tests pass.

### Milestone 9: residual diagnosis

- predicted post-action
- verification measurement
- four residual hypotheses
- belief update

Completion criterion: fault-injection tests pass.

### Milestone 10: planner + closed loop

- candidates
- feasibility
- objective
- resources
- baselines
- coordinator

Completion criterion: generate a test in which the closed-loop demo improves the specified
metric over an open-loop baseline. Do not require a fixed statistical significance margin
for CI to pass.

### Milestone 11: ROS 2 and experiment automation

- messages/actions
- MoveIt/Nav2 adapter
- batch runner
- reports

Completion criterion: ROS 2 measurement and action round trips work, and the headless
sweep runs.

## 31. Definition of Done

The simulator is implementation-complete only when every item below is satisfied:

- It starts in both GUI and headless modes.
- It supports point sources, surface sources, and sources on contaminated objects.
- It supports air, solid, and thin-sheet attenuation.
- Decontamination, shielding, movement, and removal update both scene and radiation field.
- Mobile-robot and manipulator action interfaces exist.
- Every action completes in deterministic mode.
- Physics mode includes demonstrated shield and object manipulation.
- Detector counts and dose maps update immediately before and after actions.
- Source estimation and uncertainty work.
- Predicted values, observations, and residuals are saved and visualized.
- Belief updates based on residuals work.
- The resource-constrained planner and baselines work.
- The `TruthState` leakage test passes.
- All unit and integration tests pass.
- A manifest containing version, seed, configuration, and hardware is saved.
- The API and scenario schema are documented.
- One end-to-end closed-loop demo and reproduction script exist.

## 32. Prohibited implementation practices

- Never access `TruthState` from an estimator or planner.
- Do not implement decontamination and shielding as the same activity-scaling operation.
- After shield placement, read the actual USD pose instead of passing the predicted pose
  directly to the radiation engine.
- Do not recompute the entire dose map on every physics tick.
- Do not embed large per-face activity arrays directly in USD custom attributes.
- Do not leave units for source activity, dose, count, or length ambiguous.
- Do not run heavy synchronous computation in UI callbacks.
- Do not hardcode physical data without a source and version.
- Do not implement actions as visual animation only. Always update scene state and radiation
  state consistently.
- Do not reveal post-action effects to the planner through direct `TruthState` differences.
  Always obtain them through remeasurement.
