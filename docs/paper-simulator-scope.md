# Paper simulator scope and evaluation contract

Decision date: 2026-08-29

## Central contribution

RadInterAct evaluates radiological measurement and robot-executed
countermeasures in one mutable three-dimensional scene. A robot action changes
the same geometry, source pose/presence, or spatial activity distribution used
by the next radiation calculation. This scene consistency is the paper's
central contribution.

The paper is a simulator-development and validation study. It does not present
a new source-estimation algorithm, action-selection algorithm, language model,
ROS interface, or universal robot controller as its primary contribution.

This decision follows the measurement-and-countermeasure requirements in
`20260521GroupI-Morita.pdf`, especially the distinct mechanisms for treatment,
shielding, relocation/removal, their resulting dose changes, and the evaluation
of effect, remaining uncertainty, and consumed resources (pp. 3, 10, and 12).

## Final simulator included in the paper

| Element | Final decision |
| --- | --- |
| Scene state | USD authors geometry, materials, source metadata, and per-face activity. PhysX and Embree are synchronized derivatives with revision and hash audits. |
| Radiation sources | Point samples, activity on the exact irregular visible triangle mesh, and voxelized volume activity. |
| Main radiation configuration | Cs-137 gamma emission with energy-resolved primary attenuation. A buildup-corrected result is reported only after the exact correction table passes the research-data gate. |
| Main detectors | A calibrated omnidirectional detector and a rotating-shield detector whose physical shielding geometry is synchronized with the scene. The checked-in implementations remain synthetic until their controlled calibrations are supplied. |
| Detector observations | Response matrices/effective area, energy redistribution, angular response, background, Poisson counts, dead time, saturation, pose uncertainty, and encoder uncertainty where applicable. |
| Surface treatment | Contact-, normal-, speed-, footprint-, and dwell-conditioned dry removal on the displayed irregular mesh, with removed activity assigned to waste and an explicit balance. |
| Shielding | Physical grasp, transport, placement, correction, settling, and subsequent path recalculation. |
| Relocation | Physical manipulation updates the contaminated object's root pose and attached source pose. |
| Disposal | The object and source remain present inside an explicit shielded storage region; containment is checked after settling. |
| Robots | Articulated Nova Carter measurement robot and Ridgeback + Franka countermeasure robot using wheel/arm joints, Lula IK, gripper actuation, constraints, contact, and collision. |
| Evidence | The articulated countermeasure result accepts only `physical_robot_execution`. The direct-USD rotating-shield component gate is `kinematic_scene_edit`; it remains a separate detector-validation result and is never pooled with robot-execution metrics. |
| Reproducibility | Fixed seeds, stage/config/data hashes, runtime versions, immutable artifacts, and aggregate statistics over an announced seed set. |

## Implemented platform extensions

The repository also implements the following capabilities because they are
useful for later experiments and prevent the architecture from being tied to
one sensor or operation:

- reference-table photon buildup correction;
- gamma, X-ray, neutron, alpha, and beta transport kernels;
- eighteen detector models across nine families plus plugins/external readings;
- reduced-order pressure-water treatment with water and activity balances;
- grid, continuous, and particle-assisted source estimation;
- uncertainty-aware, resource-constrained action selection;
- visualization/dashboard controls, strict local natural-language commands,
  ROS 2 integration, and descriptor-based robot fleets.

These are implemented capabilities, not automatically validated paper claims.
Each needs its own applicable calibration and quantitative evidence before it is
reported as scientifically accurate. The main abstract should name only the
features actually exercised by the paper's quantitative experiments.

## Evaluation design

### 1. Radiation and detector validation

- inverse-square and slab-attenuation analytic checks;
- comparison of primary and, when enabled, corrected photon response with an
  independent photon-transport reference;
- controlled measurements for count rate, energy-bin response, background,
  dead time, and rotating-shield angular response;
- source surface/volume discretization convergence;
- reported error and uncertainty over the stated energy, distance, material,
  thickness, and angle ranges only.

The physical rotating-shield program uses
`configs/detectors/rotating_shield_counter.physical.synthetic.yaml`. For each
commanded posture, the runtime applies the actual posture to the configured USD
rotation operation, synchronizes the material-tagged shield geometry into
Embree, and obtains the binned rate from the common radiation/detector path.
Commanded, actual, and encoder angles are recorded separately. The legacy
response-mask configuration remains an explicitly synthetic approximation and
is not used as evidence for physical-geometry synchronization.

### 2. Operation-level validation

- shield and object action success, grasp success, final root-pose error,
  collision violations, and disposal containment;
- treated-area coverage, contact-qualified dwell, removed activity, waste
  activity, and activity-balance error;
- geometry/source revision changes and subsequent detector-response changes;
- operation duration and resource consumption.

### 3. Integrated scenario

The fixed articulated scenario performs an initial multi-station
omnidirectional survey, dry wall treatment, shield placement and correction,
contaminated-drum relocation and shielded disposal, obstacle relocation, and
final verification measurement. The rotating-shield angular program is
evaluated in a separate detector component experiment. Report:

- spectral/count-rate error and angular-response error;
- action success fraction and per-operation final-pose error;
- treatment coverage and activity-balance error;
- pre/post primary, corrected-source, background, and total detector components;
- measured/estimated residuals and remaining uncertainty;
- measurement time, work time, robot runtime, shield/media/water/waste capacity,
  and operation count used;
- wall time, simulated action time, rays, native trace calls, cache hits, and
  selective ray updates;
- mean, spread, and confidence intervals across the declared fixed seeds.

The total detector rate alone is not interpreted as countermeasure effectiveness
when background or detector relocation dominates. Source components and
like-for-like detector poses must be reported separately.

## Evidence and data gates

`synthetic_validation_only` runs answer whether state transitions, physical
execution, balances, synchronization, and artifact generation behave as
implemented. They cannot establish real material, detector, or treatment
accuracy.

`research_evaluation` requires all of the following and fails if any item is
missing or out of range:

1. authoritative material attenuation and isotope emission data;
2. calibrated response data for each evaluated detector;
3. a reference-calibrated buildup surface if corrected photon results are used;
4. material/tool-specific treatment parameters for quantitative removal claims;
5. exact SHA-256 agreement between validated files and runtime values;
6. no extrapolation outside validated energy or optical-depth ranges.

## Work remaining before the scientific paper release

No additional general-purpose simulator subsystem is planned. The remaining
work is to acquire and validate the exact paper evidence:

1. calibrate the two main detector configurations in a controlled setup;
2. build the independent photon-reference benchmark and correction table;
3. identify dry-treatment parameters for the chosen surface/tool pair;
4. execute controlled pre/post countermeasure measurements in a representative
   mock-up;
5. run the frozen scenario across the declared seeds and replace qualitative
   claims with numerical uncertainty intervals.

Future work in the paper should therefore describe broader experimental
calibration and representative mock-up validation, rather than promise a longer
list of detector or operation types that the platform already supports.
