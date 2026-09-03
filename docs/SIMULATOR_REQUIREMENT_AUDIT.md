# RadInterAct simulator requirement audit

Audit date: 2026-08-29

## Verdict

The requested simulator capabilities are implemented. The remaining work is
scientific calibration and data collection, not another software feature
expansion. A capability is therefore reported with two independent statuses:

- **Implemented** means that the code path, validation, and artifact contract
  exist.
- **Research-ready** means that the exact run uses authoritative or measured
  inputs whose provenance and hashes satisfy the research-data gate.

The checked-in vertical-slice configuration is intentionally
`synthetic_validation_only`. It demonstrates implementation consistency and
physical scene execution; it is not evidence of detector or material accuracy.

## Final capability audit

| Capability | Implementation | Research boundary |
| --- | --- | --- |
| Shared mutable scene | Complete | USD is the authored source of geometry, material labels, source metadata, and per-face activity; PhysX and Embree consume synchronized derivatives. |
| Point, surface, and volume sources | Complete | Surface treatment, visualization, and radiation sampling use the same irregular activity-bearing mesh. |
| Photon transport | Complete | Energy-dependent primary gamma/X-ray attenuation is implemented. A corrected run must load a hash-bound buildup surface covering every traversed material, energy, and optical depth; missing coverage fails. |
| Other radiation types | Complete | Neutron removal and alpha/beta range kernels share the scene path provider. They are reduced-order models and retain their data classification. |
| Detector response | Complete | Incident transported fluence is passed through effective-area response curves/matrices, background, angular response, energy redistribution, dead time, saturation, and Poisson sampling. Detector code cannot perform private transport. The physical rotating-shield path authors the configured USD posture, synchronizes its tagged attenuation mesh, and then calls this same response path. |
| Detector catalog | Complete | Eighteen built-in models cover nine families, six directionalities, gamma/X-ray, neutron, alpha, and beta responses; custom and external-reading adapters use the same schema. |
| Dynamic radiation updates | Complete | Transform, material, source-pose, source-presence, and per-face activity changes invalidate only affected cached rays and update subsequent observations. |
| Dry surface treatment | Complete | Contact distance, surface normal, speed, footprint, and dwell control a material-parameterized removal model with explicit waste activity and balance checks. |
| Water treatment | Complete | A reduced-order nozzle/wetting/removal/recovery/runoff model tracks clean water, wastewater, redeposition, discharge, and activity balance on the visible irregular mesh. It does not claim CFD fidelity. |
| Shield placement and correction | Complete | An articulated mobile manipulator grasps, transports, releases, and settles the shield; its settled pose changes Embree paths. |
| Object relocation and disposal | Complete | Per-object grasp, stand-off, parking, and root-placement metadata replace array-order conventions. Disposal retains the source inside explicit shielding instead of deleting it. |
| Physical execution | Complete | Ridgeback + Franka and Nova Carter use wheel and arm joints, Lula IK, gripper closure, payload constraints, contact, collision, and settling checks. |
| Kinematic scene editing | Complete and isolated | Direct USD pose/source edits are valid for `kinematic_scene_edit` experiments only. They are never relabeled or automatically substituted for `physical_robot_execution`. |
| Feasibility and resources | Complete | Live-scene reachability, collision, route, stability, disposal containment, time, runtime, shield, media, water, wastewater, and operation-count constraints fail closed. |
| Estimation | Complete | Grid Poisson/L1/TV, Fisher/bootstrap uncertainty, continuous position/activity MLE, particle-assisted MLE, and a public-observation Isaac estimator are implemented. Hidden simulator sources are rejected. |
| Action selection | Complete | Deterministic resource- and risk-aware selection plus versioned comparison policies are implemented. Candidate generation uses public estimates and live scene contracts. |
| Application integrations | Complete | Dashboard visualization/controls, strict local natural-language commands, ROS 2 adapters, external detector readings, environment import, and generic robot descriptions are implemented. |
| Evidence artifacts | Complete | Seed, evidence class, stage/config/data hashes, operation results, spectra, estimation residuals, resources, timing, transport counters, activity balances, and the executing Isaac/Embree/renderer/GPU/driver versions are recorded. |
| Paper collector | Complete | The collector rejects failed/incomplete runs, mixed evidence, duplicate seeds, stage or data hash mismatches, synthetic inputs in research mode, missing operation classes, non-native transport, invalid balances, all-zero physical execution time, missing runtime identity, and pooled runs from different execution runtimes. |

## Paper-facing decision

The paper evaluates the scene-consistent simulator, not every optional platform
adapter. Its main articulated scenario contains:

1. Cs-137 gamma measurement with an omnidirectional detector in the articulated
   scenario and a separate physical-geometry rotating-shield angular program;
2. contact-conditioned dry treatment of an irregular wall source;
3. shield placement and repositioning;
4. contaminated-object relocation and shielded disposal;
5. a final measurement after the scene and activity changes.

Water treatment, additional particles and detector models, source-estimation
variants, action-selection variants, natural-language control, ROS 2, and
generic robots remain implemented extension capabilities. They enter the main
paper only if their own calibrated inputs and quantitative evaluation are
completed; otherwise they are reported as platform support, not as validated
scientific results.

The complete scope and metric contract is recorded in
[`paper-simulator-scope.md`](paper-simulator-scope.md).

## Current executable evidence

- Portable Python suite: 277 passed and 6 native-only cases skipped in the
  portable environment. The same 6 Embree integration cases passed after the
  native host environment was loaded.
- Twelve Isaac gates cover extension load, dynamic scene synchronization,
  physical rotating shielding, performance, visualization, dashboard controls,
  workflow services, physics actions, articulated robots, and object
  manipulation.
- The rotating-shield gate changed the same lead mesh at all four postures.
  Its direct-geometry component evidence is labeled `kinematic_scene_edit`;
  the shielded posture produced 4.103 cps versus 360,752.329 cps at the three
  unobstructed postures under synthetic inputs.
- The seed-11 articulated scenario completed all eight operation phases and 11
  physical action records with `physical_robot_execution` evidence. It reports
  228.150 s of simulated action time, maximum final-pose error of 0.002470 m,
  treatment coverage of 0.780684, 4,453,052.664 Bq removed, and activity-balance
  error of -4.657e-9 Bq.
- Native transport in that run recorded 95 trace calls, 13,999 rays, 532 cache
  hits, and 6,016 selectively updated rays.
- The paper collector accepted the runtime-identified synthetic artifact at
  `artifacts/gui-validation/paper-seed-11-runtime.json` and wrote immutable
  outputs under `artifacts/paper-evaluation/synthetic-seed-11-runtime/`.
- The execution manifest records Isaac Sim 6.0.1.0, Embree 4.3.0,
  `RealTimePathTracing`, an NVIDIA GeForce RTX 5090, and driver 580.173.02.

This evidence establishes software and simulated-operation behavior. Quantified
scientific accuracy still requires the research-calibration work listed in the
scope document. The accepted paper bundle contains one seed and therefore does
not establish repeatability or sampling uncertainty.
