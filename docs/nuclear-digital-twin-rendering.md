# Nuclear facility digital-twin rendering

## What OceanSim does

OceanSim does not replace Isaac Sim with a separate offline renderer. It uses
Isaac Sim/Omniverse rendering, real photogrammetric digital twins, calibrated
sensor image formation, and RTX/Replicator products. RadCounterSim follows the
same division: the facility asset supplies geometric and texture realism, while
the simulator supplies physically meaningful lighting, sensor effects, robot
motion, and radiation state.

A procedural room made from a few boxes cannot become a convincing facility by
turning on RTX. Use a survey-derived visual asset whenever visual fidelity is an
experimental requirement.

## Geometry contract

The visual twin and the physical twin are deliberately separate.

| Layer | Preferred source | Purpose |
| --- | --- | --- |
| Visual | textured metric USD from photogrammetry or DCC | RGB and human inspection |
| Collision | decimated watertight mesh | PhysX and robot planning |
| Radiation | material-labelled mesh | Embree attenuation and source transport |
| Navigation | occupancy/semantic map | Nav2 and autonomous exploration |

`asset.visual_uri` accepts native USD directly. OBJ, FBX, glTF, GLB, DAE, STL,
and PLY are converted by Omni Asset Converter with materials and external
textures preserved. A RadCounterSim environment `manifest.json` can be used as a
visual fallback, but it has geometry and PBR fallback materials rather than the
original photogrammetry texture set.

STEP, IGES, BREP, IFC, E57, LAS, LAZ, PCD, Gazebo SDF, URDF, Xacro, and OctoMap
continue through the format-neutral environment importer. Normalize them first:

```bash
uv run radcounter-import-environment configs/environments/facility.yaml
```

Use the resulting manifest for collision/radiation and export or convert the
high-detail source to a textured USD for `visual_uri`. This avoids using a
multi-million-triangle scan as a collision mesh.

## Large facilities

The existing large-environment builder partitions physical geometry into
independent tile payloads and creates distance LODs. The visual USD should use
the same tile grid and package each tile as a USD payload. Repeated pipe
supports, handrails, trays, lights, drums, and shield modules should be authored
as references and matched by `instanceable_patterns`.

Runtime memory is bounded by all of the following:

- USD payload tile residency and distance LOD
- instanceable repeated references
- RTX texture streaming with a tier-specific memory budget
- camera resolution and product-rate scaling
- bounded dust, steam, and spray particle counts
- Path Tracing only for a requested capture on sufficiently strong GPUs

Radiation rays pin intersected physical tiles at LOD0. Lowering visual quality
never changes source activity, Embree geometry, detector truth, depth truth, or
the planner state.

## PBR materials and lighting

[`configs/materials/nuclear_pbr.yaml`](../configs/materials/nuclear_pbr.yaml)
defines concrete, epoxy floor, wet epoxy, stainless and painted steel, lead,
rust, dust, rubber, and contaminated water. Native source materials are
preserved by default. Binding rules only supply materials to unbound normalized
geometry unless `override_existing` is explicitly enabled.

Facility lights support dome/HDRI, rectangular luminaires, disks, spheres, and
distant lights. Record fixture position, illuminance, correlated color
temperature, camera exposure, and white balance during the facility survey;
store the survey revision in `lighting.calibration_id`.

## Radiation camera response

The RGB camera postprocessor is stateful and models:

- dose-rate-dependent transient bright events
- vertical readout streaks and local saturation/blooming
- color loss and increased read noise
- probabilistic dropped/held frames
- cumulative permanent hot pixels from total absorbed dose

The effect is applied to the RGB render product only. Depth, semantic labels,
LiDAR, and radiation products remain separate. Coefficients must be fitted to
the selected camera under the intended dose field; defaults are demonstrative,
not detector calibration data.

## Decontamination visuals

Water spray is rendered as a bounded point product aligned with the real nozzle
axis. Wet traces are thin PBR ribbons on contacted surfaces. Fog and dust are
quality-gated; weak GPUs disable volumetric fog and retain a sparse spray so the
operator can still understand the action. These visuals consume decontamination
truth from the existing tool/contact model. They do not replace fluid dynamics
or calculate removal efficiency from appearance.

## Independent products

`IsaacRenderProductManager` supports RGB, depth, normals, semantic labels,
instance labels, LiDAR, and radiation products. RGB/depth/labels use Replicator
annotators. LiDAR and radiation use registered providers, allowing the existing
RTX LiDAR rig and radiation detector backend to remain authoritative.

## GPU policy

| Tier | Automatic threshold | Interactive | Capture | Main reductions |
| --- | --- | --- | --- | --- |
| strong | RTX and at least 16 GB | RTX Real-Time | Path Tracing, 64 spp | none |
| balanced | RTX and at least 8 GB | RTX Real-Time | Path Tracing, 16 spp | 0.85 resolution, half effects |
| weak | RTX and at least 3.5 GB | RTX Real-Time | RTX Real-Time | 0.65 resolution, no volumes, LOD bias |
| fallback | no detected RTX | Storm | Storm | 0.5 resolution, sparse effects |

Set `RADCOUNTER_RENDER_TIER` to force a tier. For reproducible benchmark runs,
also record GPU name, VRAM, driver, selected tier, actual frame-time histogram,
resident tile count, and texture memory.

## Run

Set the real facility asset in the example descriptor and launch with the Isaac
Sim 6.0 Python environment:

```bash
export RADCOUNTER_FACILITY_USD=/absolute/path/to/facility.usd
/home/moeu/.local/isaacsim/6.0.1-uv/.venv/bin/python \
  scripts/isaac_nuclear_digital_twin.py \
  configs/rendering/nuclear_digital_twin.example.yaml
```

For a converged still or reference sensor frame on a strong GPU, add
`--capture`. Interactive robot operation remains RTX Real-Time.

## Quantitative calibration

Capture real and simulated RGB/depth frames from matched metric camera poses.
Lock exposure and white balance, align images with surveyed fiducials, mask
dynamic objects, and evaluate RGB MAE/RMSE/PSNR/global SSIM/edge error/color
histogram divergence plus depth MAE/RMSE/absolute-relative/delta metrics using
`compare_rgb` and `compare_depth`.

Do not tune and report on the same frames. Fit lighting, PBR, atmosphere, and
camera-radiation coefficients on calibration poses, then report held-out poses,
dose rates, wet/dry states, and robot viewpoints. Store the descriptor, USD
digest, reference-frame digest, renderer tier, and random seed with every run.
