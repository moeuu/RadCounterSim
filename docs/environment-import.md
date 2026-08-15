# External 3D environments

RadCounterSim accepts simulator-neutral maps and CAD geometry. Every imported
environment is normalized to metres, a right-handed coordinate system, and
Z-up before the same triangles are sent to USD rendering, static PhysX
collision, and Embree radiation transport.

## Supported inputs

| Input family | Formats | Adapter |
|---|---|---|
| Native scene | USD, USDA, USDC, USDZ | Non-destructive USD reference |
| Mesh exchange | glTF, GLB, OBJ, STL, PLY, DAE, 3MF, OFF | `trimesh` |
| Robotics world | SDF, Gazebo `.world`, URDF, Xacro | XML hierarchy and URI resolver |
| CAD/B-rep | STEP/STP, IGES/IGS, BREP | Gmsh/OpenCASCADE tessellation |
| Point cloud | PCD (ASCII/binary), XYZ/PTS, point-only PLY | Occupied-voxel surface mesh |
| Proprietary/external | FBX, IFC, E57, LAS/LAZ, OctoMap, other | Configured converter or plugin |

FBX is offered to `trimesh` first. Installations without an FBX backend must
configure an external converter. Binary-compressed PCD similarly requires a
conversion to PLY or uncompressed PCD.

Install the standard import stack and the CAD adapter through `uv`:

```bash
uv sync --all-groups
# Equivalent CAD-only addition:
uv sync --group cad
```

Do not install Gmsh or mesh readers into the system Python. When an Isaac Sim
Python environment lacks the CAD package, the extension invokes the locked
project `uv` environment and consumes its normalized manifest.

## Direct use

Import a file outside Isaac Sim to inspect its resolved geometry:

```bash
uv run radcounter-import-environment plant.glb
uv run radcounter-import-environment cell.step --units mm --material steel
uv run radcounter-import-environment room.world
```

The command prints the manifest path and writes these files under the configured
cache directory:

- `manifest.json`: source digest, dependencies, format, bounds, material IDs,
  collision flags, warnings, and the fully resolved import configuration.
- `normalized_meshes.npz`: vertices in metres and triangle indices without
  pickle data.
- `environment.usda`: generated lazily when Isaac Sim loads the result.

Named system profiles can combine an environment descriptor with independently
selected robot and detector sets. See `docs/system-profiles.md` and use
`uv run radcounter-system list`. The Fukushima Daiichi SolidWorks integration
also documents its native SolidWorks-to-USD conversion there. On Linux this is
performed by the HOOPS Exchange converter bundled with Isaac Sim; `.SLDASM`
and `.SLDPRT` are not passed to the simulator as if they were portable meshes.

In the RadCounterSim Operations window, operators select an environment, robot
set, and detector set by display name and press **選択した構成を適用 / Apply**.
No catalog ID, file path, or CLI knowledge is needed for normal operation.
Invalid detector/robot combinations are rejected before the stage changes.
The separate LLM field controls robot tasks; it does not interpret or change
the system-selection controls. Catalog environments may declare reviewed local
preparation scripts; when their asset is missing the Apply button becomes a
one-click **Prepare** action instead of asking the operator to run commands.

A direct USD path preserves the existing stage and prim paths. Put an arbitrary
untagged USD asset in a descriptor when it should be wrapped with default
collision and radiation-material metadata. Existing USD-only workflows are
unchanged.

## Descriptor

Use `configs/environments/vertical_slice_import.yaml` as the complete example.
The essential form is:

```yaml
schema_version: "1.0"
environment:
  environment_id: reactor-cell
  uri: /data/maps/cell.step
  format: step
  coordinate_system:
    units: mm
    up_axis: "+Z"
    forward_axis: "+X"
    handedness: right
  default_material_id: concrete
  material_rules:
    - pattern: "*lead-wall*"
      material_id: lead
  collision:
    enabled: true
    geometry_source: auto
    approximation: triangle_mesh
  cad_mesh_size_m: 0.025
```

The same `environment` object may be embedded in a normal scenario YAML.
Relative paths resolve from that YAML file, not from the process working
directory.

## Gazebo and ROS assets

SDF includes and mesh URIs support ordinary relative paths and `model://`.
Search roots come from `model_search_paths`, `GZ_SIM_RESOURCE_PATH`, and
`GAZEBO_MODEL_PATH`. URDF meshes also support `package://`; roots come from
`package_search_paths`, `ROS_PACKAGE_PATH`, and `AMENT_PREFIX_PATH/share`.

URDF joint trees are imported at zero joint position. SDF model, link, visual,
collision, nested-model, and include transforms are composed. In `auto` mode,
authored collision geometry drives PhysX and Embree when available; otherwise
visual geometry is used. Visual geometry remains visible without being counted
twice by radiation transport.

## Coordinate and material rules

Automatic conventions are format-specific. Robotics XML and CAD default to
Z-up/+X-forward. glTF and FBX default to Y-up/-Z-forward. Formats without a
reliable convention should set units and axes explicitly. A left-handed input
is reflected and its triangle winding is reversed.

Visual material names do not define gamma attenuation. `material_rules` maps
mesh, node, or prim names to RadCounterSim material IDs; unmatched geometry uses
`default_material_id`. This prevents a visually grey CAD wall from silently
being treated as air.

## External converters and plugins

An external converter is an argument list, never a shell string:

```yaml
external_converter:
  command: [my-scene-converter, "{input}", "{output}"]
  output_format: glb
  timeout_s: 900
```

The placeholders are mandatory. For reusable Python integrations, publish an
entry point in the `radcounter.environment_importers` group. Its loaded object
must construct an importer implementing `formats` and `load(source, context)`.
Applications may also call `EnvironmentImportPipeline.register_importer()`.

This extension point is the escape hatch for vendor CAD, BIM, mapping, and
proprietary simulator formats without adding simulator-specific code to
`radcounter.core`.

## Environments larger than memory

Set `environment.streaming.enabled: true`, import the source, then run `radcounter-build-large-environment` on its normalized manifest. The resulting tile index is simulator-independent; Isaac uses USD payloads while Embree transport can request the same LOD0 tiles. Full policy and limits are documented in `docs/large-environments.md`.
