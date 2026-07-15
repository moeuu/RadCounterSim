# Isaac Sim extension

`radcounter.isaac` is an independent Kit extension for Isaac Sim 6.x. It owns a
small host window, USD stage lifecycle, timeline controls, full three-dimensional
shield/object poses, and conversion of tagged USD meshes into the Embree transport
scene.

Add `source/extensions` to the Isaac Sim extension search path and enable
`radcounter.isaac`. The portable `radcounter` package and optional native module
must be visible to Isaac's Python process. Build the native module with
`scripts/build_native.sh`; use `uv` rather than unmanaged `pip` for Python tooling.

USD meshes participating in attenuation must define the custom string attribute
`radcounter:materialId`. Its value must be present in the material-index mapping
provided to `UsdEmbreeSceneAdapter.rebuild()`.

The extension declares only direct Kit dependencies. PhysX remains optional because
stage inspection and radiation-only runs are valid without physics execution.
