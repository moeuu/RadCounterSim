# Large environment streaming

RadInterAct stores large environments as spatial tiles with independent LOD
payloads. The complete site is never required in the Isaac Sim working set.

## Data path

1. Import OBJ, FBX, glTF, CAD, point-cloud, USD, or a simulator-exported mesh
   through the normal environment importer.
2. Run radcounter-build-large-environment against the normalized manifest.
3. Generate the lightweight root USD with LargeEnvironmentUsdWriter.
4. Open the root USD and start LargeEnvironmentRuntime on the Kit main thread.

Example preprocessing:

~~~bash
uv run radcounter-build-large-environment \
  .cache/environments/large_facility/manifest.json \
  --output .cache/environments/large_facility/streaming \
  --tile-size 25 \
  --physics-radius 40 \
  --visual-radius 250
~~~

Example Isaac setup:

~~~python
from radcounter.isaac.usd.large_environment import LargeEnvironmentUsdWriter
from radcounter.isaac.usd.streaming_runtime import LargeEnvironmentRuntime

LargeEnvironmentUsdWriter().write(
    ".cache/environments/large_facility/streaming/streaming-index.json",
    ".cache/environments/large_facility/large_facility.usda",
)
runtime = LargeEnvironmentRuntime.discover(omni.usd.get_context().get_stage())
runtime.start()
~~~

## Working-set policy

- Tiles inside physics_radius_m use LOD0 with collision and radiation
  transport enabled.
- More distant visible tiles use progressively coarser deterministic LODs.
- Tiles outside visual_radius_m are unloaded.
- unload_hysteresis_m prevents churn at distance boundaries.
- max_loaded_tiles and max_loaded_triangles bound normal memory use.
- max_changes_per_update prevents frame spikes from USD resynchronization.
- Robot and detector prims tagged rad:stream:focus=true drive the working
  set automatically.

Radiation is not allowed to use only the visual neighborhood. Before an Embree
batch, pass every finite source-detector segment to pin_radiation_segments.
Every intersected tile is forced to LOD0 until that transport batch completes.
RadiationTileLease provides this lifetime as a context manager.

## Scale limits

Runtime display and processing are out-of-core. The first conversion still
uses the source format reader's memory behavior. For city-scale data, export
pre-tiled USD/3D-Tiles-style chunks or convert each survey/CAD sector
separately, then combine their streaming indexes. Native USD payload datasets
can also bypass monolithic mesh conversion.

The tile cache uses compressed NPZ as an interchange representation and USD
payloads for Isaac. The core index and selector are simulator-independent, so
another renderer or an Embree-only process can consume the same working set.
