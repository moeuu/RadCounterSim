# Manchester nuclear simulation assets

RadInterAct can use the University of Manchester's **3D Simulation Assets
for Nuclear Environments [Gazebo Format]** as a high-detail visual validation
scene.

- Article: <https://figshare.manchester.ac.uk/articles/software/25224974>
- DOI: `10.48420/25224974.v1`
- License: Creative Commons Attribution 4.0 International (`CC BY 4.0`)
- Used scene: `500L_Drum_Store.zip`

These assets are realistic authored simulation models inspired by nuclear
facilities. They are **not an as-scanned record of a named operating nuclear
facility**. The JSI TRIGA 2026 dataset is used separately when measured point
cloud and co-registered radiation data are required.

Fetch and extract the scene without committing third-party binary data:

```bash
uv run python scripts/fetch_manchester_nuclear_assets.py
```

Run the Isaac Sim 6 GUI validation:

```bash
scripts/host_env.sh /home/moeu/.local/isaacsim/6.0.1-uv/.venv/bin/python \
  scripts/isaac_manchester_drum_store_validation.py --duration 30
```

The loader preserves the SDF model/link/visual transforms, per-mesh scale, and
BaseColor, Normal, Roughness, and Metallic maps. Converted USD files are cached
under `.cache/usd/manchester-500l/`.
