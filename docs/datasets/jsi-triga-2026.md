# JSI TRIGA Mark II measured survey

RadInterAct uses the public May 2026 survey of the Jožef Stefan Institute
TRIGA Mark II research reactor as its measured nuclear-facility validation
scene. The survey was acquired by a modified Clearpath Jackal and includes a
53,529-point colored 3D LiDAR map, a 2D SLAM map, and co-registered radiation
counts from a Scionix CeBr3 detector.

Source:

- Seb Oakes, University of Manchester, DOI `10.48420/32727696.v1`
- <https://figshare.manchester.ac.uk/articles/dataset/32727696>
- License: BSD 3-Clause

The third-party files are downloaded to `.cache` and are not committed.

```bash
scripts/fetch_jsi_triga_dataset.sh
```

The PCD comes from PCL and contains repeated `_` padding fields plus a trailing
reserved region. `load_pcd_xyz_rgb` bounds binary records using the declared
point count, makes padding names unique, and decodes packed RGB.

Run the Isaac Sim 6.0.1 GUI validation:

```bash
/home/moeu/.local/isaacsim/6.0.1-uv/.venv/bin/python \
  scripts/isaac_jsi_triga_validation.py --capture --duration-s 20
```

The GUI overlays three independently traceable products:

- the measured colored LiDAR cloud
- a concrete PBR surface proxy generated from the measured points
- the co-registered measured radiation path

The referenced Clearpath Jackal follows the measured detector route. The output
under `artifacts/jsi_triga_gui_validation` records the screenshot, renderer
tier, frame-time distribution, point/sample counts, scene bounds, and source
manifest.

TEPCO has publicly documented Fukushima Daiichi point-cloud acquisition and
digital-twin use, but does not currently provide the underlying point cloud with
a reusable data license. TEPCO images and PDFs must therefore be treated as
visual references, not extracted as simulation assets.
