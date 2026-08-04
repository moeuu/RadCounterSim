# Large-environment endurance benchmark

`scripts/isaac_large_fleet_endurance.py` is a GUI endurance gate for a
heterogeneous radiation-survey fleet. Its default run lasts 600 wall-clock
seconds rather than advancing 600 simulated seconds as fast as possible.

## Workload

- 4 km x 4 km logical environment, divided into 10,000 deterministic tiles.
- UGV and drone follow independent routes for approximately 1.8 km and 4.8 km.
- RTX LiDAR, RGB and depth camera acquisition, and two radiation detectors.
- Lead-shield response after 200 seconds and water-decontamination response
  after 400 seconds.
- GUI telemetry, start/middle/final captures, and JSON performance metrics.

The weak-GPU profile keeps only the nearest 50 tiles resident. Ground,
obstacles, and source markers use USD PointInstancer prototypes, so geometry
memory is bounded by the active working set rather than logical map area.

The adaptive governor changes only representation and update rates. It never
changes source truth, robot pose, detector calibration, or mitigation truth.
This separation is required so performance adaptation cannot alter an
experiment's scientific result.

## Run

~~~bash
ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES \
PYTHONPATH="$PWD:$PWD/source/extensions/radcounter.isaac" \
/home/moeu/.local/isaacsim/6.0.1-uv/.venv/bin/python \
scripts/isaac_large_fleet_endurance.py \
  --profile configs/performance/weak_gpu.yaml \
  --duration 600 \
  --output .cache/large-fleet-endurance
~~~

The acceptance gate requires at least 10 average FPS, bounded tile residency,
successful LiDAR/RGB/depth reads, radiation measurements, no sensor errors,
and at least 98 percent of the requested wall-clock duration.

The profile is intentionally conservative: 960 x 540 viewport, 640 x 360
camera, 5 Hz RTX sensors, disabled point-cloud drawing, and adaptive LOD. It is
a workload profile, not proof that a specific low-end GPU is supported. Final
hardware qualification must run the same gate on each target GPU and driver.

## Automatic hardware profiles

Use `--profile auto`, or omit the option, to query GPU VRAM through
`nvidia-smi` and select one of four profiles:

- `cpu_fallback`: less than 4 GB or no NVIDIA GPU;
- `weak_gpu`: 4 GB to less than 8 GB;
- `balanced_gpu`: 8 GB to less than 16 GB;
- `strong_gpu`: 16 GB or more.

Set `RADCOUNTER_GPU_TIER=weak_gpu` to force a tier during qualification. Static
profile selection controls startup resolution, resident-tile cap, and native
sensor tick rates. The runtime governor then reacts to measured frame time and
adjusts LOD, active radius, read rates, and environment update rates. This
allows the same scenario and scientific truth to run on different machines.
