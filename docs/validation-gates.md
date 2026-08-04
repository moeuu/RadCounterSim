# External and physical validation gates

These gates keep software-loopback, workload-profile, and physical evidence separate.
A synthetic or emulated run can validate plumbing, but it cannot satisfy a physical gate.

## Flight stacks and rotor dynamics

Run the pinned PX4 SIH image and the official ArduCopter 4.6.3 SITL binary:

```bash
uv run python scripts/validate_sitl.py --stack px4 --duration-s 20
uv run python scripts/validate_sitl.py --stack ardupilot --duration-s 20
```

Both commands require a heartbeat, attitude and position streams, arming, and at least
0.5 m of vertical motion. They also run the deterministic motor-lag and rigid-body gate.
Their evidence is SITL only and never qualifies as a physical flight test.

## ROS 2 sensor bandwidth

```bash
uv run python scripts/validate_ros2_bandwidth.py --duration-s 30
```

The host wrapper uses the pinned official ROS 2 Jazzy image. Separate publisher and
subscriber processes exchange PointCloud2, RGB Image, and TF messages through DDS.
Per-topic delivery, payload bandwidth, and p50/p99 latency are written to
`.cache/ros2-bandwidth/metrics.json`.

## Detector HIL

The protocol is canonical JSON Lines with a protocol version, sequence number, device
clock, optional host timestamp, a complete DetectorReading, and CRC32. TCP and serial
transports feed the existing ExternalDetectorReadingBuffer.

```bash
uv run python scripts/validate_detector_hil.py --mode loopback --duration-s 5
uv run python scripts/validate_detector_hil.py \
  --mode serial --device /dev/ttyACM0 --duration-s 600 --require-physical
```

Loopback injects one sequence gap and one CRC fault and requires both to be detected.
Only a serial run with explicit hardware serial and calibration identifiers qualifies
as physical. TCP remains external-live evidence and cannot satisfy the physical gate.

## Source and shielding physics

```bash
uv run python scripts/validate_physics_model.py \
  observations.csv metadata.yaml --require-measured
```

The CSV supports absolute source response, inverse-square response, and material
attenuation. Acceptance uses Poisson uncertainty, normalized residuals, reduced
chi-square, and fitted attenuation-coefficient error. The supplied Cs-137/lead file is
an analytic pipeline fixture and is deliberately rejected by `--require-measured`.

## Physical 4 GB and 8 GB GPU gates

The workflow `external-and-hardware-validation` is manual because it targets registered
self-hosted machines. Each runner must carry an actual VRAM label and provide
`RADCOUNTER_ISAAC_ENDURANCE_COMMAND`, which must write the endurance metrics file.
The post-gate independently reads nvidia-smi and rejects a profile run performed on a
different memory class. Runner labels and post-gates also distinguish GPU count, OS,
and R550/R580 driver tracks.

```bash
uv run python scripts/validate_hardware_gate.py \
  --vram-class 8gb --expected-os linux \
  --metrics .cache/large-fleet-endurance/metrics.json
```

Record the GPU model, total memory, driver, OS, Isaac Sim build, full log, and metrics.
Never relabel a 32 GB run as 4 GB or 8 GB evidence; renderer settings do not impose a
physical VRAM ceiling.
