# Definition of Done

## Simulator implementation

- [x] USD scene state drives rendering, collision, path-length calculation,
  source state, and visible per-face activity.
- [x] Primary and reference-corrected photon paths share one response pipeline;
  neutron and charged-particle reduced-order kernels use the same scene paths.
- [x] Point, irregular surface, and volume sources are supported.
- [x] Eighteen built-in detector models and custom/external detector adapters
  consume transported incident fluence rather than private shielding models.
- [x] The physical rotating-shield mode moves its material-tagged USD geometry,
  synchronizes Embree at each actual posture, and records commanded, actual,
  and encoder angles separately; its direct-pose gate is labeled
  `kinematic_scene_edit`.
- [x] Dry-contact and water treatment track activity and material/resource
  balances on the displayed activity-bearing mesh.
- [x] Shield placement/correction, object relocation, and shielded disposal have
  articulated robot execution and post-settle validation.
- [x] Continuous and particle-assisted source estimation, uncertainty,
  residual analysis, resource-aware action selection, application controls,
  ROS 2 adapters, and generic robot descriptions are implemented.
- [x] Kinematic scene edits and physical robot execution use distinct evidence
  classes with no automatic substitution.

## Paper artifact acceptance

- [x] Every run records its seed, evidence class, stage/config/data hashes, and
  runtime versions.
- [x] Every required operation is present and successful.
- [x] Physical evidence uses native Embree and reports nonzero simulated action
  time.
- [x] Final pose error, contact coverage, activity balance, spectra, residuals,
  resource use, and transport/cache counters are finite and present.
- [x] Physical artifacts record Python, package, Isaac Sim, Embree, renderer,
  GPU, memory, and driver identity; artifacts from different execution
  runtimes cannot be pooled silently.
- [x] Multiple artifacts can be aggregated only when their stage,
  configuration, physics class, and required data bundle agree.
- [x] A research run cannot use synthetic fixtures or silently fall back to
  them.

## Scientific paper release

- [ ] Load authoritative, provenance- and hash-checked material and isotope
  data for the exact evaluated energy range.
- [ ] Load calibrated omnidirectional and rotating-shield detector responses.
- [ ] Calibrate any enabled buildup correction against an independent photon
  transport reference over the evaluated materials and optical depths.
- [ ] Identify dry-treatment parameters for the selected material/tool pair
  using controlled measurements.
- [ ] Compare simulated detector observations with controlled measurements and
  report uncertainty.
- [ ] Execute the fixed scenario for the declared seed set and report
  distributional metrics rather than a single seed.
- [ ] Record the target GPU, Isaac renderer, Embree, and host versions in the
  release manifest.

Unchecked scientific items block quantitative accuracy claims, but do not
invalidate synthetic implementation-verification artifacts.

## Release checks

```bash
uv lock --check
uv run ruff check .
uv run python -m pytest -q
export RADCOUNTER_HOST_ENV_NO_ROS=1
source scripts/host_env.sh
PYTHONPATH="$PWD/build/native/python:$PWD" \
  uv run python -m pytest -q tests/integration/test_embree_runtime.py
uv run python scripts/check_regression.py --seed 42
uv run python scripts/audit_host_gates.py --require-all
```

The articulated paper command and collector are documented in
[`experiment-artifacts.md`](experiment-artifacts.md).
