# Definition of Done

## Portable core

- [x] Python dependencies are locked and executed with `uv`.
- [x] Truth state and planner-visible belief state are separate types.
- [x] Analytic radiation, counting, dead-time, cache, and dose-map paths exist.
- [x] Decontamination, shielding, object motion, waste, and resource semantics exist.
- [x] Estimation, uncertainty, residual diagnosis, and belief update exist.
- [x] Open-loop, greedy, nearest, random, oracle, and residual closed-loop planners exist.
- [x] Closed-loop workflow supports pause, resume, stop, termination reasons, and snapshots.
- [x] Seven deterministic experiment cases emit reproducibility artifacts.
- [x] Regression bounds and a portable benchmark command are versioned.

## Host integrations

- [x] Isaac Sim runtime gate: load the extension in a supported Isaac Sim installation.
- [x] Embree runtime gate: build and execute native occlusion queries against Embree 4.
- [x] Physics runtime gate: execute shield placement and object motion in PhysX.
- [x] ROS 2 runtime gate: build the Jazzy workspace and pass message/action/service round trips.
- [ ] GPU runtime gate: record the target GPU and Isaac renderer versions in a release manifest.

The first four host gates passed on the local Isaac Sim 6.0.1, Embree 4.3, and
ROS 2 Jazzy host. The canonical release-manifest GPU/renderer record remains
open; runtime reports alone do not satisfy that release-artifact requirement.

Run the portable release checks with:

```bash
uv lock --check
uv run ruff check .
uv run pytest
uv run python scripts/check_regression.py --seed 42
uv run python scripts/audit_host_gates.py
```

On a release host, add `--require-all` to the final command. A nonzero result means
the release cannot claim the Isaac/Embree/ROS/GPU integration gates.
